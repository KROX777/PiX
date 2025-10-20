import pprint
import numpy as np
from joblib import Parallel, delayed
import time as time_module
import random
from pix.hypotheses_tree import LightTree
from pix.hypotheses_tree import HypothesesTree
from pix.utils.others import *
import sympy as sp
import scipy
import torch
import yaml
import os
import sys
from pathlib import Path
import logging

# 全局缓存：避免每次都重新加载权重/初始化模型
_SR4MDL_WEIGHT_STATE = None  # 过滤后的 state_dict
_SR4MDL_WEIGHT_META = None   # (weights_path, model_keys_hash)
_SR4MDL_DEVICE = None
_SR4MDL_IMPORT_DONE = False

class EarlyStopException(Exception):
    """Custom exception to signal early stopping"""
    pass

def parse_custom_operators(custom_binary_ops='', custom_unary_ops='', custom_leaf_ops=''):
    """Parse custom operator specifications into nd2py operator classes"""
    
    # Available operator mappings
    BINARY_OPS = {
        'Add': None, 'Sub': None, 'Mul': None, 'Div': None, 'Pow': None
    }
    
    UNARY_OPS = {
        'Sin': None, 'Cos': None, 'Tan': None,
        'Log': None, 'Exp': None, 'Sqrt': None,
        'Inv': None, 'Neg': None, 'Pow2': None, 'Pow3': None,
        'Arcsin': None, 'Arccos': None, 'Cot': None, 'Tanh': None
    }
    
    # Parse binary operators
    binary_ops = []
    if custom_binary_ops.strip():
        for op_name in custom_binary_ops.split(','):
            op_name = op_name.strip()
            if op_name in BINARY_OPS:
                # Will be resolved later when nd2 is imported
                binary_ops.append(op_name)
            else:
                print(f"Unknown binary operator: {op_name}. Available: {list(BINARY_OPS.keys())}")
    
    # Parse unary operators
    unary_ops = []
    if custom_unary_ops.strip():
        for op_name in custom_unary_ops.split(','):
            op_name = op_name.strip()
            if op_name in UNARY_OPS:
                # Will be resolved later when nd2 is imported
                unary_ops.append(op_name)
            else:
                print(f"Unknown unary operator: {op_name}. Available: {list(UNARY_OPS.keys())}")
    
    # Parse leaf constants
    leaf_ops = []
    if custom_leaf_ops.strip():
        for leaf_val in custom_leaf_ops.split(','):
            leaf_val = leaf_val.strip()
            try:
                if leaf_val.lower() == 'pi':
                    leaf_ops.append(('pi', np.pi))
                elif leaf_val.lower() == 'e':
                    leaf_ops.append(('e', np.e))
                else:
                    val = float(eval(leaf_val, {"__builtins__": {}}, {"pi": np.pi, "e": np.e}))
                    leaf_ops.append(('custom', val))
            except Exception as e:
                print(f"Failed to parse leaf constant '{leaf_val}': {e}")
    
    return binary_ops, unary_ops, leaf_ops

def NN_port(y_data, x_data, preset=None, allowed_functions=None,
            search_max_iter=None,               # 最大迭代次数 (覆盖 n_iter)
            target_r2=0.999,                    # 提前停止的 R2 阈值
            time_limit=8.0,                     # 秒级时间上限 (soft limit)
            sample_num=10000,                   # 采样点上限（>则下采样），增加以保留更多数据
            child_num=50,                       # MCTS child_num (与 search.py 一致，默认 50)
            n_playout=100,                      # 每次模拟次数 (与 search.py 一致，默认 100)
            d_playout=10,                       # 模拟深度 (与 search.py 一致，默认 10)
            max_len=30,                         # 表达式最大长度
            hint_function = None,            # 可选：提供 "x1+x2*sin(x3)" 用于自动提取算子集合
            log: bool = True,                   # 是否初始化 logger 并输出与 search.py 接近的日志
            strict_search: bool = False,        # True 时不做 early_stop/time_limit 干预，尽量复刻 search.py 行为
            verbose=False,                      # 详细输出
            # 新增参数，与 search.py 一致
            c=1.41,                             # MCTS c parameter
            eta=0.999,                          # Complexity decay factor in reward
            extra_leaf='',                      # Extra numeric constants, comma separated
            disable_prod_model=False,           # Disable multiplicative model fitting
            max_power_abs=None,                 # Clip absolute value of exponents
            complexity_limit=None,              # Hard cap for expression complexity
            complexity_alpha=1.0,               # Scale complexity in reward
            custom_binary_ops='',               # Custom binary operators
            custom_unary_ops='',                # Custom unary operators  
            custom_leaf_ops='',                 # Custom leaf constants
            use_custom_ops_only=False,          # Use only custom operators
            config=None):                       # Config object to read binary/unary ops from
    """直接调用 SR4MDL (search.py 逻辑) 进行符号回归并返回 loss。

    Args:
        y_data: (N,) 数组
        x_data: (N, d) 数组
        allowed_functions: 限制 unary 函数集合 (例如 ['sin','cos','log'])
        search_max_iter: 限制 MCTS 搜索迭代数 (n_iter)

    Returns:
        loss = max(0, 1-R2)
    """
    t0 = time_module.time()
    x_data = np.array(x_data, dtype=np.float64)
    y_data = np.array(y_data, dtype=np.float64).reshape(-1)
    
    n, d = x_data.shape
    print(f"[SR4MDL] 输入数据统计 - data shape X={x_data.shape}, y={y_data.shape}")
    print(f"[SR4MDL] 输入 X 范围: [{x_data.min():.6f}, {x_data.max():.6f}], 均值: {x_data.mean():.6f}")
    print(f"[SR4MDL] 输入 y 范围: [{y_data.min():.6f}, {y_data.max():.6f}], 均值: {y_data.mean():.6f}, 标准差: {y_data.std():.6f}")
    
    print(f"[验证] 检验数据是否满足 y = 0.01 * (1 + log(x) + cos(x) / x)")
    if d == 1:  # 只对单变量数据进行验证
        x_vals = x_data[:, 0]  # 取第一列作为x
        # 计算理论y值
        y_theoretical = 0.01 * (x_vals + np.log(x_vals) + np.cos(x_vals) / x_vals)
        
        # 计算残差
        residuals = y_data - y_theoretical
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        max_abs_error = np.abs(residuals).max()
        
        print(f"[验证] 理论y值范围: [{y_theoretical.min():.6f}, {y_theoretical.max():.6f}]")
        print(f"[验证] 实际y值范围: [{y_data.min():.6f}, {y_data.max():.6f}]")
        print(f"[验证] 残差统计:")
        print(f"[验证]   均方误差(MSE): {mse:.8f}")
        print(f"[验证]   均方根误差(RMSE): {rmse:.8f}")
        print(f"[验证]   最大绝对误差: {max_abs_error:.8f}")
        print(f"[验证]   残差范围: [{residuals.min():.8f}, {residuals.max():.8f}]")
        print(f"[验证]   残差均值: {residuals.mean():.8f}")
        print(f"[验证]   残差标准差: {residuals.std():.8f}")
        
        # 计算相关系数
        correlation = np.corrcoef(y_data, y_theoretical)[0, 1]
        print(f"[验证]   相关系数: {correlation:.8f}")
        
        # 计算R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        print(f"[验证]   R²: {r2:.8f}")
        
        # 判断拟合质量
        if rmse < 1e-10:
            print(f"[验证] 结论: 数据完全符合公式 y = 0.01 * (1 + log(x) + cos(x))")
        elif rmse < 1e-6:
            print(f"[验证] 结论: 数据非常接近公式 y = 0.01 * (1 + log(x) + cos(x))")
        elif rmse < 1e-3:
            print(f"[验证] 结论: 数据较好地符合公式 y = 0.01 * (1 + log(x) + cos(x))")
        elif r2 > 0.99:
            print(f"[验证] 结论: 数据与公式高度相关，但可能有缩放差异")
        else:
            print(f"[验证] 结论: 数据不符合公式 y = 0.01 * (1 + log(x) + cos(x))")
            
        # 显示前10个数据点的详细对比
        print(f"[验证] 前10个数据点详细对比:")
        for i in range(min(10, len(x_vals))):
            print(f"[验证]   x[{i}]={x_vals[i]:.6f}, y_actual={y_data[i]:.8f}, y_theory={y_theoretical[i]:.8f}, diff={residuals[i]:.8f}")
    else:
        print(f"[验证] 跳过验证：数据有{d}个变量，不是单变量数据")
    
    if x_data.shape[0] > sample_num:
        print(f"[SR4MDL] 准备下采样: {x_data.shape[0]} -> {sample_num}")
        print(f"[SR4MDL] 下采样前详细统计:")
        print(f"[SR4MDL] X 完整统计: 最小值={x_data.min():.6f}, 最大值={x_data.max():.6f}, 均值={x_data.mean():.6f}, 标准差={x_data.std():.6f}")
        print(f"[SR4MDL] y 完整统计: 最小值={y_data.min():.6f}, 最大值={y_data.max():.6f}, 均值={y_data.mean():.6f}, 标准差={y_data.std():.6f}")
        print(f"[SR4MDL] y 百分位数: 1%={np.percentile(y_data, 1):.6f}, 5%={np.percentile(y_data, 5):.6f}, 95%={np.percentile(y_data, 95):.6f}, 99%={np.percentile(y_data, 99):.6f}")
        
        # 使用分层采样而不是完全随机采样，以保持数据分布
        try:
            # 按y值排序，然后每隔一定间隔取样，这样可以保持分布特征
            sorted_idx = np.argsort(y_data)
            step = len(y_data) // sample_num
            if step > 1:
                # 分层采样：每隔step取一个点，加上一些随机性
                stratified_idx = sorted_idx[::step][:sample_num]
                # 如果还不够sample_num个点，随机补充
                if len(stratified_idx) < sample_num:
                    remaining_idx = np.setdiff1d(np.arange(len(y_data)), stratified_idx)
                    extra_needed = sample_num - len(stratified_idx)
                    extra_idx = np.random.choice(remaining_idx, extra_needed, replace=False)
                    idx = np.concatenate([stratified_idx, extra_idx])
                else:
                    idx = stratified_idx
                print(f"[SR4MDL] 使用分层采样策略")
            else:
                # 如果步长<=1，退回到随机采样
                idx = np.random.choice(x_data.shape[0], sample_num, replace=False)
                print(f"[SR4MDL] 使用随机采样策略")
        except Exception as e:
            print(f"[SR4MDL] 分层采样失败，退回到随机采样: {e}")
            idx = np.random.choice(x_data.shape[0], sample_num, replace=False)
        
        # 应用采样
        x_data_original = x_data.copy()
        y_data_original = y_data.copy()
        x_data = x_data[idx]
        y_data = y_data[idx]
        
        print(f"[SR4MDL] 下采样后数据统计:")
        print(f"[SR4MDL] X 范围变化: [{x_data_original.min():.6f}, {x_data_original.max():.6f}] -> [{x_data.min():.6f}, {x_data.max():.6f}]")
        print(f"[SR4MDL] y 范围变化: [{y_data_original.min():.6f}, {y_data_original.max():.6f}] -> [{y_data.min():.6f}, {y_data.max():.6f}]")
        print(f"[SR4MDL] y 均值变化: {y_data_original.mean():.6f} -> {y_data.mean():.6f}")
        print(f"[SR4MDL] y 标准差变化: {y_data_original.std():.6f} -> {y_data.std():.6f}")
        print(f"[SR4MDL] down-sampled to {x_data.shape[0]} points")

    sr4mdl_path = Path(__file__).parent / "SR4MDL"
    if str(sr4mdl_path) not in sys.path:
        sys.path.insert(0, str(sr4mdl_path))
    try:
        from sr4mdl.model import MDLformer
        from sr4mdl.search import MCTS4MDL
        from sr4mdl.env import Tokenizer, sympy2eqtree, str2sympy
        from nd2py.utils import AttrDict, init_logger, seed_all
        import nd2py as nd2
    except Exception as e:
        print(f"[SR4MDL] import failed: {e}")
        return 1.0

    if log:
        # 初始化 logger（避免重复初始化）
        if not logging.getLogger('sr4mdl.search').handlers:
            try:
                seed_all(0)
                init_logger('sr4mdl', 'inline', None)
            except Exception:
                pass
        logger = logging.getLogger('sr4mdl.search')
        logger.note = lambda msg: logger.log(getattr(logging, 'NOTE', logging.INFO), msg) if hasattr(logger, 'log') else logger.info(msg)
    else:
        logger = None

    # 变量字典 - 强制使用浮点数组，避免整数数组在负幂运算时报错
    var_names = [f"x{i+1}" for i in range(d)]
    X_dict = {var_names[i]: x_data[:, i].astype(np.float64) for i in range(d)}
    
    # 调试信息：打印数据类型确认
    print(f"[DEBUG] X_dict dtypes: {[(k, v.dtype, 'integer_check:', np.all(v == np.round(v))) for k, v in X_dict.items()]}")
    print(f"[DEBUG] y_data dtype: {y_data.dtype}, integer_check: {np.all(y_data == np.round(y_data))}")
    
    # 如果数据看起来像整数，添加小的噪声确保它们是"真正的"浮点数
    for var_name, var_data in X_dict.items():
        if np.all(var_data == np.round(var_data)):
            # 添加微小噪声避免严格整数值导致的幂运算问题
            X_dict[var_name] = var_data + np.random.normal(0, 1e-12, var_data.shape)
            print(f"[DEBUG] Added noise to {var_name} to avoid integer values")
    
    if np.all(y_data == np.round(y_data)):
        y_data = y_data + np.random.normal(0, 1e-12, y_data.shape)
        print(f"[DEBUG] Added noise to y_data to avoid integer values")

    # 从配置中读取操作符设置（优先级：config > hint_function > 默认值）
    config_binary_ops = None
    config_unary_ops = None
    config_leaf_ops = None
    
    # 从配置中读取 SR 参数（如果提供了 config）
    config_params = {}
    
    if config is not None:
        print(f"[DEBUG] config object type: {type(config)}")
        print(f"[DEBUG] config has 'problem' attr: {hasattr(config, 'problem')}")
        if hasattr(config, 'problem'):
            print(f"[DEBUG] config.problem type: {type(config.problem)}")
            print(f"[DEBUG] config.problem keys: {list(config.problem.keys()) if hasattr(config.problem, 'keys') else 'N/A'}")
            print(f"[DEBUG] 'symbolic_regression_config' in config.problem: {'symbolic_regression_config' in config.problem}")
        
        # 尝试从 config 中读取操作符设置
        try:
            if hasattr(config, 'problem') and 'symbolic_regression_config' in config.problem:
                sr_config = config.problem['symbolic_regression_config']
                print(f"[DEBUG] sr_config keys: {list(sr_config.keys()) if hasattr(sr_config, 'keys') else 'N/A'}")
                
                # 读取操作符配置
                if 'binary_ops' in sr_config:
                    config_binary_ops = sr_config['binary_ops']
                    print(f"[DEBUG] Found binary_ops in config: {config_binary_ops}")
                if 'unary_ops' in sr_config:
                    config_unary_ops = sr_config['unary_ops']
                    print(f"[DEBUG] Found unary_ops in config: {config_unary_ops}")
                if 'leaf_ops' in sr_config:
                    config_leaf_ops = sr_config['leaf_ops']
                    print(f"[DEBUG] Found leaf_ops in config: {config_leaf_ops}")
                
                # 读取搜索参数
                param_mapping = {
                    'sample_num': 'sample_num',
                    'n_iter': 'search_max_iter',
                    'search_max_iter': 'search_max_iter',
                    'time_limit': 'time_limit',
                    'target_r2': 'target_r2',
                    'child_num': 'child_num',
                    'n_playout': 'n_playout',
                    'd_playout': 'd_playout',
                    'c': 'c',
                    'eta': 'eta',
                    'complexity_alpha': 'complexity_alpha',
                    'complexity_limit': 'complexity_limit',
                    'disable_prod_model': 'disable_prod_model',
                    'max_power_abs': 'max_power_abs',
                    'max_len': 'max_len',
                    'extra_leaf': 'extra_leaf',
                    'strict_search': 'strict_search',
                    'verbose': 'verbose'
                }
                
                for config_key, param_key in param_mapping.items():
                    if config_key in sr_config:
                        config_params[param_key] = sr_config[config_key]
                
                if log and logger:
                    logger.info(f"[config] 从配置读取参数: {list(config_params.keys())}")
                    if config_binary_ops:
                        logger.info(f"[config] binary ops: {config_binary_ops}")
                    if config_unary_ops:
                        logger.info(f"[config] unary ops: {config_unary_ops}")
                    if config_leaf_ops:
                        logger.info(f"[config] leaf ops: {config_leaf_ops}")
                        
        except Exception as e:
            if log and logger:
                logger.warning(f"Failed to read operators from config: {e}")
    
    # 使用配置参数覆盖默认参数
    if 'sample_num' in config_params:
        sample_num = config_params['sample_num']
    if 'search_max_iter' in config_params:
        search_max_iter = config_params['search_max_iter']
    if 'time_limit' in config_params:
        time_limit = config_params['time_limit']
    if 'target_r2' in config_params:
        target_r2 = config_params['target_r2']
    if 'child_num' in config_params:
        child_num = config_params['child_num']
    if 'n_playout' in config_params:
        n_playout = config_params['n_playout']
    if 'd_playout' in config_params:
        d_playout = config_params['d_playout']
    if 'c' in config_params:
        c = config_params['c']
    if 'eta' in config_params:
        eta = config_params['eta']
    if 'complexity_alpha' in config_params:
        complexity_alpha = config_params['complexity_alpha']
    if 'complexity_limit' in config_params:
        complexity_limit = config_params['complexity_limit']
    if 'disable_prod_model' in config_params:
        disable_prod_model = config_params['disable_prod_model']
    if 'max_power_abs' in config_params:
        max_power_abs = config_params['max_power_abs']
    if 'max_len' in config_params:
        max_len = config_params['max_len']
    if 'extra_leaf' in config_params:
        extra_leaf = config_params['extra_leaf']
    if 'strict_search' in config_params:
        strict_search = config_params['strict_search']
    if 'verbose' in config_params:
        verbose = config_params['verbose']

    # 运算符集合获取：优先级 config > hint_function > 默认值
    print(f"[DEBUG] 决定操作符: config_binary_ops={config_binary_ops}, config_unary_ops={config_unary_ops}")
    print(f"[DEBUG] hint_function={hint_function}")
    
    if config_binary_ops is not None or config_unary_ops is not None:
        print("[DEBUG] 使用配置中的操作符")
        # 使用配置中的操作符
        if config_binary_ops is not None:
            # 将配置中的字符串映射为 nd2 类
            binary_name_map = {'Add': nd2.Add, 'Sub': nd2.Sub, 'Mul': nd2.Mul, 'Div': nd2.Div, 'Pow': nd2.Pow}
            binary = [binary_name_map[name] for name in config_binary_ops if name in binary_name_map]
            print(f"[DEBUG] 配置的 binary ops: {config_binary_ops} -> {[op.__name__ for op in binary]}")
        else:
            binary = [nd2.Mul, nd2.Div, nd2.Add, nd2.Sub]  # 默认值
            print(f"[DEBUG] 使用默认 binary ops: {[op.__name__ for op in binary]}")
            
        if config_unary_ops is not None:
            # 将配置中的字符串映射为 nd2 类
            unary_name_map = {
                'Sin': nd2.Sin, 'Cos': nd2.Cos, 'Tan': nd2.Tan,
                'Log': nd2.Log, 'Exp': nd2.Exp, 'Sqrt': nd2.Sqrt,
                'Inv': nd2.Inv, 'Neg': nd2.Neg, 'Pow2': nd2.Pow2, 'Pow3': nd2.Pow3,
                'Arcsin': nd2.Arcsin, 'Arccos': nd2.Arccos, 'Cot': nd2.Cot, 'Tanh': nd2.Tanh
            }
            unary = [unary_name_map[name] for name in config_unary_ops if name in unary_name_map]
            print(f"[DEBUG] 配置的 unary ops: {config_unary_ops} -> {[op.__name__ for op in unary]}")
        else:
            unary = [nd2.Sqrt, nd2.Cos, nd2.Sin, nd2.Pow2, nd2.Pow3, nd2.Exp, nd2.Inv, nd2.Neg, nd2.Arcsin, nd2.Arccos, nd2.Cot, nd2.Log, nd2.Tanh]
            print(f"[DEBUG] 使用默认完整 unary ops: {[op.__name__ for op in unary]}")
            
        if config_leaf_ops is not None:
            # 将配置中的数值转换为 nd2.Number（确保是浮点数）
            leaf = []
            for val in config_leaf_ops:
                if isinstance(val, (int, float)):
                    leaf.append(nd2.Number(float(val)))  # 强制转换为浮点数
                elif isinstance(val, str):
                    if val.lower() == 'pi':
                        leaf.append(nd2.Number(float(np.pi)))
                    elif val.lower() == 'e':
                        leaf.append(nd2.Number(float(np.e)))
                    else:
                        try:
                            leaf.append(nd2.Number(float(val)))
                        except ValueError:
                            if log and logger:
                                logger.warning(f"Failed to parse leaf constant: {val}")
            print(f"[DEBUG] 配置的 leaf ops: {config_leaf_ops} -> {[l.to_str(number_format='.2f') for l in leaf]}")
        else:
            leaf = [nd2.Number(1.0), nd2.Number(2.0), nd2.Number(float(np.pi))]  # 确保默认值也是浮点数
            print(f"[DEBUG] 使用默认 leaf ops: {[l.to_str(number_format='.2f') for l in leaf]}")
            
        if log and logger:
            logger.info(f"[config] using operators from config - binary: {[op.__name__ for op in binary]}, unary: {[op.__name__ for op in unary]}")
            
    elif hint_function:
        print("[DEBUG] 使用 hint_function 提取的操作符")
        try:
            f_tree = sympy2eqtree(str2sympy(hint_function))
            seen_bin=set(); seen_un=set(); seen_leaf=set(); seen_vars=set()
            extracted_bin=[]; extracted_un=[]; leaf_nodes=[]
            for op in f_tree.iter_preorder():
                if op.n_operands==2 and op.__class__ not in seen_bin:
                    seen_bin.add(op.__class__); extracted_bin.append(op.__class__)
                elif op.n_operands==1 and op.__class__ not in seen_un:
                    seen_un.add(op.__class__); extracted_un.append(op.__class__)
                if isinstance(op, nd2.Number):
                    k=op.to_str(number_format=".8f")
                    if k not in seen_leaf: seen_leaf.add(k); leaf_nodes.append(op)
            if not leaf_nodes: leaf_nodes.append(nd2.Number(1))
            binary = extracted_bin
            unary = extracted_un
            leaf = leaf_nodes
            if log and logger:
                logger.info(f"[hint] operators from hint_function={hint_function} -> bin={[op.__name__ for op in binary]}, unary={[op.__name__ for op in unary]}")
        except Exception as e:
            print(f"[SR4MDL] hint_function parse failed: {e}; fallback to default operator set")
            hint_function = None
    
    if not hint_function and (config_binary_ops is None and config_unary_ops is None):
        print("[DEBUG] 使用默认的完整操作符集合")
        # 使用与 search.py 完全一致的默认运算符集合
        binary = [nd2.Mul, nd2.Div, nd2.Add, nd2.Sub]
        unary = [nd2.Sqrt, nd2.Cos, nd2.Sin, nd2.Pow2, nd2.Pow3, nd2.Exp, nd2.Inv, nd2.Neg, nd2.Arcsin, nd2.Arccos, nd2.Cot, nd2.Log, nd2.Tanh]
        leaf = [nd2.Number(1.0), nd2.Number(2.0), nd2.Number(float(np.pi))]  # 确保是浮点数
        print(f"[DEBUG] 默认 binary: {[op.__name__ for op in binary]}")
        print(f"[DEBUG] 默认 unary: {[op.__name__ for op in unary]}")
        print(f"[DEBUG] 默认 leaf: {[l.to_str(number_format='.2f') for l in leaf]}")

    # 处理自定义操作符
    custom_binary, custom_unary, custom_leaf = parse_custom_operators(
        custom_binary_ops, custom_unary_ops, custom_leaf_ops
    )
    
    # 解析自定义操作符字符串为nd2类
    if custom_binary:
        name2cls = {'Add': nd2.Add, 'Sub': nd2.Sub, 'Mul': nd2.Mul, 'Div': nd2.Div, 'Pow': nd2.Pow}
        custom_binary = [name2cls[name] for name in custom_binary if name in name2cls]
    if custom_unary:
        name2cls_unary = {
            'Sin': nd2.Sin, 'Cos': nd2.Cos, 'Tan': nd2.Tan,
            'Log': nd2.Log, 'Exp': nd2.Exp, 'Sqrt': nd2.Sqrt,
            'Inv': nd2.Inv, 'Neg': nd2.Neg, 'Pow2': nd2.Pow2, 'Pow3': nd2.Pow3,
            'Arcsin': nd2.Arcsin, 'Arccos': nd2.Arccos, 'Cot': nd2.Cot, 'Tanh': nd2.Tanh
        }
        custom_unary = [name2cls_unary[name] for name in custom_unary if name in name2cls_unary]
    if custom_leaf:
        # 确保叶子常数为浮点数（-1 -> -1.0），避免整数负幂错误
        custom_leaf = [nd2.Number(float(val)) for name, val in custom_leaf]
    
    if use_custom_ops_only:
        # 使用仅自定义操作符
        if custom_binary:
            binary = custom_binary
            if log and logger:
                logger.note(f"Using custom binary operators only: {[op.__name__ for op in binary]}")
        if custom_unary:
            unary = custom_unary
            if log and logger:
                logger.note(f"Using custom unary operators only: {[op.__name__ for op in unary]}")
        if custom_leaf:
            leaf = custom_leaf
            if log and logger:
                logger.note(f"Using custom leaf constants only: {[op.to_str(number_format='.2f') for op in leaf]}")
    else:
        # 合并自定义操作符与现有操作符
        if custom_binary:
            for op in custom_binary:
                if op not in binary:
                    binary.append(op)
            if log and logger:
                logger.note(f"Added custom binary operators: {[op.__name__ for op in custom_binary]}")
        
        if custom_unary:
            for op in custom_unary:
                if op not in unary:
                    unary.append(op)
            if log and logger:
                logger.note(f"Added custom unary operators: {[op.__name__ for op in custom_unary]}")
        
        if custom_leaf:
            for new_leaf in custom_leaf:
                is_duplicate = False
                for existing_leaf in leaf:
                    if isinstance(existing_leaf, nd2.Number) and isinstance(new_leaf, nd2.Number):
                        if abs(existing_leaf.value - new_leaf.value) < 1e-12:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    leaf.append(new_leaf)
            if log and logger:
                logger.note(f"Added custom leaf constants: {[op.to_str(number_format='.2f') for op in custom_leaf]}")

    # 可选限制 unary 函数
    if allowed_functions:
        allow_set = set(a.lower() for a in allowed_functions)
        new_unary = [op for op in unary if op.__name__.lower() in allow_set]
        if new_unary:
            unary = new_unary
            if log and logger:
                logger.info(f"[SR4MDL] constrained unary ops -> {[op.__name__ for op in unary]}")

    if log and logger:
        logger.note('\n'.join([
            f'target: data_mode',
            f'binary operators: {[op.__name__ for op in binary]}',
            f'unary operators: {[op.__name__ for op in unary]}',
            'leaf: [' + ', '.join([l.to_str(number_format=".2f") for l in leaf]) + ']',
            f'variables: {list(X_dict.keys())}'
        ]))

    # 设备 & 权重缓存
    global _SR4MDL_WEIGHT_STATE, _SR4MDL_WEIGHT_META, _SR4MDL_DEVICE
    if _SR4MDL_DEVICE is None:
        _SR4MDL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = _SR4MDL_DEVICE
    weights_path = sr4mdl_path / 'weights' / 'checkpoint.pth'
    if not weights_path.exists():
        print(f"[SR4MDL] weights missing: {weights_path}")
        return 1.0

    # 构建 / 复用模型 (参数与 search.py 完全一致)
    model_keys_hash = ('mdlformer', n)
    need_reload = True
    if _SR4MDL_WEIGHT_META and _SR4MDL_WEIGHT_META == model_keys_hash and _SR4MDL_WEIGHT_STATE is not None:
        need_reload = False
    if need_reload:
        state_dict = torch.load(str(weights_path), map_location=device)
        # 创建模型结构 (参数完全参照 search.py)
        model_args = AttrDict(
            dropout=0.1, 
            d_model=512, 
            d_input=64, 
            d_output=512, 
            n_TE_layers=8, 
            max_len=50, 
            max_param=5, 
            max_var=max(10, d), 
            uniform_sample_number=n, 
            device=device, 
            use_SENet=True, 
            use_old_model=False
        )
        tokenizer = Tokenizer(-100, 100, 4, model_args.max_var)
        model = MDLformer(model_args, state_dict['xy_token_list'])
        model.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=True)
        model.eval()
        model.to(device)
        _SR4MDL_WEIGHT_STATE = {'model': model, 'tokenizer': tokenizer, 'args': model_args}
        _SR4MDL_WEIGHT_META = model_keys_hash
    else:
        model = _SR4MDL_WEIGHT_STATE['model']
        tokenizer = _SR4MDL_WEIGHT_STATE['tokenizer']
        model_args = _SR4MDL_WEIGHT_STATE['args']
        # 若样本数不同，需要更新 uniform_sample_number
        if getattr(model_args, 'uniform_sample_number', None) != n:
            model_args.uniform_sample_number = n

    n_iter = int(search_max_iter) if search_max_iter is not None else 10000
    
    # 从配置参数中获取数据处理参数
    normalize_y = config_params.get('normalize_y', False)
    normalize_all = config_params.get('normalize_all', False)
    remove_abnormal = config_params.get('remove_abnormal', True)
    keep_vars = config_params.get('keep_vars', True)
    log_per_sec = config_params.get('log_per_sec', 5)
    
    # 设置最大幂次限制，避免负幂或过大幂次导致的问题
    if max_power_abs is None:
        max_power_abs = 3  # 限制幂次在 [-3, 3] 范围内
    
    # 在传给MCTS4MDL前再次检查数据状态
    print(f"[SR4MDL] 传给MCTS4MDL前的数据统计:")
    print(f"[SR4MDL] X_dict范围: {[(k, v.min(), v.max(), v.mean(), v.std()) for k, v in X_dict.items()]}")
    print(f"[SR4MDL] y_data范围: [{y_data.min():.6f}, {y_data.max():.6f}], 均值: {y_data.mean():.6f}, 标准差: {y_data.std():.6f}")
    print(f"[SR4MDL] 归一化参数: normalize_y={normalize_y}, normalize_all={normalize_all}")
    print(f"[SR4MDL] sample_num参数: {sample_num} (可能导致MCTS4MDL内部再次采样)")
    
    # 设置一个很大的sample_num，防止MCTS4MDL内部再次采样
    internal_sample_num = max(sample_num, len(y_data) * 2)  # 确保不会被内部采样
    print(f"[SR4MDL] 设置internal_sample_num={internal_sample_num}以防止内部采样")
    
    est = MCTS4MDL(
        tokenizer=tokenizer,
        model=model,
        n_iter=n_iter,
        c=c,
        sample_num=internal_sample_num,  # 使用更大的值
        keep_vars=keep_vars,
        normalize_y=normalize_y,
        normalize_all=normalize_all,
        remove_abnormal=remove_abnormal,
        binary=binary,
        unary=unary,
        leaf=leaf,
        child_num=child_num,
        n_playout=n_playout,
        d_playout=d_playout,
        log_per_sec=log_per_sec,
        eta=eta,
        disable_prod_model=disable_prod_model,
        max_power_abs=max_power_abs,
        complexity_limit=complexity_limit,
        complexity_alpha=complexity_alpha,
    )

    # 处理额外的叶子常量 (与 search.py 一致)
    if extra_leaf:
        try:
            extras = []
            for tok in extra_leaf.split(','):
                tok = tok.strip()
                if not tok:
                    continue
                val = float(eval(tok, {}, {}))  # allow simple expressions like 1/100
                if all(abs(val - l.value) > 1e-12 if isinstance(l, nd2.Number) else True for l in est.leaf):
                    extras.append(nd2.Number(val))
            if extras:
                est.leaf.extend(extras)
                if log and logger:
                    logger.note(f"Added extra leaf constants: {[x.to_str() for x in extras]}")
        except Exception as e:
            if log and logger:
                logger.warning(f"Failed parsing extra_leaf '{extra_leaf}': {e}")

    start_time = time_module.time()
    def early_stop(r2, complexity, eq):
        if strict_search:
            return False
        if r2 >= target_r2:
            return True
        if time_module.time() - start_time > time_limit:
            if verbose:
                print(f"[SR4MDL] early stop by time>{time_limit}s, r2={r2:.5f}")
            return True
        return False

    try:
        # 模仿 search.py 的数据预处理方式
        # 1. 先检查有限性
        finite_mask = np.isfinite(y_data)
        for var_name, var_data in X_dict.items():
            finite_mask &= np.isfinite(var_data)
        
        if not finite_mask.all():
            # 过滤非有限样本，就像 search.py 那样
            old_n = len(y_data)
            for var_name in X_dict:
                X_dict[var_name] = X_dict[var_name][finite_mask]
            y_data = y_data[finite_mask]
            new_n = len(y_data)
            if log and logger:
                logger.warning(f"Filtered non-finite samples: {old_n} -> {new_n}")
        
        # 2. 确保没有严格的零值（避免 log(0) 等问题）
        avoid_zero_eps = 1e-8  # 与 search.py 的 avoid_zero_eps 参数一致
        for var_name, var_data in X_dict.items():
            mask = np.abs(var_data) < avoid_zero_eps
            if mask.any():
                # 保持符号，但避免零值
                X_dict[var_name] = np.where(mask, 
                                          np.where(var_data >= 0, avoid_zero_eps, -avoid_zero_eps),
                                          var_data)
        
        print(f"[SR4MDL] 调用est.fit前的最终数据统计:")
        print(f"[SR4MDL] X_dict keys: {list(X_dict.keys())}")
        for k, v in X_dict.items():
            print(f"[SR4MDL] {k}: shape={v.shape}, range=[{v.min():.6f}, {v.max():.6f}], mean={v.mean():.6f}, std={v.std():.6f}")
        print(f"[SR4MDL] y_data: shape={y_data.shape}, range=[{y_data.min():.6f}, {y_data.max():.6f}], mean={y_data.mean():.6f}, std={y_data.std():.6f}")
        
        print(f"[SR4MDL] Final data validation - X range: [{min(np.min(v) for v in X_dict.values()):.6f}, {max(np.max(v) for v in X_dict.values()):.6f}], y range: [{y_data.min():.6f}, {y_data.max():.6f}]")
        
        # 3. 使用与 search.py 相同的错误处理策略 - 捕获但不终止
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略运行时警告，就像 search.py 那样
            est.fit(X_dict, y_data, use_tqdm=False, early_stop=early_stop)
        
        # 检查est.fit()后数据是否被MCTS4MDL修改
        print(f"[SR4MDL] est.fit()执行后数据检查:")
        print(f"[SR4MDL] 原始 X_dict 是否被修改:")
        for k, v in X_dict.items():
            print(f"[SR4MDL] {k}: shape={v.shape}, range=[{v.min():.6f}, {v.max():.6f}], mean={v.mean():.6f}, std={v.std():.6f}")
        print(f"[SR4MDL] 原始 y_data 是否被修改: shape={y_data.shape}, range=[{y_data.min():.6f}, {y_data.max():.6f}], mean={y_data.mean():.6f}, std={y_data.std():.6f}")
        
        # 检查MCTS4MDL内部是否有处理后的数据
        if hasattr(est, 'X_dict_processed'):
            print(f"[SR4MDL] MCTS4MDL内部处理后的X_dict_processed:")
            for k, v in est.X_dict_processed.items():
                print(f"[SR4MDL] {k}: range=[{v.min():.6f}, {v.max():.6f}], mean={v.mean():.6f}, std={v.std():.6f}")
        if hasattr(est, 'y_data_processed'):
            print(f"[SR4MDL] MCTS4MDL内部处理后的y_data_processed: range=[{est.y_data_processed.min():.6f}, {est.y_data_processed.max():.6f}], mean={est.y_data_processed.mean():.6f}, std={est.y_data_processed.std():.6f}")
        
        # 检查常见的归一化属性
        if hasattr(est, 'X_mean') or hasattr(est, 'X_std') or hasattr(est, 'y_mean') or hasattr(est, 'y_std'):
            print(f"[SR4MDL] 发现归一化参数:")
            if hasattr(est, 'X_mean'):
                print(f"[SR4MDL] X_mean: {est.X_mean}")
            if hasattr(est, 'X_std'):
                print(f"[SR4MDL] X_std: {est.X_std}")
            if hasattr(est, 'y_mean'):
                print(f"[SR4MDL] y_mean: {est.y_mean}")
            if hasattr(est, 'y_std'):
                print(f"[SR4MDL] y_std: {est.y_std}")
            
        if log and logger and est.records:
            last = est.records[-1]
            if 'r2' in last:
                logger.info(f"[SR4MDL] last_iter={len(est.records)} r2={last['r2']:.6f} mdl={last.get('mdl','-')}")
    except KeyboardInterrupt:
        print('[SR4MDL] interrupted')
    except Exception as e:
        print(f"[SR4MDL] search error: {e}")
        # 不要立即返回错误，尝试继续处理
        import traceback
        print(f"[SR4MDL] Full traceback: {traceback.format_exc()}")
        # 检查是否有部分结果可以使用
        if not hasattr(est, 'eqtree') or est.eqtree is None:
            return 1.0

    # 预测 / 评估 (与 search.py 一致)
    try:
        print(f"[SR4MDL] 准备调用est.predict():")
        print(f"[SR4MDL] 传给predict的X_dict:")
        for k, v in X_dict.items():
            print(f"[SR4MDL] {k}: shape={v.shape}, range=[{v.min():.6f}, {v.max():.6f}], mean={v.mean():.6f}, std={v.std():.6f}")
            
        y_pred = est.predict(X_dict)
        
        print(f"[SR4MDL] predict()返回结果:")
        print(f"[SR4MDL] y_pred: shape={y_pred.shape}, range=[{y_pred.min():.6f}, {y_pred.max():.6f}], mean={y_pred.mean():.6f}, std={y_pred.std():.6f}")
        print(f"[SR4MDL] 用于评估的y_data: shape={y_data.shape}, range=[{y_data.min():.6f}, {y_data.max():.6f}], mean={y_data.mean():.6f}, std={y_data.std():.6f}")
    except Exception as e:
        print(f"[SR4MDL] predict error: {e}")
        return 1.0
    
    # 计算评分 (完全与 search.py 一致)
    try:
        from sr4mdl.utils import RMSE_score, R2_score
        rmse = RMSE_score(y_data, y_pred) 
        r2 = R2_score(y_data, y_pred)
    except ImportError:
        # 备用计算方法
        rmse = float(np.sqrt(np.mean((y_pred - y_data)**2)))
        if np.std(y_data) == 0:
            r2 = 1.0 if rmse < 1e-6 else 0.0
        else:
            denom = np.sum((y_data - np.mean(y_data))**2)
            r2 = 1 - float(np.sum((y_pred - y_data)**2) / (denom + 1e-12))
    
    expr = est.eqtree
    t1 = time_module.time()
    
    # 输出详细信息 (与 search.py 风格一致)
    internal_r2 = getattr(est.best, 'r2', None)
    internal_complexity = getattr(est.best, 'complexity', None)
    internal_reward = getattr(est.best, 'reward', None)
    
    if internal_r2 is not None and log and logger:
        try:
            recomputed_reward = (eta ** (complexity_alpha * internal_complexity)) / (2 - internal_r2)
        except Exception:
            recomputed_reward = None
        logger.note(
            f'Result = {est.eqtree}, RMSE = {rmse:.4f}, R2 = {r2:.4f} | Internal(best) R2={internal_r2:.5f}, complexity={internal_complexity}, '
            f'internal_reward={internal_reward:.5f} (recomputed={recomputed_reward})'
        )
    else:
        print(f"[SR4MDL] expr: {expr}, R2={r2:.6f}, RMSE={rmse:.4f}, iters={len(est.records)}, time={t1-t0:.3f}s")
    
    loss = max(0, 1 - r2)
    return loss

def optimize_with_timeout(mse_func, init_params, constr_dict_list, prev_sol_best=None, verbose=False):    
    global iteration_count, params, start_time
    iteration_count = 0
    params = init_params
    start_time = 0

    # ---- hyper params ----
    time_limit = 1500
    bound_f_coef = 500
    check_time_nit = 9

    def callback(xk,):
        global iteration_count, params, start_time
        iteration_count += 1
        params = xk
        time_elapsed = time_module.time() - start_time  
        
        cur_f, cur_nit =  mse_func(xk), iteration_count
        if time_elapsed > time_limit: 
            raise EarlyStopException
        if prev_sol_best and cur_nit >= check_time_nit:
            prev_f, prev_nit = prev_sol_best["fun"], prev_sol_best["nit"]
            bound_f = bound_f_coef * (prev_nit/cur_nit)**2 * prev_f
            if cur_f > bound_f:
                raise EarlyStopException
    try:
        start_time = time_module.time()  
        if len(init_params) > 0:
            sol = scipy.optimize.minimize(mse_func, init_params, constraints=constr_dict_list, method="SLSQP",options={"disp":verbose}, callback=callback)
        else:
            sol = {"fun":mse_func(init_params), "x":init_params, "nit":1}
        sol["status"] = "Success"
    except EarlyStopException:
        sol = {"fun":mse_func(params), "x":params, "nit":iteration_count, "status":"EarlyStop"}
    sol['time'] = time_module.time() - start_time  
    return sol

def single_test(cfg, root_dir, deci_list, deleted_coef=[], init_params=None, verbose=True, preset=None, allowed_functions=None):
    tree = HypothesesTree(cfg, root_dir)
    SR_list = []
    for i in deci_list:
        related_vars = tree.activate_node(i, verbose=True)
        if len(related_vars) > 0:
            for rel in related_vars:
                SR_list.append(rel)
    # --- prepare for SR ---
    # construct library.
    tree.calculator.upd_local_dict()
    dummy_x = sp.symbols('x')
    tmp_local_dict = tree.calculator.local_dict
    tmp_local_dict['x'] = dummy_x
    lib = cfg.problem['symbolic_regression_config'].copy()
    lib = [sp.sympify(f, locals=tmp_local_dict) for f in lib['functions']]
        
    if len(SR_list) > 1:
        raise NotImplementedError("Multiple SR not supported yet.")
        
    # for (y, x_vars) in SR_list:
    #     tree.calculator.update_unknown_var(y, '0')
    #     for x in x_vars:
    #         for f in lib:
    #             name = str(f)+"_STR_coef_"+str(x)+"_for_"+str(y)
    #             if name in deleted_coef:
    #                 continue
    #             coef = sp.symbols(name)
    #             tree.calculator.sp_unknown_quantities[name] = coef
    #             tree.calculator.sp_derived_quantities[y] += coef * f.subs(dummy_x, tree.calculator.local_dict[x])
    # tree.calculator.upd_local_dict()
    
    if verbose:
        print("deci_list", deci_list)
        print("unknown_quantities", tree.calculator.sp_unknown_quantities.keys())
        print("equation", tree.calculator.equation_buffer)
    
    # --- optimization---
    #initial guess
    if init_params is None:
        init_params = np.random.rand(len(tree.calculator.sp_unknown_quantities))
        params_constr = tree.calculator.constraints
        for i, constr_list in enumerate(params_constr.values()):
            for c in constr_list:
                if "init" in c:
                    init_params[i] += c["init"]

    train_loss_func, mse_list_train = tree.calculator.get_loss_func(deci_list_len=len(deci_list))
    valid_loss_func, mse_list_valid = tree.calculator.get_loss_func(mode="valid", deci_list_len=len(deci_list))
    
    def SR_loss_func(params):
        params = params.copy()
        param_names = list(tree.calculator.sp_unknown_quantities.keys())
        
        print(f"SR_loss_func 调用，params: {dict(zip(param_names, params))}")
        
        #bound coef in [-0.05, +0.05]
        y, x_vars = SR_list[0]
        if y == "mu" and x_vars[0] == "gamma":
            # 从已知的动量方程中提取mu与gamma的关系
            # 动量方程: rho * conserve(V) - div(-p * I + tau) - F = 0
            # 其中 tau = 2*mu*S - (2/X_dim)*mu*div(V)*I
            # 通过设置mu=1和mu=0来分离出mu的系数项
            
            print("=== 从已知动量方程提取mu-gamma关系 ===")
            
            # 获取 gamma 的数据
            x_func = tree.calculator.gen_np_func([tree.calculator.sp_derived_quantities[x_vars[0]]])[0]
            x_data_raw = x_func(tree.calculator.train_args, params)
            print(f"gamma数据范围: [{x_data_raw.min():.6f}, {x_data_raw.max():.6f}]")
            
            # 使用当前的params来计算mu
            if 'mu' in param_names:
                mu_idx = param_names.index('mu')
                
                # 设置 mu=1 来计算其系数
                params_mu1 = params.copy()
                params_mu1[mu_idx] = 1.0
                
                # 设置 mu=0 来计算非粘性部分
                params_mu0 = params.copy()
                params_mu0[mu_idx] = 0.0

                # 获取动量方程 (向量形式)
                momentum_eq = tree.calculator.sp_equation[1]
                print(f"动量方程类型: {type(momentum_eq)}")
                print(f"动量方程形状: {momentum_eq.shape if hasattr(momentum_eq, 'shape') else 'No shape'}")
                
                # 提取动量方程的x和y分量
                try:
                    if hasattr(momentum_eq, '__getitem__') and hasattr(momentum_eq, 'shape') and len(momentum_eq.shape) > 0:
                        momentum_x_eq = momentum_eq[0]  # x分量
                        momentum_y_eq = momentum_eq[1]  # y分量
                        print("成功提取动量方程的x和y分量")
                    else:
                        raise ValueError("动量方程不是向量形式")
                        
                except Exception as e:
                    print(f"提取动量方程分量失败: {e}")
                    return 1e10
                
                # 分别计算x和y分量的函数
                momentum_x_func = tree.calculator.gen_np_func([momentum_x_eq])[0]
                momentum_y_func = tree.calculator.gen_np_func([momentum_y_eq])[0]
                
                print("正在计算动量方程残差...")
                # 计算 mu=1 时的残差
                residual_x_mu1 = momentum_x_func(tree.calculator.train_args, params_mu1)
                residual_y_mu1 = momentum_y_func(tree.calculator.train_args, params_mu1)
                
                # 计算 mu=0 时的残差
                residual_x_mu0 = momentum_x_func(tree.calculator.train_args, params_mu0)
                residual_y_mu0 = momentum_y_func(tree.calculator.train_args, params_mu0)
                
                print(f"x分量残差 mu=1: [{residual_x_mu1.min():.6f}, {residual_x_mu1.max():.6f}]")
                print(f"y分量残差 mu=1: [{residual_y_mu1.min():.6f}, {residual_y_mu1.max():.6f}]")
                print(f"x分量残差 mu=0: [{residual_x_mu0.min():.6f}, {residual_x_mu0.max():.6f}]")
                print(f"y分量残差 mu=0: [{residual_y_mu0.min():.6f}, {residual_y_mu0.max():.6f}]")
                
                # 计算mu的系数项 (拉普拉斯算子项)
                # 从调试结果知道：mu系数 = -∇²u (负拉普拉斯算子)
                mu_coefficient_x = residual_x_mu1 - residual_x_mu0  # x分量的mu系数
                mu_coefficient_y = residual_y_mu1 - residual_y_mu0  # y分量的mu系数
                
                print(f"mu系数x分量范围: [{mu_coefficient_x.min():.6f}, {mu_coefficient_x.max():.6f}]")
                print(f"mu系数y分量范围: [{mu_coefficient_y.min():.6f}, {mu_coefficient_y.max():.6f}]")

                # 根据动量方程平衡条件计算mu值
                # 动量方程 = 0，即：residual_mu0 + mu * mu_coefficient = 0
                # 因此：mu = -residual_mu0 / mu_coefficient
                
                print("=== 从动量方程反推mu的物理过程 ===")
                print("1. 动量方程: rho * conserve(V) - div(-p*I + tau) - F = 0")
                print("2. 其中 tau = 2*mu*S，包含粘性项 mu*∇²u")  
                print("3. 设置μ=1和μ=0计算残差差异，得到μ的系数项(拉普拉斯算子)")
                print("4. 通过平衡条件 μ = -残差_无粘 / 拉普拉斯系数 反推μ值")
                
                # 根据动量方程平衡条件计算mu值
                # 动量方程 = 0，即：residual_mu0 + mu * mu_coefficient = 0
                # 因此：mu = -residual_mu0 / mu_coefficient
                
                print("=== 从动量方程反推mu的物理过程 ===")
                print("1. 动量方程: rho * conserve(V) - div(-p*I + tau) - F = 0")
                print("2. 简化后mu项: -mu*∇²u (负拉普拉斯算子)")  
                print("3. 理论上：x和y分量应该给出相同的mu值（因为mu是标量）")
                print("4. 通过平衡条件 μ = -残差_无粘 / 拉普拉斯系数 反推μ值")
                
                # 重新设计mu值计算策略
                # 关键思想：mu是标量，x和y分量应该给出相同的值
                
                print("\n=== mu系数分析 ===")
                print(f"x分量系数统计: 均值={mu_coefficient_x.mean():.6f}, 标准差={mu_coefficient_x.std():.6f}")
                print(f"y分量系数统计: 均值={mu_coefficient_y.mean():.6f}, 标准差={mu_coefficient_y.std():.6f}")
                
                # 使用更合理的阈值：基于系数的标准差
                std_x = np.std(mu_coefficient_x)
                std_y = np.std(mu_coefficient_y)
                
                # 阈值设为1个标准差，这样可以保留大部分数据
                threshold_x = std_x * 1.0  # 保留约68%的数据
                threshold_y = std_y * 1.0
                
                print(f"新的阈值策略 - x分量: {threshold_x:.6f}, y分量: {threshold_y:.6f}")
                
                # 简化的掩码：只排除系数过小的点
                mask_x = np.abs(mu_coefficient_x) > threshold_x
                mask_y = np.abs(mu_coefficient_y) > threshold_y
                
                print(f"有效数据点 - x分量: {mask_x.sum()}/{mask_x.size} ({mask_x.mean()*100:.1f}%)")
                print(f"有效数据点 - y分量: {mask_y.sum()}/{mask_y.size} ({mask_y.mean()*100:.1f}%)")
                
                # 计算mu值
                mu_values_x = np.where(mask_x, -residual_x_mu0 / mu_coefficient_x, np.nan)
                mu_values_y = np.where(mask_y, -residual_y_mu0 / mu_coefficient_y, np.nan)
                
                # 统计有限值
                valid_x = np.isfinite(mu_values_x) & mask_x
                valid_y = np.isfinite(mu_values_y) & mask_y
                
                print(f"有限mu值计算点 - x分量: {valid_x.sum()}, y分量: {valid_y.sum()}")
                
                if valid_x.sum() == 0 or valid_y.sum() == 0:
                    print("错误: 没有有效的mu值计算点")
                    return 1e10
                
                # 提取有效值
                mu_x_valid = mu_values_x[valid_x]
                mu_y_valid = mu_values_y[valid_y]
                
                print(f"\n=== 原始mu值统计对比 ===")
                print(f"x分量原始mu值:")
                print(f"  数量: {len(mu_x_valid)}")
                print(f"  范围: [{mu_x_valid.min():.6f}, {mu_x_valid.max():.6f}]")
                print(f"  均值: {mu_x_valid.mean():.6f}")
                print(f"  中位数: {np.median(mu_x_valid):.6f}")
                print(f"  标准差: {mu_x_valid.std():.6f}")
                
                print(f"y分量原始mu值:")
                print(f"  数量: {len(mu_y_valid)}")
                print(f"  范围: [{mu_y_valid.min():.6f}, {mu_y_valid.max():.6f}]")
                print(f"  均值: {mu_y_valid.mean():.6f}")
                print(f"  中位数: {np.median(mu_y_valid):.6f}")
                print(f"  标准差: {mu_y_valid.std():.6f}")
                
                # 一致性检查：比较x和y分量的mu值
                if len(mu_x_valid) > 100 and len(mu_y_valid) > 100:
                    # 使用中位数进行比较（更稳健）
                    median_x = np.median(mu_x_valid)
                    median_y = np.median(mu_y_valid)
                    relative_diff = abs(median_x - median_y) / ((median_x + median_y) / 2) * 100
                    
                    print(f"\n=== mu值一致性检查 ===")
                    print(f"x分量中位数: {median_x:.6f}")
                    print(f"y分量中位数: {median_y:.6f}")
                    print(f"相对差异: {relative_diff:.2f}%")
                    
                    if relative_diff < 20:
                        print("✓ x和y分量的mu值基本一致")
                    else:
                        print("✗ x和y分量的mu值差异较大，可能存在数值问题")
                
                # 合并策略：由于mu应该是标量，我们合并两个分量的数据
                print(f"\n=== 数据合并策略 ===")
                print("由于mu是材料的标量属性，合并x和y分量的所有有效数据")
                
                # 简单的异常值过滤
                def simple_outlier_filter(values, factor=2.0):
                    """使用更宽松的异常值过滤"""
                    median_val = np.median(values)
                    mad = np.median(np.abs(values - median_val))  # 中位数绝对偏差
                    threshold_upper = median_val + factor * mad * 1.4826  # 1.4826是MAD到标准差的转换系数
                    threshold_lower = median_val - factor * mad * 1.4826
                    # 确保mu为正值
                    threshold_lower = max(threshold_lower, 1e-8)
                    mask = (values >= threshold_lower) & (values <= threshold_upper)
                    return values[mask], mask
                
                mu_x_filtered, _ = simple_outlier_filter(mu_x_valid)
                mu_y_filtered, _ = simple_outlier_filter(mu_y_valid)
                
                print(f"异常值过滤后 - x分量: {len(mu_x_filtered)}, y分量: {len(mu_y_filtered)}")
                
                # 合并所有过滤后的mu值
                mu_values_combined = np.concatenate([mu_x_filtered, mu_y_filtered])
                
                print(f"合并后的mu值统计:")
                print(f"  总数量: {len(mu_values_combined)}")
                print(f"  范围: [{mu_values_combined.min():.6f}, {mu_values_combined.max():.6f}]")
                print(f"  均值: {mu_values_combined.mean():.6f}")
                print(f"  中位数: {np.median(mu_values_combined):.6f}")
                print(f"  标准差: {mu_values_combined.std():.6f}")
                
                if len(mu_values_combined) < 1000:
                    print("警告: 合并后数据点较少，但继续进行")
                
                # 生成对应的gamma数据
                # 需要为合并的mu数据生成对应的gamma值
                x_data_flat = x_data_raw.flatten()
                
                # 获取x分量和y分量的有效索引
                valid_indices_x = np.where(valid_x.flatten())[0]
                valid_indices_y = np.where(valid_y.flatten())[0]
                
                # 应用异常值过滤的索引
                _, filter_mask_x = simple_outlier_filter(mu_x_valid)
                _, filter_mask_y = simple_outlier_filter(mu_y_valid)
                
                final_indices_x = valid_indices_x[filter_mask_x]
                final_indices_y = valid_indices_y[filter_mask_y]
                
                # 合并索引并提取对应的gamma值
                final_indices_combined = np.concatenate([final_indices_x, final_indices_y])
                x_data_combined = x_data_flat[final_indices_combined]
                
                # 最终数据整理
                x_data = x_data_combined.reshape(-1, 1)  # (N, 1)
                y_data = mu_values_combined.flatten()    # (N,)
                
                print(f"\n=== 最终符号回归数据 ===")
                print(f"x_data (gamma): 形状={x_data.shape}, 范围=[{x_data.min():.6f}, {x_data.max():.6f}]")
                print(f"y_data (mu): 形状={y_data.shape}, 范围=[{y_data.min():.6f}, {y_data.max():.6f}]")
                print(f"数据质量检查: 全部有限值={np.all(np.isfinite(x_data)) and np.all(np.isfinite(y_data))}")
                
                # 验证数据长度一致性
                if len(x_data) != len(y_data):
                    print("错误: x和y数据长度不匹配")
                    return 1e10
                    
                print("✓ 改进的mu提取逻辑完成，数据已准备用于符号回归")
                
            else:
                print("错误: 'mu' not found in parameter names")
                return 1e10
        return NN_port(y_data, x_data, preset=preset, allowed_functions=allowed_functions, config=cfg)
    
    if len(SR_list) == 0: #ordinary solver
        sol = optimize_with_timeout(train_loss_func, init_params, tree.calculator.get_constr_dict_list(), prev_sol_best={"fun":1e-3, "nit":5}, verbose=True)
    else:
        params_name = list(tree.calculator.sp_unknown_quantities.keys())
        #--- hyper-params---
        l2_reg_coef = 1e-2
        tol_w = 0.005
        tree.calculator.upd_local_dict()
        
        # init_params[1] = 9.81
        SR_loss_func(init_params)
        
        # 这里应该有实际的优化过程，但目前先用初始参数
        # sol = optimize_with_timeout(SR_loss_func, init_params, tree.calculator.get_constr_dict_list(), prev_sol_best={"fun":1e-3, "nit":5}, verbose=True)
        return
    
    # ---return infos---
    if verbose:
        print('Loss', sol['fun'])
    
    ret_sol = dict()
    ret_sol['train_loss'] = sol['fun']
    ret_sol['valid_loss'] = valid_loss_func(sol['x'])
    ret_sol['deci_list'] = deci_list
    params_name = list(map(str, tree.calculator.sp_unknown_quantities.keys())) #must return str (instead of sympy symbol), for pprint.
    ret_sol['params'] = dict(zip(params_name, sol['x']))
    ret_sol['time'] = sol['time']
    ret_sol['nit'] = sol['nit']
    ret_sol['status'] = sol['status']

    if ret_sol['status'] == "Success": # record detailed infos of train loss.
        ret_sol['train_mse_list'] = mse_list_train(sol['x'])
        ret_sol['valid_mse_list'] = mse_list_valid(sol['x'])

    return ret_sol

def para_test(cfg, root_dir, deci_lists, n_jobs=10, time_limit=None, preset=None, allowed_functions=None):
    n_parts = 10
    n_dicts_per_part = len(deci_lists)//n_parts
    sols = []
    start_time = time_module.time()  
    for i in range(len(deci_lists)):
        sol = single_test(cfg, root_dir, deci_lists[i], deleted_coef=[], init_params=None, preset=preset, allowed_functions=allowed_functions)
        if sol is not None:
            sols.append(sol)
    # Parallel processing (commented out for test)
    # for i in range(n_parts):
    #     if i < n_parts-1:
    #         part_i_deci_lists = deci_lists[i*n_dicts_per_part:(i+1)*n_dicts_per_part]
    #     else:
    #         part_i_deci_lists = deci_lists[i*n_dicts_per_part:]
    #     sols += Parallel(n_jobs=n_jobs)(delayed(single_test)(cfg, root_dir, deci_list) for deci_list in part_i_deci_lists)
        
    #     tot_time_elapsed = time_module.time() - start_time  
    #     print(f"Part {i} finished, {len(part_i_deci_lists)=}, {tot_time_elapsed=}", flush=1)
    #     if time_limit and tot_time_elapsed > time_limit:
    #         print("timeout.")
    #         break
    return sols

@timing_with_return
def brute_force_search(cfg, root_dir, dataname_tuple, datafold_tuple=(0,1), out_file=None, n_jobs=10, verbose=False, time_limit=None, preset=None, allowed_functions=None):
    dummy_tree = LightTree(cfg, root_dir)
    deci_lists = dummy_tree.generate_all_possibilities()
    random.shuffle(deci_lists)
    print(f"{len(deci_lists)=},{n_jobs=}")

    sols = para_test(cfg, root_dir, deci_lists, n_jobs, time_limit, preset=preset, allowed_functions=allowed_functions)
    top_3_sols = sorted(sols, key=lambda sol:sol['valid_loss'])[:3]
    opt_sol = top_3_sols[0]
    total_time = time_to_str(sum([sol['time'] for sol in sols]))
    print("------result------")
    print("Search terminated successfully. Time used: ", total_time)
    print(f"Final result:")
    pprint.pprint(opt_sol)

    if verbose:
        print("------solution details------")
        for sol in sols:
            print(sol)
    return top_3_sols

def k_fold_cv_bfs(cfg, root_dir, dataname_tuple, k=1, out_file=None, n_jobs=10, verbose=False, time_limit=None, preset=None, allowed_functions=None):
    search_time_list = []
    top_3_nodes_list = []
    for i in range(k):
        print(f"=== {i+1}-th fold ===", flush=1)
        datafold_tuple = (i,k)
        top_3_sols, search_time = brute_force_search(cfg, root_dir, dataname_tuple, datafold_tuple, out_file, n_jobs, verbose, time_limit=time_limit, preset=preset, allowed_functions=allowed_functions)
        for sol in top_3_sols:
            pprint.pprint(sol)
        search_time_list.append(search_time)
    print("Search time mean={}, std={}".format(np.mean(search_time_list), np.std(search_time_list)))
