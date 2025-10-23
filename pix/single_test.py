"""
Single Test Script for PiX
Users can input hypothesis IDs and run targeted tests.
"""

import sys
import os
import hydra
import logging
from pathlib import Path
from omegaconf import OmegaConf
import warnings
import numpy as np
import pprint

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

# Import and initialize logger to match search.py
try:
    from nd2py.utils import init_logger
    import time
    # Create log directory and file for single_test
    log_dir = os.path.join('./results/single_test')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.log")
    init_logger('sr4mdl', 'single_test', log_file)
except ImportError:
    # Fallback if nd2py not available
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger('sr4mdl.single_test')

def validate_hypothesis_combination(cfg, hypothesis_ids):
    """
    验证假设组合是否有效
    
    Args:
        cfg: 配置对象
        hypothesis_ids: 假设编号列表
        
    Returns:
        tuple: (is_valid, error_message, suggested_combinations)
    """
    try:
        from hypotheses_tree import LightTree
        
        # 创建轻量级树来验证
        light_tree = LightTree(cfg, ROOT_DIR)
        
        # 生成所有有效的组合
        all_valid_combinations = light_tree.generate_all_possibilities()
        
        # 检查用户输入的组合是否在有效组合中
        if hypothesis_ids in all_valid_combinations:
            return True, "组合有效", []
        
        # 如果无效，分析原因并提供建议
        error_messages = []
        suggestions = []
        
        # 检查是否有重复的分支选择
        node_to_branch = {}  # 记录每个根分支下选择的节点
        
        for hyp_id in hypothesis_ids:
            if hyp_id == 0:
                continue  # 跳过根节点
                
            # 找到这个节点属于哪个根分支
            root_branch = find_root_branch(light_tree, hyp_id)
            if root_branch not in node_to_branch:
                node_to_branch[root_branch] = []
            node_to_branch[root_branch].append(hyp_id)
        
        # 检查每个分支是否只有一个有效路径
        for branch, selected_nodes in node_to_branch.items():
            branch_paths = light_tree._get_all_paths_from_node(branch)
            
            # 检查选择的节点是否构成一个有效路径
            valid_path_found = False
            for path in branch_paths:
                if set(selected_nodes).issubset(set(path)):
                    # 检查是否选择了完整路径（到叶子节点）
                    if selected_nodes[-1] == path[-1]:  # 最后一个是叶子节点
                        valid_path_found = True
                        break
            
            if not valid_path_found:
                branch_node = light_tree.nodes[branch]
                error_messages.append(f"分支 '{branch_node.name}' (ID: {branch}) 的选择无效")
                
                # 提供该分支的有效选择
                for path in branch_paths:
                    path_names = [light_tree.nodes[nid].name for nid in path]
                    suggestions.append(f"分支 {branch_node.name}: {' -> '.join(path_names)} (IDs: {path})")
        
        # 检查是否遗漏了某些必需的分支
        root_children = light_tree.root.children_nodes
        selected_branches = set(node_to_branch.keys())
        missing_branches = set(root_children) - selected_branches
        
        if missing_branches:
            for branch_id in missing_branches:
                branch_node = light_tree.nodes[branch_id]
                error_messages.append(f"缺少分支 '{branch_node.name}' (ID: {branch_id}) 的选择")
        
        error_message = "; ".join(error_messages) if error_messages else "未知验证错误"
        
        return False, error_message, suggestions[:5]  # 最多显示5个建议
        
    except Exception as e:
        return False, f"验证过程中出错: {e}", []

def find_root_branch(light_tree, node_id):
    """
    找到节点所属的根分支
    
    Args:
        light_tree: LightTree 实例
        node_id: 节点ID
        
    Returns:
        int: 根分支的ID
    """
    current_id = node_id
    while current_id != 0:
        node = light_tree.nodes[current_id]
        if node.father_node == 0:  # 这是根的直接子节点
            return current_id
        current_id = node.father_node
    return 0

def get_valid_combinations_sample(cfg, max_samples=5):
    """
    获取一些有效组合的示例
    
    Args:
        cfg: 配置对象
        max_samples: 最大示例数量
        
    Returns:
        list: 有效组合示例
    """
    try:
        from hypotheses_tree import LightTree
        
        light_tree = LightTree(cfg, ROOT_DIR)
        all_valid = light_tree.generate_all_possibilities()
        
        # 返回前几个示例
        samples = []
        for i, combo in enumerate(all_valid[:max_samples]):
            combo_names = [light_tree.nodes[nid].name for nid in combo if nid != 0]
            samples.append({
                'ids': combo,
                'names': combo_names,
                'description': ' + '.join(combo_names)
            })
        
        return samples
    except Exception as e:
        print(f"获取有效组合示例时出错: {e}")
        return []

def clear_module_cache():
    """
    清理Python模块缓存，确保重新导入时获得干净状态
    """
    import sys
    modules_to_clear = []
    
    # 找到需要清理的模块
    for module_name in sys.modules.keys():
        if (module_name.startswith('pix.') or 
            module_name.startswith('methods.') or
            module_name.startswith('hypotheses_') or
            module_name.startswith('calculator') or
            module_name.startswith('data_loader')):
            modules_to_clear.append(module_name)
    
    # 清理模块
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

def reload_config():
    """
    重新加载配置文件，确保获得干净的配置状态
    
    Returns:
        OmegaConf: 配置对象，如果失败返回None
    """
    try:
        # 重新加载主配置
        config_path = os.path.join(project_root, "cfg", "config.yaml")
        cfg = OmegaConf.load(config_path)
        
        # 重新加载问题配置
        problem_name = "Fluid Mechanics"
        problem_config_path = os.path.join(project_root, "cfg", "problem", f"{problem_name}.yaml")
        
        if os.path.exists(problem_config_path):
            problem_cfg = OmegaConf.load(problem_config_path)
            cfg.problem = problem_cfg
        else:
            print(f"错误: 找不到问题配置文件 {problem_config_path}")
            return None
            
        return cfg
    except Exception as e:
        print(f"重新加载配置时出错: {e}")
        return None

def run_single_test_with_hypotheses(cfg, root_dir, hypothesis_ids, mode="bfsearch", verbose=True):
    """
    运行单个测试，使用指定的假设编号组合
    每次测试都会重新初始化所有组件
    
    Args:
        cfg: 配置对象
        root_dir: 项目根目录
        hypothesis_ids: 要测试的假设编号列表，例如 [1, 3, 5]
        mode: 测试模式，"bfsearch" 或 "bfsearch_new_sr"
        verbose: 是否显示详细信息
    
    Returns:
        dict: 测试结果字典
    """
    print(f"\n=== 运行单个测试 (模式: {mode}) ===")
    print(f"假设编号组合: {hypothesis_ids}")
    print("重新初始化所有组件...")
    
    try:
        # 清理可能存在的全局状态和模块缓存
        import gc
        print("清理内存和模块缓存...")
        clear_module_cache()
        gc.collect()
        
        # 根据模式导入不同的 single_test 函数
        if mode.lower() == "bfsearch":
            from methods.BFSearch import single_test
            print("使用 BFSearch 模式")
        elif mode.lower() == "bfsearch_new_sr":
            from pix.methods.BFSearch_new_SR import single_test
            print("使用 BFSearch_new_SR 模式")
        else:
            print(f"错误: 未知的模式 '{mode}'，支持的模式: 'bfsearch', 'bfsearch_new_sr'")
            return None
            
        from hypotheses_tree import HypothesesTree
        
        # 重新加载配置以确保是干净的状态
        fresh_cfg = reload_config()
        if fresh_cfg is None:
            print("错误: 无法重新加载配置")
            return None
        
        print("开始测试...")
        
        # 运行单个测试，每次都从头开始
        result = single_test(
            cfg=fresh_cfg,
            root_dir=root_dir,
            deci_list=hypothesis_ids,
            deleted_coef=[],  # 每次都是空的删除系数列表
            init_params=None,  # 每次都重新随机初始化参数
            verbose=verbose
        )
        
        if result is not None:
            print("\n=== 测试结果 ===")
            print(f"训练损失: {result['train_loss']:.6f}")
            print(f"验证损失: {result['valid_loss']:.6f}")
            print(f"假设组合: {result['deci_list']}")
            print(f"用时: {result['time']:.2f} 秒")
            print(f"迭代次数: {result['nit']}")
            print(f"状态: {result['status']}")
            
            if verbose and 'params' in result:
                print("\n参数值:")
                pprint.pprint(result['params'])
                
            if 'train_mse_list' in result:
                print(f"\n训练 MSE 列表: {result['train_mse_list']}")
                print(f"验证 MSE 列表: {result['valid_mse_list']}")
                
        else:
            print("测试失败，返回结果为 None")
            
        return result
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def show_available_hypotheses(cfg):
    """
    显示所有可用的假设及其描述
    
    Args:
        cfg: 配置对象
    """
    print("\n=== 可用假设列表 ===")
    
    if 'hypotheses' in cfg.problem:
        for hypothesis in cfg.problem.hypotheses:
            print(f"ID: {hypothesis.id:2d} | 名称: {hypothesis.name}")
            if hypothesis.get('related_variables'):
                print(f"    相关变量: {hypothesis.related_variables}")
            if hypothesis.get('definitions'):
                print(f"    定义: {hypothesis.definitions}")
            if hypothesis.get('require_sr'):
                print(f"    需要符号回归: {hypothesis.require_sr}")
            print()
    else:
        print("错误: 配置中没有找到假设列表")

def select_test_mode():
    """
    让用户选择测试模式
    
    Returns:
        str: 选择的模式 ("bfsearch" 或 "bfsearch_new_sr")
    """
    print("\n=== 选择测试模式 ===")
    print("1. BFSearch (原版)")
    print("2. BFSearch_new_SR (新版，支持符号回归)")
    
    while True:
        try:
            choice = input("\n请选择模式 (1 或 2): ").strip()
            
            if choice == "1":
                return "bfsearch"
            elif choice == "2":
                return "bfsearch_new_sr"
            else:
                print("请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n程序已中断")
            return None

def interactive_single_test():
    """
    交互式单个测试函数
    允许用户输入假设编号组合并运行测试
    每次测试都会重新初始化所有组件
    """
    print("=== PiX 单个测试工具 ===")
    
    # 选择测试模式
    mode = select_test_mode()
    if mode is None:
        return
    
    # 先加载配置显示可用假设
    try:
        cfg = reload_config()
        if cfg is None:
            print("错误: 无法加载配置")
            return
        
        # 显示可用假设
        show_available_hypotheses(cfg)
        
    except Exception as e:
        print(f"加载配置时出错: {e}")
        return
    
    print(f"\n当前模式: {mode}")
    print("请输入要测试的假设编号组合")
    print("示例: 1,3,5 或 [1, 3, 5]")
    print("输入 'q' 或 'quit' 退出")
    print("输入 'mode' 切换模式")
    print("注意: 每次测试都会重新初始化所有组件，确保结果的独立性")
    
    while True:
        try:
            user_input = input(f"\n[{mode}] 请输入假设编号 (用逗号分隔): ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("退出程序")
                break
            
            if user_input.lower() == 'mode':
                new_mode = select_test_mode()
                if new_mode:
                    mode = new_mode
                    print(f"已切换到模式: {mode}")
                continue
            
            if not user_input:
                continue
            
            # 解析用户输入
            if user_input.startswith('[') and user_input.endswith(']'):
                # 处理列表格式输入
                user_input = user_input[1:-1]
            
            # 分割并转换为整数
            hypothesis_ids = [int(x.strip()) for x in user_input.split(',') if x.strip()]
            
            if not hypothesis_ids:
                print("错误: 未输入有效的假设编号")
                continue
                
            print(f"解析的假设编号: {hypothesis_ids}")
            
            # 每次测试都重新运行，确保独立性
            result = run_single_test_with_hypotheses(
                cfg=None,  # 传入None，函数内部会重新加载配置
                root_dir=ROOT_DIR,
                hypothesis_ids=hypothesis_ids,
                mode=mode,
                verbose=True
            )
            
            if result:
                print(f"\n✓ 测试完成! 训练损失: {result['train_loss']:.6f}, 验证损失: {result['valid_loss']:.6f}")
            else:
                print("✗ 测试失败")
                
        except ValueError as e:
            print(f"输入格式错误: {e}")
            print("请输入有效的数字，用逗号分隔，例如: 1,3,5")
        except KeyboardInterrupt:
            print("\n测试已中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main_hydra(cfg):
    """
    使用 Hydra 配置的主函数
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    workspace_dir = Path.cwd()
    logger.info(f"工作空间: {workspace_dir}")
    logger.info(f"项目根目录: {ROOT_DIR}")
    
    # 这里可以添加从配置文件读取假设编号的逻辑
    # 或者调用交互式函数
    interactive_single_test()

def test_with_specific_hypotheses(hypothesis_ids, mode="bfsearch"):
    """
    直接测试指定的假设编号组合
    每次调用都会重新初始化所有组件
    
    Args:
        hypothesis_ids: 假设编号列表，例如 [1, 3, 5]
        mode: 测试模式，"bfsearch" 或 "bfsearch_new_sr"
    
    Returns:
        dict: 测试结果
    """
    return run_single_test_with_hypotheses(
        cfg=None,  # 传入None，函数内部会重新加载配置
        root_dir=ROOT_DIR,
        hypothesis_ids=hypothesis_ids,
        mode=mode,
        verbose=True
    )

if __name__ == "__main__":
    # 可以选择运行交互式模式或直接测试
    import sys
    
    if len(sys.argv) > 1:
        # 如果有命令行参数，尝试解析为假设编号
        try:
            # 检查是否指定了模式
            mode = "bfsearch"  # 默认模式
            if len(sys.argv) > 2 and sys.argv[2].lower() in ["bfsearch", "bfsearch_new_sr"]:
                mode = sys.argv[2].lower()
            
            hypothesis_ids = [int(x) for x in sys.argv[1].split(',')]
            print(f"从命令行参数获取假设编号: {hypothesis_ids}")
            print(f"使用模式: {mode}")
            result = test_with_specific_hypotheses(hypothesis_ids, mode)
        except ValueError:
            print("命令行参数格式错误，请使用: python single_test.py 1,3,5 [bfsearch/bfsearch_new_sr]")
            print("切换到交互式模式...")
            interactive_single_test()
    else:
        # 交互式模式
        interactive_single_test()