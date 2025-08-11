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

from methods.BFSearch import single_test
from hypotheses_tree import HypothesesTree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_test_with_hypotheses(cfg, root_dir, hypothesis_ids, verbose=True):
    """
    运行单个测试，使用指定的假设编号组合
    
    Args:
        cfg: 配置对象
        root_dir: 项目根目录
        hypothesis_ids: 要测试的假设编号列表，例如 [1, 3, 5]
        verbose: 是否显示详细信息
    
    Returns:
        dict: 测试结果字典
    """
    print(f"\n=== 运行单个测试 ===")
    print(f"假设编号组合: {hypothesis_ids}")
    
    try:
        # 运行单个测试
        result = single_test(
            cfg=cfg,
            root_dir=root_dir,
            deci_list=hypothesis_ids,
            deleted_coef=[],
            init_params=None,
            STR_iter_max=4,
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

def interactive_single_test():
    """
    交互式单个测试函数
    允许用户输入假设编号组合并运行测试
    """
    print("=== PiX 单个测试工具 ===")
    
    # 先加载配置显示可用假设
    try:
        config_path = os.path.join(project_root, "cfg", "config.yaml")
        cfg = OmegaConf.load(config_path)
        
        # 加载问题配置
        problem_name = "Fluid Mechanics"
        problem_config_path = os.path.join(project_root, "cfg", "problem", f"{problem_name}.yaml")
        
        if os.path.exists(problem_config_path):
            problem_cfg = OmegaConf.load(problem_config_path)
            cfg.problem = problem_cfg
        else:
            print(f"错误: 找不到问题配置文件 {problem_config_path}")
            return
        
        # 显示可用假设
        show_available_hypotheses(cfg)
        
    except Exception as e:
        print(f"加载配置时出错: {e}")
        return
    
    print("请输入要测试的假设编号组合")
    print("示例: 1,3,5 或 [1, 3, 5]")
    print("输入 'q' 或 'quit' 退出")
    
    while True:
        try:
            user_input = input("\n请输入假设编号 (用逗号分隔): ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("退出程序")
                break
            
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
            
            # 运行测试
            result = run_single_test_with_hypotheses(
                cfg=cfg,
                root_dir=ROOT_DIR,
                hypothesis_ids=hypothesis_ids,
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

def test_with_specific_hypotheses(hypothesis_ids):
    """
    直接测试指定的假设编号组合
    
    Args:
        hypothesis_ids: 假设编号列表，例如 [1, 3, 5]
    
    Returns:
        dict: 测试结果
    """
    # 加载配置
    config_path = os.path.join(project_root, "cfg", "config.yaml")
    cfg = OmegaConf.load(config_path)
    
    # 加载问题配置 - 默认使用 Fluid Mechanics
    problem_name = "Fluid Mechanics"  # 默认问题类型
    problem_config_path = os.path.join(project_root, "cfg", "problem", f"{problem_name}.yaml")
    
    if os.path.exists(problem_config_path):
        problem_cfg = OmegaConf.load(problem_config_path)
        cfg.problem = problem_cfg
    else:
        print(f"错误: 找不到问题配置文件 {problem_config_path}")
        return None
    
    return run_single_test_with_hypotheses(
        cfg=cfg,
        root_dir=ROOT_DIR,
        hypothesis_ids=hypothesis_ids,
        verbose=True
    )

if __name__ == "__main__":
    # 可以选择运行交互式模式或直接测试
    import sys
    
    if len(sys.argv) > 1:
        # 如果有命令行参数，尝试解析为假设编号
        try:
            hypothesis_ids = [int(x) for x in sys.argv[1].split(',')]
            print(f"从命令行参数获取假设编号: {hypothesis_ids}")
            result = test_with_specific_hypotheses(hypothesis_ids)
        except ValueError:
            print("命令行参数格式错误，请使用逗号分隔的数字，例如: python single_test.py 1,3,5")
            print("切换到交互式模式...")
            interactive_single_test()
    else:
        # 交互式模式
        interactive_single_test()