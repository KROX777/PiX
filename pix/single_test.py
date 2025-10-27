"""
Single Test Script for PiX
Users can input hypothesis IDs and run targeted tests.
"""

import sys
import os
import time
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

# Configure a shared log file for this session and initialize SR4MDL logger early
SHARED_LOG_DIR = os.path.join(ROOT_DIR, 'results', 'logs', time.strftime('%Y%m%d_%H%M%S'))
SHARED_LOG_FILE = os.path.join(SHARED_LOG_DIR, 'pix.log')
os.makedirs(SHARED_LOG_DIR, exist_ok=True)
# If not already set by a parent process, set shared log file for downstream modules (e.g., SR4MDL.search)
os.environ.setdefault('PIX_LOG_FILE', SHARED_LOG_FILE)

# Import SR4MDL.search early so it picks up PIX_LOG_FILE and initializes the shared logger
try:
    from pix.methods.SR4MDL import search as _sr4mdl_search  # noqa: F401
except Exception:
    # Defer errors until the mode that needs it; not all modes import SR4MDL
    _sr4mdl_search = None

# Set up console logging for immediate feedback; use the shared 'sr4mdl' logger tree to unify outputs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('sr4mdl.single_test')
logger.propagate = True  # ensure it bubbles up to the 'sr4mdl' handlers initialized by SR4MDL.search
logger.info(f"Shared log file: {os.environ.get('PIX_LOG_FILE')}")

def validate_hypothesis_combination(cfg, hypothesis_ids):
    from hypotheses_tree import LightTree
    light_tree = LightTree(cfg, ROOT_DIR)
    all_valid_combinations = light_tree.generate_all_possibilities()

    if hypothesis_ids in all_valid_combinations:
        return True, "组合有效", []

    error_messages = []
    suggestions = []
    node_to_branch = {}  # 记录每个根分支下选择的节点
    
    for hyp_id in hypothesis_ids:
        if hyp_id == 0:
            continue  # 跳过根节点

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

def find_root_branch(light_tree, node_id):
    current_id = node_id
    while current_id != 0:
        node = light_tree.nodes[current_id]
        if node.father_node == 0:  # 这是根的直接子节点
            return current_id
        current_id = node.father_node
    return 0

def get_valid_combinations_sample(cfg, max_samples=5):
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
    print(f"\n=== 运行单个测试 (模式: {mode}) ===")
    print(f"假设编号组合: {hypothesis_ids}")
    
    try:
        import gc
        clear_module_cache()
        gc.collect()

        if mode.lower() == "bfsearch":
            from pix.methods.BFSearch import single_test
        elif mode.lower() == "bfsearch_new_sr":
            from pix.methods.BFSearch_new_SR import single_test
        else:
            print(f"错误: 未知的模式 '{mode}'，支持的模式: 'bfsearch', 'bfsearch_new_sr'")
            return None
            
        from hypotheses_tree import HypothesesTree
        
        # 重新加载配置以确保是干净的状态
        fresh_cfg = reload_config()
        if fresh_cfg is None:
            print("错误: 无法重新加载配置")
            return None

        logger.info("开始单次测试 | 模式=%s | 假设=%s", mode, hypothesis_ids)

        result = single_test(
            cfg=fresh_cfg,
            root_dir=root_dir,
            deci_list=hypothesis_ids,
            deleted_coef=[],
            init_params=None,
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
            try:
                logger.info(
                    "测试完成 | 训练损失=%.6f | 验证损失=%.6f | 假设组合=%s | 用时=%.2fs | 迭代=%s | 状态=%s",
                    result.get('train_loss', float('nan')),
                    result.get('valid_loss', float('nan')),
                    result.get('deci_list'),
                    result.get('time', float('nan')),
                    result.get('nit'),
                    result.get('status')
                )
            except Exception:
                pass
            
            if verbose and 'params' in result:
                print("\n参数值:")
                pprint.pprint(result['params'])
                
            if 'train_mse_list' in result:
                print(f"\n训练 MSE 列表: {result['train_mse_list']}")
                print(f"验证 MSE 列表: {result['valid_mse_list']}")
                
        else:
            print("测试失败，返回结果为 None")
            logger.warning("测试失败，返回结果为 None | 模式=%s | 假设=%s", mode, hypothesis_ids)
            
        return result
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        logger.exception("测试过程中出现错误: %s", e)
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
    print("\n=== 选择测试模式 ===")
    print("1. BFSearch")
    print("2. BFSearch_new_SR")
    
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
    print("=== PiX 单个测试工具 ===")
    mode = select_test_mode()
    if mode is None:
        return

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
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    workspace_dir = Path.cwd()
    logger.info(f"工作空间: {workspace_dir}")
    logger.info(f"项目根目录: {ROOT_DIR}")
    
    interactive_single_test()

def test_with_specific_hypotheses(hypothesis_ids, mode="bfsearch"):
    return run_single_test_with_hypotheses(
        cfg=None,  # 传入None，函数内部会重新加载配置
        root_dir=ROOT_DIR,
        hypothesis_ids=hypothesis_ids,
        mode=mode,
        verbose=True
    )

if __name__ == "__main__":
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