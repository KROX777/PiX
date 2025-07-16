import sys
import os
import hydra
import logging
from pathlib import Path
from omegaconf import OmegaConf
import warnings
import numpy as np
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.numpy_utils import pooling
import sympy as sp

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

from methods.BFSearch import k_fold_cv_bfs
from methods.MCTS import MCTS
from hypotheses_tree import HypothesesTree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_bfsearch(cfg):
    print("\n=== 运行 BFSearch ===")
    
    k_fold_cv_bfs(
        cfg=cfg,
        root_dir=ROOT_DIR,
        dataname_tuple=(cfg.dataset_path, "COSMOL"),
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
        time_limit=cfg.time_limit
    )

def run_mcts(cfg):
    print("\n=== 运行 MCTS ===")
    tree = MCTS(cfg=cfg, 
                root_dir=ROOT_DIR, 
                exploration_weight=cfg.exploration_weight,
                max_rollout=cfg.max_rollout,
                n_jobs=cfg.n_jobs,
                verbose_level=cfg.verbose_level)
    tree.k_fold_cv_search()

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    workspace_dir = Path.cwd()
    logger.info(f"工作空间: {workspace_dir}")
    logger.info(f"项目根目录: {ROOT_DIR}")
    method = cfg.method.lower()
    
    print("=== PiX 科学发现系统 ===")
    print(f"数据集: {cfg.dataset_path}")
    print(f"问题类型: {cfg.problem.name}")
    print(f"方法: {method}")

    if method == 'bfsearch':
        run_bfsearch(cfg)
    elif method == 'mcts':
        try:
            results = run_mcts(cfg)
            print(f"\nMCTS 成功完成，找到 {len(results)} 个结果")
        except Exception as e:
            print(f"MCTS 运行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"未知的方法: {method}")
        print("支持的方法: 'bfsearch', 'mcts'")
        return
    
    print("\n=== 程序完成 ===")

if __name__ == "__main__":
    main()
