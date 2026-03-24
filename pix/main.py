"""
Main entry point for PiX PDE discovery system.

This module provides the primary interface for running PDE system discovery
using different algorithms (BFSearch, MCTS, etc.).
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import warnings

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

# Import core components
from methods.BFSearch import k_fold_cv_bfs
from methods.MCTS import MCTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_bfsearch(cfg: DictConfig) -> None:
    """
    Execute BFSearch algorithm for PDE discovery.
    
    Args:
        cfg: Hydra configuration object containing search parameters.
    """
    logger.info("Starting BFSearch algorithm")
    print("\n=== Running BFSearch ===")
    
    k_fold_cv_bfs(
        cfg=cfg,
        root_dir=ROOT_DIR,
        dataname_tuple=(cfg.dataset_path, "COSMOL"),
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
        time_limit=cfg.time_limit
    )
    
    logger.info("BFSearch completed successfully")


def run_mcts(cfg: DictConfig) -> Optional[dict]:
    """
    Execute MCTS algorithm for PDE discovery.
    
    Args:
        cfg: Hydra configuration object containing search parameters.
        
    Returns:
        Results dictionary from MCTS search, or None if failed.
    """
    logger.info("Starting MCTS algorithm")
    print("\n=== Running MCTS ===")
    
    tree = MCTS(
        cfg=cfg, 
        root_dir=ROOT_DIR, 
        exploration_weight=cfg.exploration_weight,
        max_rollout=cfg.max_rollout,
        n_jobs=cfg.n_jobs,
        verbose_level=cfg.verbose_level
    )
    results = tree.k_fold_cv_search()
    
    logger.info("MCTS completed successfully")
    return results


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for PiX PDE discovery system.
    
    Configurable via Hydra with parameters in pix/cfg/config.yaml.
    Supports multiple discovery algorithms (bfsearch, mcts, etc.).
    
    Args:
        cfg: Hydra configuration object.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Log execution environment
    workspace_dir = Path.cwd()
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Project root: {ROOT_DIR}")
    
    # Parse method selection
    method = cfg.method.lower()
    
    # Print system information
    print("=" * 50)
    print("=== PiX Scientific Discovery System ===")
    print("=" * 50)
    print(f"Dataset: {cfg.dataset_path}")
    print(f"Problem type: {cfg.problem.name}")
    print(f"Method: {method}")
    print("=" * 50 + "\n")

    # Execute selected method
    if method == 'bfsearch':
        run_bfsearch(cfg)
    elif method == 'mcts':
        try:
            results = run_mcts(cfg)
            if results:
                print(f"\nMCTS completed successfully, found {len(results)} results")
            else:
                print("\nMCTS execution completed with no results")
        except Exception as e:
            logger.error(f"MCTS execution failed: {e}", exc_info=True)
            print(f"MCTS execution failed: {e}")
            raise
    else:
        logger.error(f"Unknown method: {method}")
        print(f"Unknown method: {method}")
        print("Supported methods: 'bfsearch', 'mcts'")
        return
    
    print("\n" + "=" * 50)
    print("=== Program completed ===")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

