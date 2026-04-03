"""
Single test script for targeted PDE discovery evaluation.

This module provides utilities for running individual hypothesis tests
and debugging specific hypothesis combinations without running full searches.

Features:
    - Validate hypothesis combinations
    - Run single or combined hypothesis tests
    - Interactive testing interface
    - Module cache clearing for clean test execution
    - Detailed result logging
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import hydra
from omegaconf import OmegaConf, DictConfig
import warnings
import numpy as np
import pprint
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

# # Import SR4MDL.search early so it picks up PIX_LOG_FILE and initializes the shared logger
try:
    from pix.methods.SR4MDL import search as _sr4mdl_search  # noqa: F401
except Exception:
    print("Warning: SR4MDL module not available, logging may be incomplete")

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
        return True, "Combination is valid", []

    error_messages = []
    suggestions = []
    node_to_branch = {}  # Record which node is selected for each root branch
    
    for hyp_id in hypothesis_ids:
        if hyp_id == 0:
            continue  # Skip root node

        root_branch = find_root_branch(light_tree, hyp_id)
        if root_branch not in node_to_branch:
            node_to_branch[root_branch] = []
        node_to_branch[root_branch].append(hyp_id)
    
    # Check if each branch has only one valid path
    for branch, selected_nodes in node_to_branch.items():
        branch_paths = light_tree._get_all_paths_from_node(branch)
        
        # Check if selected nodes form a valid path
        valid_path_found = False
        for path in branch_paths:
            if set(selected_nodes).issubset(set(path)):
                # Check if a complete path is selected (to leaf node)
                if selected_nodes[-1] == path[-1]:  # Last node is leaf node
                    valid_path_found = True
                    break
        
        if not valid_path_found:
            branch_node = light_tree.nodes[branch]
            error_messages.append(f"Branch '{branch_node.name}' (ID: {branch}) selection is invalid")
            
            # Provide valid selections for this branch
            for path in branch_paths:
                path_names = [light_tree.nodes[nid].name for nid in path]
                suggestions.append(f"Branch {branch_node.name}: {' -> '.join(path_names)} (IDs: {path})")
    
    # Check if any required branches are missing
    root_children = light_tree.root.children_nodes
    selected_branches = set(node_to_branch.keys())
    missing_branches = set(root_children) - selected_branches
    
    if missing_branches:
        for branch_id in missing_branches:
            branch_node = light_tree.nodes[branch_id]
            error_messages.append(f"Missing branch '{branch_node.name}' (ID: {branch_id}) selection")
    
    error_message = "; ".join(error_messages) if error_messages else "Unknown validation error"
    
    return False, error_message, suggestions[:5]  # Show at most 5 suggestions

def find_root_branch(light_tree, node_id):
    current_id = node_id
    while current_id != 0:
        node = light_tree.nodes[current_id]
        if node.father_node == 0:  # This is direct child of root
            return current_id
        current_id = node.father_node
    return 0

def get_valid_combinations_sample(cfg, max_samples=5):
    try:
        from hypotheses_tree import LightTree
        
        light_tree = LightTree(cfg, ROOT_DIR)
        all_valid = light_tree.generate_all_possibilities()
        
        # Return first few samples
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
        print(f"Error retrieving valid combination samples: {e}")
        return []

def clear_module_cache():
    """
    Clear Python module cache to ensure clean state on reimport
    """
    import sys
    modules_to_clear = []
    
    # Find modules to clear
    for module_name in sys.modules.keys():
        if (module_name.startswith('pix.') or 
            module_name.startswith('methods.') or
            module_name.startswith('hypotheses_') or
            module_name.startswith('calculator') or
            module_name.startswith('data_loader')):
            modules_to_clear.append(module_name)
    
    # Clear modules
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

def reload_config():
    """
    Reload config file to ensure clean state
    
    Returns:
        OmegaConf: Config object, returns None if failed
    """
    try:
        # Reload main config
        config_path = os.path.join(project_root, "cfg", "config.yaml")
        cfg = OmegaConf.load(config_path)
        
        # Reload problem config
        problem_name = "Fluid Mechanics"
        problem_config_path = os.path.join(project_root, "cfg", "problem", f"{problem_name}.yaml")
        
        if os.path.exists(problem_config_path):
            problem_cfg = OmegaConf.load(problem_config_path)
            cfg.problem = problem_cfg
        else:
            print(f"Error: Cannot find problem config file {problem_config_path}")
            return None
            
        return cfg
    except Exception as e:
        print(f"Error reloading config: {e}")
        return None

def run_single_test_with_hypotheses(cfg, root_dir, hypothesis_ids, mode="bfsearch", verbose=True):
    print(f"\n=== Running single test (mode: {mode}) ===")
    print(f"Hypothesis ID combination: {hypothesis_ids}")
    
    try:
        import gc
        clear_module_cache()
        gc.collect()

        if mode.lower() == "bfsearch":
            from pix.methods.BFSearch import single_test
        elif mode.lower() == "bfsearch_new_sr":
            from pix.methods.BFSearch_new_SR import single_test
        else:
            print(f"Error: Unknown mode '{mode}'. Supported modes: 'bfsearch', 'bfsearch_new_sr'")
            return None
            
        from hypotheses_tree import HypothesesTree
        
        # Reload config to ensure clean state
        fresh_cfg = reload_config()
        if fresh_cfg is None:
            print("Error: Cannot reload config")
            return None

        logger.info("Starting single test | mode=%s | hypotheses=%s", mode, hypothesis_ids)

        result = single_test(
            cfg=fresh_cfg,
            root_dir=root_dir,
            deci_list=hypothesis_ids,
            deleted_coef=[],
            init_params=None,
            verbose=verbose
        )
        
        if result is not None:
            print("\n=== Test Results ===")
            print(f"Training loss: {result['train_loss']:.6f}")
            print(f"Validation loss: {result['valid_loss']:.6f}")
            print(f"Hypothesis combination: {result['deci_list']}")
            print(f"Time elapsed: {result['time']:.2f} seconds")
            print(f"Iterations: {result['nit']}")
            print(f"Status: {result['status']}")
            try:
                logger.info(
                    "Test completed | train_loss=%.6f | valid_loss=%.6f | hypothesis=%s | time=%.2fs | iterations=%s | status=%s",
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
                print("\nParameter values:")
                pprint.pprint(result['params'])
                
            if 'train_mse_list' in result:
                print(f"\nTraining MSE list: {result['train_mse_list']}")
                print(f"Validation MSE list: {result['valid_mse_list']}")
                
        else:
            print("Test failed, result is None")
            logger.warning("Test failed, result is None | mode=%s | hypotheses=%s", mode, hypothesis_ids)
            
        return result
        
    except Exception as e:
        print(f"Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
        logger.exception("Error occurred during testing: %s", e)
        return None

def show_available_hypotheses(cfg):
    """
    Display all available hypotheses and their descriptions.
    
    Args:
        cfg: Configuration object.
    """
    print("\n=== Available Hypotheses ===")
    
    if 'hypotheses' in cfg.problem:
        for hypothesis in cfg.problem.hypotheses:
            print(f"ID: {hypothesis.id:2d} | Name: {hypothesis.name}")
            if hypothesis.get('related_variables'):
                print(f"    Related variables: {hypothesis.related_variables}")
            if hypothesis.get('definitions'):
                print(f"    Definitions: {hypothesis.definitions}")
            if hypothesis.get('require_sr'):
                print(f"    Requires symbolic regression: {hypothesis.require_sr}")
            print()
    else:
        print("Error: Hypothesis list not found in configuration")

def select_test_mode():
    print("\n=== Select Test Mode ===")
    print("1. BFSearch")
    print("2. BFSearch_new_SR")
    
    while True:
        try:
            choice = input("\nSelect mode (1 or 2): ").strip()
            
            if choice == "1":
                return "bfsearch"
            elif choice == "2":
                return "bfsearch_new_sr"
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nProgram interrupted")
            return None

def interactive_single_test():
    print("=== PiX Single Test Tool ===")
    mode = select_test_mode()
    if mode is None:
        return

    try:
        cfg = reload_config()
        if cfg is None:
            print("Error: Cannot load configuration")
            return
        
        # Display available hypotheses
        show_available_hypotheses(cfg)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    print(f"\nCurrent mode: {mode}")
    print("Enter hypothesis ID combination to test")
    print("Examples: 1,3,5 or [1, 3, 5]")
    print("Enter 'q' or 'quit' to exit")
    print("Enter 'mode' to switch modes")
    print("Note: Each test reinitializes all components for independent results")
    
    while True:
        try:
            user_input = input(f"\n[{mode}] Enter hypothesis IDs (comma-separated): ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Exit program")
                break
            
            if user_input.lower() == 'mode':
                new_mode = select_test_mode()
                if new_mode:
                    mode = new_mode
                    print(f"Switched to mode: {mode}")
                continue
            
            if not user_input:
                continue
            
            # Parse user input
            if user_input.startswith('[') and user_input.endswith(']'):
                # Handle list format input
                user_input = user_input[1:-1]
            
            # Split and convert to integers
            hypothesis_ids = [int(x.strip()) for x in user_input.split(',') if x.strip()]
            
            if not hypothesis_ids:
                print("Error: No valid hypothesis IDs entered")
                continue
                
            print(f"Parsed hypothesis IDs: {hypothesis_ids}")
            
            # Rerun test each time to ensure independence
            result = run_single_test_with_hypotheses(
                cfg=None,  # Pass None to reload config internally
                root_dir=ROOT_DIR,
                hypothesis_ids=hypothesis_ids,
                mode=mode,
                verbose=True
            )
            
            if result:
                print(f"\n✓ Test completed! Training loss: {result['train_loss']:.6f}, Validation loss: {result['valid_loss']:.6f}")
            else:
                print("✗ Test failed")
                
        except ValueError as e:
            print(f"Input format error: {e}")
            print("Please enter valid numbers, comma-separated, e.g.: 1,3,5")
        except KeyboardInterrupt:
            print("\nTest interrupted")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main_hydra(cfg):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    workspace_dir = Path.cwd()
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Project root: {ROOT_DIR}")
    
    interactive_single_test()

def test_with_specific_hypotheses(hypothesis_ids, mode="bfsearch"):
    return run_single_test_with_hypotheses(
        cfg=None,
        root_dir=ROOT_DIR,
        hypothesis_ids=hypothesis_ids,
        mode=mode,
        verbose=True
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If command line args, try to parse as hypothesis IDs
        try:
            # Check if mode is specified
            mode = "bfsearch"  # Default mode
            if len(sys.argv) > 2 and sys.argv[2].lower() in ["bfsearch", "bfsearch_new_sr"]:
                mode = sys.argv[2].lower()
            
            hypothesis_ids = [int(x) for x in sys.argv[1].split(',')]
            print(f"Get hypothesis IDs from command line args: {hypothesis_ids}")
            print(f"Using mode: {mode}")
            result = test_with_specific_hypotheses(hypothesis_ids, mode)
        except ValueError:
            print("Command line arg format error. Usage: python single_test.py 1,3,5 [bfsearch/bfsearch_new_sr]")
            print("Switching to interactive mode...")
            interactive_single_test()
    else:
        # Interactive mode
        interactive_single_test()