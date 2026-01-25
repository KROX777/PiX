import pprint
import numpy as np
import time as time_module
import random
import os
import sys
from pix.hypotheses_tree import LightTree
from pix.hypotheses_tree import HypothesesTree
from pix.utils.others import *
import sympy as sp
from pix.utils.scipy_utils import optimize_with_timeout
try:
    from pix.methods.SR4MDL.search import search as sr4mdl_search
except Exception:
    print("Warning: SR4MDL module not available, sr4mdl_search will be None")
    sr4mdl_search = None
import copy
import logging
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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


def run_snip_symbolic_regression(x_arr, y_arr, cfg, ckpt_path=None, device=None, top_k=10, batch_data=None):
    """Run SNIP E2E model with LSO genetic algorithm optimization.

    x_arr, y_arr: 1D numpy arrays (used for evaluation/refinement)
    batch_data: Optional list of n groups, each with (x, y) tuples for generation. 
                These n groups will be used as initial population for LSO genetic algorithm.
    cfg: config object (may contain 'snip_ckpt' path or 'use_snip' flag)
    """
    # Ensure local SNIP package (pix/methods/SNIP/symbolicregression) is importable
    snip_pkg_dir = os.path.join(os.path.dirname(__file__), 'SNIP')
    if os.path.isdir(snip_pkg_dir) and snip_pkg_dir not in sys.path:
        sys.path.insert(0, snip_pkg_dir)

    try:
        from pix.methods.SNIP.parsers import get_parser as snip_get_parser
        from pix.methods.SNIP.symbolicregression.envs import build_env as snip_build_env
        from pix.methods.SNIP.symbolicregression.model import build_modules as snip_build_modules
        from pix.methods.SNIP.model import SNIPSymbolicRegressor
        from pix.methods.SNIP.LSO_fit import lso_fit, LSOFitNeverGrad
        from pix.methods.SNIP.symbolicregression.model.model_wrapper import ModelWrapper
        import pix.methods.SNIP.symbolicregression.model.utils_wrapper as utils_wrapper
        from collections import defaultdict
    except Exception as e:
        raise ImportError(f"SNIP modules not available: cannot run SNIP symbolic regression (tried local SNIP dir: {snip_pkg_dir}). Original error: {e}")

    # Determine device: default to CPU if CUDA not available
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build params similar to LSO_eval
    parser = snip_get_parser()
    params = parser.parse_args([])
    params.use_diffusion = False
    params.cpu = (device == 'cpu')
    params.batch_size = 1
    params.eval_only = True
    params.reload_model_snipenc = ""
    params.reload_model_e2edec = ""
    params.max_input_dimension = 10
    params.device = torch.device(device)
    
    # LSO optimizer params from cfg or defaults
    params.lso_optimizer = cfg.get('lso_optimizer', 'gwo')
    params.lso_pop_size = cfg.get('lso_pop_size', 50)
    base_iteration = cfg.get('lso_max_iteration', 80)
    params.lso_max_iteration = max(1, base_iteration // 2)
    params.lso_stop_r2 = cfg.get('lso_stop_r2', 0.99)
    params.beam_size = cfg.get('beam_size', 10)
    params.n_trees_to_refine = params.beam_size
    params.max_input_points = cfg.get('max_input_points', 200)
    params.rescale = cfg.get('rescale', True)
    
    ckpt_path = cfg.get('snip_ckpt', None)
    if ckpt_path is None:
        raise RuntimeError('SNIP checkpoint path not provided in cfg["snip_ckpt"]')
    
    # Build env and modules
    env = snip_build_env(params)
    modules = snip_build_modules(env, params)

    # Reload checkpoint similar to LSO_eval.reload_model
    if os.path.isfile(ckpt_path):
        data = torch.load(ckpt_path, map_location=params.device, weights_only=False)
        for k, v in modules.items():
            if k in data:
                try:
                    weights = data[k]
                    v.load_state_dict(weights)
                    print(f"Loaded weights for module: {k}")
                except RuntimeError as original_error:
                    # try stripping 'module.' prefix
                    try:
                        stripped = {name.partition('.')[-1]: w for name, w in data[k].items()}
                        v.load_state_dict(stripped)
                        print(f"Loaded (stripped) weights for module: {k}")
                    except RuntimeError:
                         # If stripping also fails, raise the original error
                         print(f"Failed to load weights for {k}: {original_error}")
                         raise original_error
                v.requires_grad = False
    else:
        raise RuntimeError(f'SNIP checkpoint not found: {ckpt_path}')

    # Move modules to device and eval
    for m in modules.values():
        m.to(params.device)
        m.eval()

    # Create SNIPSymbolicRegressor model wrapper
    model = SNIPSymbolicRegressor(params=params, env=env, modules=modules)
    model.to(params.device)

    # Prepare data: n groups of (x, y) samples for initial population generation
    if batch_data is not None:
        # batch_data is list of (x_i, y_i) tuples, each representing a group
        X_list = []
        Y_list = []
        for bx, by in batch_data:
            X_list.append(np.array(bx).reshape(-1, 1))
            Y_list.append(np.array(by).reshape(-1, 1))
    else:
        # Single group fallback
        X_list = [np.array(x_arr).reshape(-1, 1)]
        Y_list = [np.array(y_arr).reshape(-1, 1)]

    # Scale X similar to LSO_eval
    scaler = utils_wrapper.StandardScaler() if params.rescale else None
    scale_params = {}
    if scaler is not None:
        scaled_X = []
        for i, x in enumerate(X_list):
            scaled_X.append(scaler.fit_transform(x))
            scale_params[i] = scaler.get_params()
    else:
        scaled_X = X_list

    # Prepare sample_to_learn structure for LSO
    # For simplicity, use first group for fitting (or average/concatenate if needed)
    X_scaled_to_fit = scaled_X[0]
    Y_scaled_to_fit = Y_list[0]
    x_to_fit = X_list[0]
    y_to_fit = Y_list[0]
    
    # Split into train/test if needed (here we use all for fitting)
    sample_to_learn = {
        'X_scaled_to_fit': [X_scaled_to_fit],
        'Y_scaled_to_fit': [Y_scaled_to_fit],
        'x_to_fit': [x_to_fit],
        'y_to_fit': [y_to_fit],
        'x_to_predict': [x_to_fit],  # use same for prediction in this context
        'y_to_predict': [y_to_fit]
    }

    batch_results = defaultdict(list)
    bag_number = 1

    # Run LSO optimization (genetic algorithm)
    with torch.no_grad():
        if params.lso_optimizer == "gwo":
            batch_results = lso_fit(sample_to_learn, env, params, model, batch_results, bag_number)
        else:
            opt_LSO = LSOFitNeverGrad(env, params, model, sample_to_learn, batch_results, bag_number)
            batch_results = opt_LSO.fit_func()

    # Extract candidate results
    def build_candidate_from_tree(predicted_tree, r2_final, mse_val):
        skeleton_expr = None
        skeleton_placeholders = []
        skeleton_values = []
        skeleton_tree = None
        best_expr = "Unknown"
        if predicted_tree is not None:
            try:
                expr_str = predicted_tree.infix() if hasattr(predicted_tree, 'infix') else str(predicted_tree)
                replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
                for old, new in replace_ops.items():
                    expr_str = expr_str.replace(old, new)
                best_expr = expr_str
            except Exception:
                best_expr = str(predicted_tree)
            try:
                skeleton_tree, constants = env.generator.function_to_skeleton(predicted_tree, constants_with_idx=True)
                for c in constants:
                    try:
                        skeleton_values.append(float(c))
                    except Exception:
                        try:
                            skeleton_values.append(float(sp.N(str(c))))
                        except Exception:
                            skeleton_values.append(None)
                for j in range(len(constants)):
                    name = f"CONSTANT_{j}"
                    if name not in env.simplifier.local_dict:
                        env.simplifier.local_dict[name] = sp.Symbol(name, real=True)
                    skeleton_placeholders.append(env.simplifier.local_dict[name])
                skeleton_expr = env.simplifier.tree_to_sympy_expr(skeleton_tree)
            except Exception as err:
                print(f"Warning: failed to extract SNIP skeleton for candidate: {err}")
        return {
            'expr': best_expr,
            'mse': mse_val,
            'r2': r2_final,
            'sympy_expr': skeleton_expr,
            'const_placeholders': skeleton_placeholders,
            'const_values': skeleton_values,
            'predicted_tree': predicted_tree,
            'skeleton_tree': skeleton_tree
        }

    candidates = []
    candidate_records = batch_results.get("_candidate_records")
    if candidate_records:
        y_var = np.var(y_to_fit)
        for rec in candidate_records[:top_k]:
            tree = rec.get('tree')
            r2_val = rec.get('r2', 0.0)
            mse_val = rec.get('mse')
            if mse_val is None:
                mse_val = (1.0 - r2_val) * y_var if r2_val < 1.0 else 1e-10
            candidates.append(build_candidate_from_tree(tree, r2_val, mse_val))
    else:
        r2_key = 'r2_zero_direct_fit'
        if r2_key not in batch_results or len(batch_results[r2_key]) == 0:
            r2_key = 'r2_zero_final_fit'
        if r2_key in batch_results and len(batch_results[r2_key]) > 0:
            r2_list = batch_results[r2_key]
            mse_list = batch_results.get('direct_fit_mse', [None] * len(r2_list))
            predicted_list = batch_results.get('direct_predicted_tree', [])
            y_var = np.var(y_to_fit)
            order = sorted(range(len(r2_list)), key=lambda i: r2_list[i], reverse=True)
            top_indices = order[:top_k]
            for idx in top_indices:
                r2_final = r2_list[idx]
                predicted_tree = predicted_list[idx] if idx < len(predicted_list) else None
                mse_val = None
                if idx < len(mse_list) and mse_list[idx] is not None:
                    mse_val = mse_list[idx]
                else:
                    mse_val = (1.0 - r2_final) * y_var if r2_final < 1.0 else 1e-10
                candidates.append(build_candidate_from_tree(predicted_tree, r2_final, mse_val))

    if candidates:
        print(f"LSO optimization complete: collected top-{len(candidates)} candidates (best R2={candidates[0]['r2']:.6f})")
        return candidates[:top_k]
    else:
        raise RuntimeError('LSO optimization failed to produce valid results')

def single_test(cfg, root_dir, deci_list, deleted_coef=[], init_params=None, verbose=True, preset=None, allowed_functions=None, method="directxy"):
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
    
    if verbose:
        print("deci_list", deci_list)
        print("unknown_quantities", tree.calculator.sp_unknown_quantities.keys())
        print("equation", tree.calculator.equation_buffer)
        print("SR_list ", SR_list)
    
    # --- optimization---
    ## DEBUG
    # tree.calculator.register_unknown_var('b0')
    # tree.calculator.register_unknown_var('b1')
    # tree.calculator.register_unknown_var('b2')
    # tree.calculator.update_unknown_var('mu', 'b0 + b1/(b2+gamma)')
    # tree.calculator.upd_local_dict()
    # train_loss_func, mse_list_train = tree.calculator.get_loss_func(deci_list_len=len(deci_list))
    # valid_loss_func, mse_list_valid = tree.calculator.get_loss_func(mode="valid", deci_list_len=len(deci_list))
    # init_params = [0, 0, 0]

    # sol = optimize_with_timeout(train_loss_func, init_params, tree.calculator.get_constr_dict_list(), prev_sol_best=None, verbose=True)
    # print(sol['x'])
    # print(sol['fun'])
    # a = input('Press Enter to continue...')
    ## DEBUG
    
    if len(SR_list) == 0: #ordinary solver
        # Disable aggressive bound-based early stop by default; rely on time limit/practical convergence.
        train_loss_func, mse_list_train = tree.calculator.get_loss_func(deci_list_len=len(deci_list))
        valid_loss_func, mse_list_valid = tree.calculator.get_loss_func(mode="valid", deci_list_len=len(deci_list))
        if init_params is None:
            init_params = np.random.rand(len(tree.calculator.sp_unknown_quantities))
            init_params = np.ones_like(init_params)
            params_constr = tree.calculator.constraints
            for i, constr_list in enumerate(params_constr.values()):
                for c in constr_list:
                    if "init" in c:
                        init_params[i] += c["init"]
        sol = optimize_with_timeout(train_loss_func, init_params, tree.calculator.get_constr_dict_list(), prev_sol_best=None, verbose=True)
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
        if 'stop_reason' in sol:
            ret_sol['stop_reason'] = sol['stop_reason']

        if ret_sol['status'] == "Success": # record detailed infos of train loss.
            ret_sol['train_mse_list'] = mse_list_train(sol['x'])
            ret_sol['valid_mse_list'] = mse_list_valid(sol['x'])

        return ret_sol
    else:
        params_name = list(tree.calculator.sp_unknown_quantities.keys())
        tree.calculator.upd_local_dict()
        (y, x_vars) = SR_list[0]
        if method == "directxy":
            from pix.methods.pde_solver import run_solver
            # from pix.methods.hippylib_ns_inverse import run_solver_hippylib
            
            if y == "mu":
                u_x = tree.calculator.args_data[1]
                v_x = tree.calculator.args_data[9]
                u_y = tree.calculator.args_data[2]
                v_y = tree.calculator.args_data[10]

                S_xx = u_x
                S_xy = 0.5 * (u_y + v_x)
                S_yy = v_y
                S_ddot_S = S_xx**2 + 2 * S_xy**2 + S_yy**2
                gamma = np.sqrt(2 * S_ddot_S)
                gamma_sym = sp.symbols('gamma')
                gt_expr_str = cfg.problem['gt_expression']
                gt_expr = sp.sympify(gt_expr_str, locals={'gamma': gamma_sym, 'cos': sp.cos, 'sin': sp.sin, 'tan': sp.tan, 'log': sp.log, 'exp': sp.exp, 'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E})
                gt_func = sp.lambdify(gamma_sym, gt_expr, 'numpy')
                gt = gt_func(gamma)

                t_min, t_max = 81, 90
                Fx = 0
        
                # Bind the imported search function into a local name so the inner obj closure
                # doesn't treat it as a free variable from an outer scope. This avoids rare Python
                # scoping issues where a name might be considered unbound in closures.
                sr_search_fn = sr4mdl_search
                
                use_hippylib = cfg.get('use_hippylib', False)
                
                # if use_hippylib:
                #     print(f"[BFSearch] Using HIPPYLIB solver for μ estimation")
                #     gamma_reg = cfg.problem.get('gamma_regularization', 1e-6)   # wtf?
                #     maxiter_adj = cfg.get('adjoint_maxiter', 20)

                # def obj(x):
                #     # x is Fy (external force in y-direction)
                #     if use_hippylib:
                #         raise NotImplementedError("hippylib disabled")
                #         # # Use hippylib FEniCS-based solver
                #         # mu = run_solver_hippylib(
                #         #     tree.calculator.args_data,
                #         #     t_min=t_min,
                #         #     t_max=t_max,
                #         #     dx=tree.calculator.dx,
                #         #     dy=tree.calculator.dy,
                #         #     dt=tree.calculator.dt,
                #         #     gamma=gamma_reg,
                #         #     maxiter=maxiter_adj,
                #         #     gtol=1e-5,
                #         #     gt=gt,
                #         #     verbose=cfg.verbose,
                #         #     Fx=Fx,
                #         #     Fy=x
                #         # )
                #     else:
                #         mu, g_est = run_solver(
                #             tree.calculator.args_data, 
                #             t_min=t_min, 
                #             t_max=t_max, 
                #             dx=tree.calculator.dx, 
                #             dy=tree.calculator.dy, 
                #             dt=tree.calculator.dt,
                #             gt=gt,
                #             Fx=Fx,
                #             Fy=0,       # Fy=0 when estimate_g_global=True
                #             estimate_g_global=True
                #         )
                    
                #     sl = slice(t_min, t_max + 1)
                #     mu_sub = mu[:, :, sl, 0]
                #     print(mu_sub)
                #     print(g_est)
                    
                #     gamma_flat = gamma[:, :, sl].flatten()
                #     mu_flat = mu_sub.flatten()
                #     # drop non-finite pairs, then sort by gamma
                #     mask = np.isfinite(gamma_flat) & np.isfinite(mu_flat)
                #     gamma_flat = gamma_flat[mask]
                #     mu_flat = mu_flat[mask]
                #     order = np.argsort(gamma_flat)
                #     gamma_sorted = gamma_flat[order]
                #     mu_sorted = mu_flat[order]
                #     calculator_copy = copy.deepcopy(tree.calculator)
                #     use_snip = cfg.get('use_snip', False)

                #     if use_snip:
                #         best_expr, best_loss = run_snip_symbolic_regression(gamma_sorted, mu_sorted, cfg)
                #         print("SNIP Best:", best_expr, best_loss)
                #         return best_loss

                #     else:
                #         expr, loss = sr_search_fn(calculator=calculator_copy, 
                #                                 y_name='mu', 
                #                                 x_name=['gamma'], 
                #                                 other_params_name=None,
                #                                 deci_list_len=len(deci_list),
                #                                 X_override={'gamma': gamma_sorted}, 
                #                                 y_override=mu_sorted, 
                #                                 mode="inverse", 
                #                                 n_iter=100, 
                #                                 is_final_optim=True,
                #                                 log_every_n_iters=25)
                #         print("SR4MDL Best: ", expr, loss)
                #         return loss

                # obj(0)  # Testing g=0
                
                if use_hippylib:
                    raise NotImplementedError("hippylib disabled")
                else:
                    mu, g_est = run_solver(
                        tree.calculator.args_data, 
                        t_min=t_min, 
                        t_max=t_max, 
                        dx=tree.calculator.dx, 
                        dy=tree.calculator.dy, 
                        dt=tree.calculator.dt,
                        gt=gt,
                        Fx=Fx,
                        Fy=0,       # Fy=0 when estimate_g_global=True
                        estimate_g_global=True
                    )
                
                sl = slice(t_min, t_max + 1)
                mu_sub = mu[:, :, sl, 0]
                print(mu_sub)
                print(g_est)
                
                gamma_flat = gamma[:, :, sl].flatten()
                mu_flat = mu_sub.flatten()
                
                # Drop non-finite pairs, then sort by gamma
                mask = np.isfinite(gamma_flat) & np.isfinite(mu_flat)
                gamma_flat = gamma_flat[mask]
                mu_flat = mu_flat[mask]
                order = np.argsort(gamma_flat)
                gamma_sorted = gamma_flat[order]
                mu_sorted = mu_flat[order]
                
                batch_data = None
                if len(mu_sorted) > 200:
                    k = 200
                    kmeans = KMeans(n_clusters=k, max_iter=200, random_state=42)
                    kmeans.fit(mu_sorted.reshape(-1, 1))
                    
                    # Compute cluster size statistics
                    labels = kmeans.labels_
                    unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
                    print(f"KMeans clustering: {len(mu_sorted)} points -> {k} clusters")
                    print(f"  Cluster sizes - max: {cluster_sizes.max()}, min: {cluster_sizes.min()}, "
                          f"avg: {cluster_sizes.mean():.2f}, std: {cluster_sizes.std():.2f}")
                    
                    # c = input("Press Enter to continue...")
                    snip_mode = cfg.get('snip_mode', 'random')
                    if snip_mode == 'random':
                        print(f"  Using random sampling mode: generating 50 batches")
                        batch_data = []
                        for _ in range(50):
                            indices = []
                            for i in range(k):
                                cluster_indices = np.where(labels == i)[0]
                                if len(cluster_indices) > 0:
                                    indices.append(np.random.choice(cluster_indices))
                            indices = np.array(indices)
                            g_sub = gamma_sorted[indices]
                            m_sub = mu_sorted[indices]
                            order_sub = np.argsort(g_sub)
                            batch_data.append((g_sub[order_sub], m_sub[order_sub]))

                    centroids = kmeans.cluster_centers_
                    closest, _ = pairwise_distances_argmin_min(centroids, mu_sorted.reshape(-1, 1))
                    gamma_sorted = gamma_sorted[closest]
                    mu_sorted = mu_sorted[closest]
                    order = np.argsort(gamma_sorted)
                    gamma_sorted = gamma_sorted[order]
                    mu_sorted = mu_sorted[order]
                
                calculator_copy = copy.deepcopy(tree.calculator)
                use_snip = cfg.get('use_snip', False)

                if use_snip:
                    snip_candidates = run_snip_symbolic_regression(gamma_sorted, mu_sorted, cfg, batch_data=batch_data)
                    calculator_template = copy.deepcopy(tree.calculator)
                    best_snip_solution = None
                    for cand_idx, snip_result in enumerate(snip_candidates):
                        tree.calculator = copy.deepcopy(calculator_template)
                        cand_r2 = snip_result.get('r2', float('nan'))
                        cand_mse = snip_result.get('mse', float('nan'))
                        print(f"SNIP Candidate {cand_idx+1}: {snip_result.get('expr')} (R2={cand_r2:.6f}, MSE~{cand_mse:.6e})")
                        snip_expr = snip_result.get('sympy_expr')
                        const_placeholders = snip_result.get('const_placeholders') or []
                        const_values = snip_result.get('const_values') or []
                        if snip_expr is None:
                            print("  -> Skipping candidate: missing symbolic skeleton.")
                            continue
                        
                        placeholder_map = {}
                        snip_param_names = []
                        for idx, placeholder_sym in enumerate(const_placeholders):
                            param_name = f"snip_c{cand_idx}_{idx}"
                            snip_param_names.append(param_name)
                            tree.calculator.register_unknown_var(param_name)
                            placeholder_map[placeholder_sym] = tree.calculator.sp_unknown_quantities[param_name]
                        
                        symbol_subs = {}
                        for sym in list(snip_expr.free_symbols):
                            name = getattr(sym, 'name', '')
                            if name.startswith("x_"):
                                try:
                                    feature_idx = int(name.split("_")[1])
                                except ValueError:
                                    raise RuntimeError(f"Unexpected SNIP feature symbol '{name}'")
                                if feature_idx >= len(x_vars):
                                    raise RuntimeError(f"SNIP expression references {name} but only {len(x_vars)} SR variables are provided.")
                                var_name = x_vars[feature_idx]
                                symbol_subs[sym] = sp.Symbol(var_name)
                        
                        snip_expr_calc = snip_expr
                        if symbol_subs:
                            snip_expr_calc = snip_expr_calc.subs(symbol_subs)
                        remaining = [
                            sym for sym in snip_expr_calc.free_symbols
                            if isinstance(sym, sp.Symbol) and sym.name.startswith("x_")
                        ]
                        if remaining:
                            print(f"  -> Skipping candidate: unresolved inputs {remaining}")
                            continue
                        
                        if placeholder_map:
                            snip_expr_calc = snip_expr_calc.subs(placeholder_map)
                        
                        tree.calculator.update_unknown_var(y, sp.sstr(snip_expr_calc))
                        tree.calculator.upd_local_dict()
                        
                        train_loss_func, mse_list_train = tree.calculator.get_loss_func(deci_list_len=len(deci_list))
                        valid_loss_func, mse_list_valid = tree.calculator.get_loss_func(mode="valid", deci_list_len=len(deci_list))
                        init_params = np.random.rand(len(tree.calculator.sp_unknown_quantities))
                        param_value_map = {}
                        for idx, name in enumerate(snip_param_names):
                            if idx < len(const_values) and const_values[idx] is not None:
                                param_value_map[name] = const_values[idx]
                        for idx, name in enumerate(tree.calculator.sp_unknown_quantities.keys()):
                            if name in param_value_map:
                                init_params[idx] = param_value_map[name]
                        
                        sol = optimize_with_timeout(train_loss_func, init_params, tree.calculator.get_constr_dict_list(), prev_sol_best=None, verbose=True)
                        ret_sol = dict()
                        ret_sol['train_loss'] = sol['fun']
                        ret_sol['valid_loss'] = valid_loss_func(sol['x'])
                        ret_sol['deci_list'] = deci_list
                        params_name = list(map(str, tree.calculator.sp_unknown_quantities.keys()))
                        ret_sol['params'] = dict(zip(params_name, sol['x']))
                        ret_sol['time'] = sol['time']
                        ret_sol['nit'] = sol['nit']
                        ret_sol['status'] = sol['status']
                        if 'stop_reason' in sol:
                            ret_sol['stop_reason'] = sol['stop_reason']
                        if ret_sol['status'] == "Success":
                            ret_sol['train_mse_list'] = mse_list_train(sol['x'])
                            ret_sol['valid_mse_list'] = mse_list_valid(sol['x'])
                        if best_snip_solution is None or ret_sol['train_loss'] < best_snip_solution['train_loss']:
                            best_snip_solution = ret_sol
                        if ret_sol['train_loss'] < 1e-2:
                            print(f"  -> Early exit: PDE optimization reached loss {ret_sol['train_loss']:.6e}")
                            return ret_sol
                    
                    if best_snip_solution is not None:
                        return best_snip_solution
                    else:
                        raise RuntimeError("SNIP candidates did not yield a valid PDE solution.")
                else:
                    expr, loss = sr_search_fn(calculator=calculator_copy, 
                                            y_name='mu', 
                                            x_name=['gamma'], 
                                            other_params_name=None,
                                            deci_list_len=len(deci_list),
                                            X_override={'gamma': gamma_sorted}, 
                                            y_override=mu_sorted, 
                                            mode="inverse", 
                                            n_iter=100, 
                                            is_final_optim=True,
                                            log_every_n_iters=25)
                    print("SR4MDL Best: ", expr, loss)
                
                c = input("Press Enter...")

                # from scipy.optimize import dual_annealing
                # bounds = [(7, 10)]
                # class Callback:
                #     def __init__(self, maxfun):
                #         self.maxfun = maxfun
                #         self.nfun = 0
                #     def __call__(self, x, f, context):
                #         self.nfun += 1
                #         try:
                #             logging.getLogger(__name__).info("dual_annealing callback called: %d/%d", self.nfun, self.maxfun)
                #         except Exception:
                #             pass
                #         return self.nfun >= self.maxfun  # 达到次数后终止
                # result = dual_annealing(
                #     obj,
                #     bounds=bounds,
                #     maxfun=100,
                #     callback=Callback(maxfun=100),
                #     seed=42
                # )
                
                # logger = logging.getLogger(__name__)
                # logger.info("dual_annealing result.x: %s", result.x)
                c = input("Press Enter to continue...")
                            
            else:
                raise NotImplementedError(f"SR target '{y}' not supported yet in directxy method.")
        else:
            # Use globally imported sr4mdl_search to avoid creating a local binding
            sr4mdl_search(tree.calculator, y, x_vars, params_name, len(deci_list))
    return
    

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
