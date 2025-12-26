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


def run_snip_symbolic_regression(x_arr, y_arr, cfg, ckpt_path=None, device='cuda', max_input_dimension=2, top_k=10, batch_data=None):
    """Run SNIP E2E model on (x_arr, y_arr) and return best candidate expr and its MSE.

    x_arr, y_arr: 1D numpy arrays (used for evaluation/refinement)
    batch_data: Optional list of (x, y) tuples for generation. If None, uses (x_arr, y_arr).
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
        from pix.methods.SNIP.const_opt import refine as snip_refine
    except Exception as e:
        # Provide a clearer message including the attempted local SNIP path
        raise ImportError(f"SNIP modules not available: cannot run SNIP symbolic regression (tried local SNIP dir: {snip_pkg_dir}). Original error: {e}")

    # build minimal params similar to quick_test_e2e
    params = snip_get_parser().parse_args([])
    params.use_diffusion = False
    
    params.batch_size = 1
    params.eval_only = True
    params.reload_model_snipenc = ""
    params.reload_model_e2edec = ""
    params.max_input_dimension = max_input_dimension
    
    ckpt_path = cfg.get('snip_ckpt', None)
    if ckpt_path is None:
        raise RuntimeError('SNIP checkpoint path not provided in cfg["snip_ckpt"]')
    env = snip_build_env(params)
    modules = snip_build_modules(env, params)

    
    try:
        ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        for k, v in modules.items():
            if k in ckpt:
                try:
                    v.load_state_dict(ckpt[k])
                except RuntimeError:
                    stripped = {name.partition('.')[-1]: w for name, w in ckpt[k].items()}
                    v.load_state_dict(stripped)
    except Exception as e:
        print("Warning: failed to load SNIP checkpoint:", e)

    # Move modules to device and eval
    for m in modules.values():
        m.to(torch.device(device))
        m.eval()

    embedder = modules.get('embedder')
    encoder = modules.get('encoder_y')
    decoder = modules.get('decoder')
    mapper = modules.get('mapper')
    if embedder is None or encoder is None or decoder is None or mapper is None:
        raise RuntimeError('Missing SNIP modules (embedder/encoder/decoder/mapper)')

    # Prepare sequence in SNIP expected format: list of (x_arr_point, y_arr_point)
    if batch_data is not None:
        batch_input = []
        for bx, by in batch_data:
            seq = []
            for xi, yi in zip(bx, by):
                seq.append((np.array([xi], dtype=float), np.array([yi], dtype=float)))
            batch_input.append(seq)
    else:
        seq = []
        for xi, yi in zip(x_arr, y_arr):
            seq.append((np.array([xi], dtype=float), np.array([yi], dtype=float)))
        batch_input = [seq]

    with torch.no_grad():
        x_enc, x_len = embedder(batch_input)
        z_rep = encoder('fwd', x=x_enc, lengths=x_len, causal=False)
        src_enc = mapper(z_rep)
        generations, gen_len = decoder.generate_from_latent(src_enc, max_len=200)

    # Convert tokens to infix strings similar to quick_test_e2e
    generations = generations.unsqueeze(-1).view(generations.shape[0], generations.shape[1], 1)
    generations = generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
    outputs = [
        list(
    # compute MSE per candidate (first batch only) using SNIP's refine/const-fitting and simplifier
    # candidates = outputs[0] if len(outputs) > 0 else []
    
    # Flatten candidates from all batches
    all_candidates = []
    for batch_cands in outputs:
        all_candidates.extend(batch_cands)
    candidates = list(set(all_candidates))

    if not candidates:
        raise RuntimeError('SNIP produced no valid candidates')
        )
        for i in range(len(generations))
    ]

    # compute MSE per candidate (first batch only) using SNIP's refine/const-fitting and simplifier
    candidates = outputs[0] if len(outputs) > 0 else []
    if not candidates:
        raise RuntimeError('SNIP produced no valid candidates')

    X = np.array(x_arr).reshape(-1, 1)
    y = np.array(y_arr).reshape(-1,)

    # Try to run SNIP's refine (constant fitting + BFGS). Prefer using the env we built.
    refined = None
    try:
        refined = snip_refine(env, X, y, candidates, verbose=False)
    except Exception:
        # try with a fresh env (may be heavier)
        try:
            fresh_params = snip_get_parser().parse_args([])
            fresh_env = snip_build_env(fresh_params)
            refined = snip_refine(fresh_env, X, y, candidates, verbose=False)
        except Exception as e:
            # fallback: attempt simple numeric evaluation via env.simplifier where possible
            results = []
            for cand in candidates:
                try:
                    skel, _ = env.generator.function_to_skeleton(cand, constants_with_idx=False)
                    numfn = env.simplifier.tree_to_numexpr_fn(skel)
                    pred = numfn(X)[:, 0]
                    mask = np.isfinite(pred) & np.isfinite(y)
                    if mask.sum() == 0:
                        continue
                    mse = float(np.mean((pred[mask] - y[mask]) ** 2))
                    results.append((cand, mse))
                except Exception:
                    continue
            if not results:
                raise RuntimeError('SNIP produced no valid numeric candidates')
            results = sorted(results, key=lambda t: t[1])
            return results[0][0], results[0][1]

    # refined is a list of dicts with 'predicted_tree' entries
    results = []
    for cand in refined:
        try:
            tree = cand.get('predicted_tree', None)
            if tree is None:
                continue
            numfn = env.simplifier.tree_to_numexpr_fn(tree)
            pred = numfn(X)[:, 0]
            mask = np.isfinite(pred) & np.isfinite(y)
            if mask.sum() == 0:
                continue
            mse = float(np.mean((pred[mask] - y[mask]) ** 2))
            try:
                infix = tree.infix()
            except Exception:
                infix = str(tree)
            results.append((infix, mse))
        except Exception:
            continue

    if not results:
        raise RuntimeError('SNIP produced no valid numeric candidates after refinement')

    results = sorted(results, key=lambda t: t[1])
    return results[0][0], results[0][1]

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
                #     # Optionally run SNIP-based symbolic regression (from SNIP/quick_test_e2e)
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
                # drop non-finite pairs, then sort by gamma
                batch_data = None
                if len(mu_sorted) > 200:
                    k = 200
                    kmeans = KMeans(n_clusters=k, max_iter=200, random_state=42)
                    kmeans.fit(mu_sorted.reshape(-1, 1))
                    
                    snip_mode = cfg.get('snip_mode', 'centroid')
                    if snip_mode == 'random':
                        batch_data = []
                        labels = kmeans.labels_
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
                # Optionally run SNIP-based symbolic regression (from SNIP/quick_test_e2e)
                use_snip = cfg.get('use_snip', False)

                if use_snip:
                    best_expr, best_loss = run_snip_symbolic_regression(gamma_sorted, mu_sorted, cfg, max_input_dimension=1, batch_data=batch_data)
                    print("SNIP Best:", best_expr, best_loss)
                    return best_loss

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
