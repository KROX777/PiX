import pprint
import numpy as np
import time as time_module
import random
from pix.hypotheses_tree import LightTree
from pix.hypotheses_tree import HypothesesTree
from pix.utils.others import *
import sympy as sp
from pix.utils.scipy_utils import optimize_with_timeout
from pix.methods.SR4MDL.search import search as sr4mdl_search
import copy
import logging

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
    #initial guess
    # tree.calculator.register_unknown_var('b0')
    # tree.calculator.register_unknown_var('b1')
    # tree.calculator.register_unknown_var('b2')
    # tree.calculator.update_unknown_var('mu', 'b0 + b1*gamma^1.3 + b2*gamma')
    # tree.calculator.upd_local_dict()

    # print(init_params)
    # print(train_loss_func(init_params))

    # sol = optimize_with_timeout(train_loss_func, init_params, tree.calculator.get_constr_dict_list(), prev_sol_best=None, verbose=True)
    # print(sol['x'])
    # a = input('Press Enter to continue...')
    
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

                t_min, t_max = 81, 85
                Fx = 0
                
                # Bind the imported search function into a local name so the inner obj closure
                # doesn't treat it as a free variable from an outer scope. This avoids rare Python
                # scoping issues where a name might be considered unbound in closures.
                sr_search_fn = sr4mdl_search

                def obj(x):
                    mu = run_solver(tree.calculator.args_data, 
                                    t_min=t_min, 
                                    t_max=t_max, 
                                    dx=tree.calculator.dx, 
                                    dy=tree.calculator.dy, 
                                    dt=tree.calculator.dt,
                                    gt=gt,
                                    Fx=Fx,
                                    Fy=x)
                    sl = slice(t_min, t_max + 1)
                    mu_sub = mu[:, :, sl, 0]
                    
                    gamma_flat = gamma[:, :, sl].flatten()
                    mu_flat = mu_sub.flatten()
                    # drop non-finite pairs, then sort by gamma
                    mask = np.isfinite(gamma_flat) & np.isfinite(mu_flat)
                    gamma_flat = gamma_flat[mask]
                    mu_flat = mu_flat[mask]
                    order = np.argsort(gamma_flat)
                    gamma_sorted = gamma_flat[order]
                    mu_sorted = mu_flat[order]
                    calculator_copy = copy.deepcopy(tree.calculator)
                    expr, loss = sr_search_fn(calculator=calculator_copy, 
                                               y_name='mu', 
                                               x_name=['gamma'], 
                                               other_params_name=None,
                                               deci_list_len=len(deci_list),
                                               X_override={'gamma': gamma_sorted}, 
                                               y_override=mu_sorted, 
                                               mode="inverse", 
                                               n_iter=30, 
                                               is_final_optim=False,
                                               log_every_n_iters=25)
                    print(expr, loss)
                    return loss

                from scipy.optimize import dual_annealing
                bounds = [(7, 10)]
                class Callback:
                    def __init__(self, maxfun):
                        self.maxfun = maxfun
                        self.nfun = 0
                    def __call__(self, x, f, context):
                        # called periodically by dual_annealing; increment local counter
                        # Note: dual_annealing can perform multiple function evaluations
                        # between callback invocations, so relying on callback alone
                        # may not strictly limit the total 'obj' calls. Use the
                        # built-in `maxfun` parameter to enforce a hard limit.
                        self.nfun += 1
                        try:
                            logging.getLogger(__name__).info("dual_annealing callback called: %d/%d", self.nfun, self.maxfun)
                        except Exception:
                            pass
                        return self.nfun >= self.maxfun  # 达到次数后终止

                # Pass `maxfun` to enforce a hard cap on objective evaluations.
                # Keep the callback for logging/extra early-stop signalling.
                result = dual_annealing(
                    obj,
                    bounds=bounds,
                    maxfun=100,
                    callback=Callback(maxfun=100),
                    seed=42            # 随机种子（可选）
                )
                
                logger = logging.getLogger(__name__)
                logger.info("dual_annealing result.x: %s", result.x)
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
