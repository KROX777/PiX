"""
Original codes by Mingquan Feng: https://github.com/FengMingquan-sjtu/PhysPDE

Modified by Chuyang Xiang, Jul 2025.
Modification: 1) adaptive change for the new calculator and hypotheses tree architecture.
"""
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

class EarlyStopException(Exception):
    """Custom exception to signal early stopping"""
    pass

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

def single_test(cfg, root_dir, deci_list, deleted_coef=[], init_params=None, STR_iter_max=4, verbose=True):
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
    if lib['allow_nesting'] == True:
        raise NotImplementedError("Nested functions are not supported yet.")
    else:
        lib = [sp.sympify(f, locals=tmp_local_dict) for f in lib['functions']]
        
    for (y, x_vars) in SR_list:
        tree.calculator.update_unknown_var(y, '0')
        for x in x_vars:
            for f in lib:
                name = str(f)+"_STR_coef_"+str(x)+"_for_"+str(y)
                if name in deleted_coef:
                    continue
                coef = sp.symbols(name)
                tree.calculator.sp_unknown_quantities[name] = coef
                tree.calculator.sp_derived_quantities[y] += coef * f.subs(dummy_x, tree.calculator.local_dict[x])
    tree.calculator.upd_local_dict()
    
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
    
    is_STR_coefs = np.array(["_STR_coef" in p for p in tree.calculator.sp_unknown_quantities.keys()]) #boolean array
    if len(SR_list) == 0 or STR_iter_max <= 0: #ordinary solver
        sol = optimize_with_timeout(train_loss_func, init_params, tree.calculator.get_constr_dict_list(), prev_sol_best=None, verbose=True)
    else: # STRidge solver
        params_name = list(tree.calculator.sp_unknown_quantities.keys())
        #--- hyper-params---
        l2_reg_coef = 1e-2
        tol_w = 0.005 

        def pre_process(params):
            params = params.copy()
            params[is_STR_coefs] = 0.05*np.tanh(params[is_STR_coefs]) #bound coef in [-0.05, +0.05]
            #params[is_STR_coefs] /= norm[is_STR_coefs]
            return params
        
        STR_loss_func = lambda params: train_loss_func(pre_process(params)) + l2_reg_coef*(params[is_STR_coefs]**2).sum()

        if verbose:
            print(f"Start stridge_loop with {STR_iter_max=}", flush=1)
        
        sol = optimize_with_timeout(STR_loss_func, init_params, tree.calculator.get_constr_dict_list(), prev_sol_best={"fun":1e-3, "nit":5}, verbose=True)
        is_small_p = (pre_process(np.abs(sol['x'])) < tol_w) * is_STR_coefs

        if verbose:
            print(f"{tol_w=}, {is_small_p=}")
            pprint.pprint(sol)
            pprint.pprint(dict(zip(params_name, pre_process(sol['x']))))
            
        # ---threshold small p
        next_init_params = sol['x'].copy()[~is_small_p]
        for i in range(len(is_small_p)):
            if is_small_p[i]:
                deleted_coef.append(params_name[i])
                
        # --- recursive call
        if STR_iter_max>0  and is_small_p.sum()>0 :
            return single_test(cfg=cfg, 
                               root_dir=root_dir, 
                               deci_list=deci_list, 
                               deleted_coef=deleted_coef, 
                               init_params=next_init_params, 
                               STR_iter_max=STR_iter_max-1, 
                               verbose=verbose)
        else: # convert back to normal loss
            sol['x'] = pre_process(sol['x'])
            sol['fun'] = train_loss_func(sol['x'])
            if verbose:
                print("train_loss=",sol['fun'])
                print("STR_coefs_name=",np.array(params_name)[is_STR_coefs].tolist())
                print("STR_coefs(after pre_process)=",sol['x'][is_STR_coefs].tolist())    
    
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

def para_test(cfg, root_dir, deci_lists, n_jobs=10, time_limit=None):
    n_parts = 10
    n_dicts_per_part = len(deci_lists)//n_parts
    sols = []
    start_time = time_module.time()  
    for i in range(len(deci_lists)):
        sol = single_test(cfg, root_dir, deci_lists[i], deleted_coef=[], init_params=None, STR_iter_max=4)
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
def brute_force_search(cfg, root_dir, dataname_tuple, datafold_tuple=(0,1), out_file=None, n_jobs=10, verbose=False, time_limit=None):
    dummy_tree = LightTree(cfg, root_dir)
    deci_lists = dummy_tree.generate_all_possibilities()
    random.shuffle(deci_lists)
    print(f"{len(deci_lists)=},{n_jobs=}")

    sols = para_test(cfg, root_dir, deci_lists, n_jobs, time_limit)
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

def k_fold_cv_bfs(cfg, root_dir, dataname_tuple, k=1, out_file=None, n_jobs=10, verbose=False, time_limit=None):
    search_time_list = []
    top_3_nodes_list = []
    for i in range(k):
        print(f"=== {i+1}-th fold ===", flush=1)
        datafold_tuple = (i,k)
        top_3_sols, search_time = brute_force_search(cfg, root_dir, dataname_tuple, datafold_tuple, out_file, n_jobs, verbose, time_limit=time_limit)
        for sol in top_3_sols:
            pprint.pprint(sol)
        search_time_list.append(search_time)
    print("Search time mean={}, std={}".format(np.mean(search_time_list), np.std(search_time_list)))
