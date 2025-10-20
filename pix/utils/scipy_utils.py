import scipy
import time as time_module

class EarlyStopException(Exception):
    """Custom exception to signal early stopping"""
    pass

def optimize_with_timeout(mse_func, init_params, constr_dict_list, prev_sol_best=None, verbose=False,
                          time_limit: int = 20,
                          bound_f_coef: float = 500,
                          check_time_nit: int = 9):    
    """
    Run SLSQP with a callback that can early stop by time or a heuristic bound.

    Args:
        mse_func: objective function
        init_params: initial parameter array
        constr_dict_list: SLSQP constraints
        prev_sol_best: dict like {"fun": float, "nit": int} to form a pruning bound; if None, bound check is disabled
        verbose: scipy minimize disp flag
        time_limit: seconds until forced stop
        bound_f_coef: multiplier for pruning bound
        check_time_nit: start checking bound at this iteration or later
    Returns:
        dict with keys: fun, x, nit, status, time, and optionally stop_reason
    """
    # Track state locally (avoid globals). We keep names for minimal intrusion.
    iteration_count = 0
    params = init_params
    start_time = 0.0
    stop_reason = None
    last_f = None  # last evaluated objective value

    # Guarded objective: checks time before each evaluation
    def guarded_mse(x):
        nonlocal last_f, params
        # hard timeout on per-evaluation basis
        if time_limit is not None and time_limit > 0:
            if (time_module.time() - start_time) > time_limit:
                raise EarlyStopException
        f = mse_func(x)
        last_f = f
        params = x
        return f

    def callback(xk,):
        nonlocal stop_reason, iteration_count, params, start_time, last_f
        iteration_count += 1
        params = xk
        time_elapsed = time_module.time() - start_time  

        # hard timeout check also here
        if time_limit is not None and time_limit > 0 and time_elapsed > time_limit:
            stop_reason = 'time_limit'
            raise EarlyStopException

        # bound pruning using last known f (avoid re-evaluating fun in callback)
        cur_f = last_f
        cur_nit = iteration_count
        if prev_sol_best and cur_nit >= check_time_nit and cur_f is not None:
            prev_f, prev_nit = prev_sol_best["fun"], prev_sol_best["nit"]
            bound_f = bound_f_coef * (prev_nit/cur_nit)**2 * prev_f
            if cur_f > bound_f:
                stop_reason = 'bound_prune'
                raise EarlyStopException
    try:
        start_time = time_module.time()  
        if len(init_params) > 0:
            # sol = scipy.optimize.minimize(guarded_mse, init_params, constraints=constr_dict_list, method="SLSQP",options={"disp":verbose}, callback=callback)
            sol = scipy.optimize.minimize(guarded_mse, init_params, method="L-BFGS-B",
                                          options={'maxiter': 50,'disp':verbose}, callback=callback)
        else:
            val = guarded_mse(init_params)
            sol = {"fun":val, "x":init_params, "nit":1}
        sol["status"] = "Success"
    except EarlyStopException:
        # Do not re-evaluate objective; use last known f if available
        fun_val = last_f if last_f is not None else (mse_func(params) if len(init_params)>0 else float('inf'))
        sol = {"fun":fun_val, "x":params, "nit":iteration_count, "status":"EarlyStop"}
        if stop_reason is not None:
            sol['stop_reason'] = stop_reason
    sol['time'] = time_module.time() - start_time  
    return sol