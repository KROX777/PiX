import numpy as np
import time
import scipy
import sympy as sp
import time as time_module

class EarlyStopException(Exception):
    """Custom exception to signal early stopping"""
    pass

def optimize_with_timeout(mse_func, init_params, constr_dict_list, prev_sol_best=None, verbose=False,
                          time_limit: int = 20,
                          bound_f_coef: float = 500,
                          check_time_nit: int = 9):    
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

def optimize_with_lstsq(calculator, mse_func, init_params):
    """Linear least squares for coefficient fitting when equation is linear in coeffs."""
    start_time = time_module.time()
    try:
        eq = calculator.sp_equation[0] if len(calculator.sp_equation) > 0 else None
        if eq is None or (hasattr(eq, 'shape') and len(getattr(eq, 'shape', ())) > 0):
            raise ValueError("Equation is array/matrix or empty, cannot use lstsq")
        
        coeff_symbols = list(calculator.sp_unknown_quantities.values())
        q_exprs = []
        for c in coeff_symbols:
            q_expr = sp.expand(eq).coeff(c)
            if q_expr is None:
                q_expr = sp.Integer(0)
            q_exprs.append(q_expr)
        q0_expr = sp.expand(eq).subs({c: 0 for c in coeff_symbols})
        
        q0_func = sp.lambdify([calculator.args_symbols], q0_expr, 'numpy')
        q_funcs = [sp.lambdify([calculator.args_symbols], q_expr, 'numpy') for q_expr in q_exprs]
        
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            q0_val = np.asarray(q0_func(calculator.args_data), dtype=np.float64)
            q_vals = [np.asarray(q_fn(calculator.args_data), dtype=np.float64) for q_fn in q_funcs]
        
        A = np.column_stack([q.ravel() for q in q_vals]) if len(q_vals) > 0 else np.zeros((q0_val.size, 0))
        b = -q0_val.ravel()
        
        if A.shape[1] > 0:
            fit_coeffs_arr, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            fit_coeffs = np.asarray(fit_coeffs_arr, dtype=np.float64)
        else:
            fit_coeffs = np.array([], dtype=np.float64)
        
        fit_loss = float(mse_func(fit_coeffs))
        elapsed = time_module.time() - start_time
        return {
            "x": fit_coeffs,
            "fun": fit_loss,
            "nit": 1,
            "status": "Success",
            "time": elapsed,
            "method": "lstsq",
        }
    except Exception as e:
        elapsed = time_module.time() - start_time
        return {
            "x": init_params,
            "fun": float(mse_func(init_params)),
            "nit": 0,
            "status": "LSTSQ_Failed",
            "time": elapsed,
            "method": "lstsq",
            "error": str(e),
        }


def optimize_coefficients(method, mse_func, init_params, constr_dict_list, 
                          cfg=None, calculator=None, deci_list=None, **kwargs):
    """
    Unified entry point for coefficient optimization.
    
    Args:
        method: 'scipy', 'pinn', or 'lstsq'
        mse_func: Objective function for scipy
        init_params: Initial guess (numpy array)
        constr_dict_list: Constraints for SLSQP
        cfg: Full Hydra config
        calculator: PIX Calculator instance (required for PINN and LSTSQ)
        deci_list: List of active decision node IDs
    """
    if method == "scipy":
        return optimize_with_timeout(mse_func, init_params, constr_dict_list, **kwargs)
    
    elif method == "pinn":
        from pix.utils.eqgpt.PINN_optimization import optimize_with_pinn_impl
        return optimize_with_pinn_impl(calculator, cfg, deci_list, init_params, mse_func=mse_func)
    
    elif method == "lstsq":
        return optimize_with_lstsq(calculator, mse_func, init_params)

    else:
        raise ValueError(f"Unknown optimization method: {method}")
