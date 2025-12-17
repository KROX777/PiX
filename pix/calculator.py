"""
Calculator class for symbolic calculations.
All symbols are claimed and stored here.
Equations are also stored here.
Symbolic regression isn't included in this class, offering more freedom.
"""

from pix.data_loader import DataLoader
import numpy as np
import sympy as sp
import copy
from pix.utils.sympy_utils import *
from pix.utils.numpy_utils import np_grad_all, pooling
from sklearn.model_selection import KFold
import os
import time

class Calculator:
    def __init__(self, config, root_dir, datafold_tuple=(0,1), tol=1e-3, K=1):
        self.config = config
        self.data_loader = DataLoader(config)
        self.root_dir = root_dir
        # 缓存数据和网格
        self.data_loader.get_raw_data(os.path.join(root_dir, config.dataset_path), verbose=config.verbose)
        self.spatial_vars = self.data_loader.spatial_vars
        self.field_vars = self.data_loader.field_vars
        self.temporal_vars = self.data_loader.temporal_vars

        # ---- 自动符号声明----
        # 空间变量
        self.sp_spatial_vars = {var: sp.symbols(var) for var in self.spatial_vars}
        self.space_axis = [self.sp_spatial_vars[var] for var in self.spatial_vars]

        self.X = sp.Array(self.space_axis)
        self.X_dim = len(self.space_axis)
        
        # 时间变量
        self.sp_temporal_vars = {var: sp.symbols(var) for var in self.temporal_vars}
        self.has_time = 't' in self.temporal_vars
        if self.has_time:
            self.t = self.sp_temporal_vars['t']

        # 场变量（自动生成符号函数，并自动展开）
        self.sp_field_funcs = {}
        for var in self.field_vars:
            f = sp.Function(var)
            if self.has_time:
                self.sp_field_funcs[var] = f(*self.space_axis, self.t)
            else:
                self.sp_field_funcs[var] = f(*self.space_axis)
        
        # 创建速度向量（在解析derived_quantities之前）
        self._create_velocity_vector()
        
        # 导出物理量（延迟解析，在所有基础符号创建完成后）
        # 知道unknown quantity之后，会转到derived quantity，derived quantity表达式里可能有unknown quantity
        self.sp_derived_quantities = {}
        self.sp_unknown_quantities = {}
        self.sp_constants = {}
        # 常量
        for k, v in config.problem['constants'].items():
            try:
                self.sp_constants[k] = sp.sympify(v)
            except Exception as e:
                print(f"Warning: Could not parse constant '{k}': {v}, error: {e}")
        
        # 其它常量
        self.sp_constants['I'] = sp.Array(sp.eye(self.X_dim))
        
        # 函数
        self.grad = lambda a: ts_grad(a, self.X)
        self.norm = lambda tensor: sp.sqrt(ddot(tensor, tensor) + 1e-16)
        
        self.local_dict = {}
        self._build_local_dict()
        
        # 注册配置文件中定义的未知变量
        if 'unknown_variables' in config.problem:
            for var_name in config.problem['unknown_variables']:
                self.register_unknown_var(var_name)
        
        # 先注册 derived_quantities 中标记为未知的量
        for k, v in config.problem['derived_quantities'].items():
            if v.strip() == '?':
                self.sp_unknown_quantities[k] = sp.symbols(k)

        # 其它设置
        self.tol = tol
        self.dim = self.X_dim + (1 if self.has_time else 0)
        
        # 计算网格间距（等距网格）
        self.dx = None
        self.dy = None
        self.dz = None
        for i, var in enumerate(self.spatial_vars):
            if var == 'x':
                self.dx = round(self.data_loader.grids[i][1] - self.data_loader.grids[i][0], 6)
            elif var == 'y':
                self.dy = round(self.data_loader.grids[i][1] - self.data_loader.grids[i][0], 6)
            elif var == 'z':
                self.dz = round(self.data_loader.grids[i][1] - self.data_loader.grids[i][0], 6)
        self.dt = round(self.data_loader.grids[-1][1] - self.data_loader.grids[-1][0], 6) if self.has_time else None
        
        # 现在解析导出物理量（所有符号都已经创建）
        self._parse_derived_quantities()

        # 等式 - 创建副本而不是直接引用
        self.equation_buffer = config.problem['known_equations'].copy()  # 添加 .copy()
        self.sp_equation = []
        
        # 约束
        self.constraints = {} # var: {"type": str, "fun": sp function}
        
        # 函数自变量
        self.args_data = []
        self.args_symbols = []
        self.load_args()
        #split args into train and valid, if k>1. 
        if K > 1:
            k_th_fold, tot_folds = datafold_tuple
            kf = KFold(n_splits=tot_folds, shuffle=True, random_state=0)
            nt = self.args_data[0].shape[-2]
            train_idx, valid_idx = list(kf.split(np.arange(nt)))[k_th_fold]
            self.train_args = [arg[...,train_idx,:] for arg in self.args_data]
            self.valid_args = [arg[...,valid_idx,:] for arg in self.args_data]
        else:
            self.train_args, self.valid_args = self.args_data, self.args_data
            
    def __repr__(self):
        return f"Now calculator's unknown quantities: {self.sp_unknown_quantities}\nNow calculator's derived quantities: {self.sp_derived_quantities}\nNow calculator's constants: {self.sp_constants}\nNow calculator's spatial variables: {self.spatial_vars}\nNow calculator's field variables: {self.field_vars}\nNow calculator's temporal variables: {self.temporal_vars}\nNow calculator's velocity vector: {self.V}\nNow calculator's known equations: {self.known_equations}\nNow calculator's constraints: {self.constraints}"

    def register_unknown_var(self, var_name):
        if var_name not in self.sp_unknown_quantities:
            self.sp_unknown_quantities[var_name] = sp.symbols(var_name)
            # print(f"Registered new variable: {var_name}")
        else:
            print(f"Variable '{var_name}' is already registered as an unknown quantity.")
    
    def get_unknown_var_list(self):
        return list(self.sp_unknown_quantities.keys())
    
    def remove_known_var(self, var_name):
        if var_name in self.sp_derived_quantities:
            self.sp_derived_quantities.pop(var_name)
            self.sp_unknown_quantities[var_name] = sp.symbols(var_name)
            # print(f"Deleted known variable '{var_name}', now registered as unknown.")
        else:
            print(f"Variable '{var_name}' not found in derived quantities.")
    
    def update_unknown_var(self, var_name, expr):
        """
        Args:
        var_name: str
        expr: str
        """
        if var_name not in self.sp_unknown_quantities:
            print(f"Warning: Updating unknown variable '{var_name}' with expression '{expr}'")
            self.register_unknown_var(var_name)
        
        if isinstance(expr, str):
            self.upd_local_dict()
            try:
                self.sp_derived_quantities[var_name] = sp.sympify(expr, locals=self.local_dict)
                self.sp_unknown_quantities.pop(var_name)
                print(f"Updated variable '{var_name}' with expression: {expr}")
            except Exception as e:
                print(f"Error updating variable '{var_name}' with expression '{expr}': {e}")
                self.sp_unknown_quantities[var_name] = sp.symbols(var_name)
    
    def _create_velocity_vector(self):
        if 'u' in self.sp_field_funcs and 'v' in self.sp_field_funcs:
            velocity_components = [self.sp_field_funcs['u'], self.sp_field_funcs['v']]
            if 'w' in self.sp_field_funcs:
                velocity_components.append(self.sp_field_funcs['w'])
            self.V = sp.Array(velocity_components)
        elif 'ux' in self.sp_field_funcs and 'uy' in self.sp_field_funcs:
            velocity_components = [self.sp_field_funcs['ux'], self.sp_field_funcs['uy']]
            if 'uz' in self.sp_field_funcs:
                velocity_components.append(self.sp_field_funcs['uz'])
            self.V = sp.Array(velocity_components)
        else:
            print("Warning: No velocity variables found, creating default V")
            self.V = sp.Array([sp.symbols('vx'), sp.symbols('vy')])

    def _parse_derived_quantities(self):
        to_parse = {}
        for k, v in self.config.problem['derived_quantities'].items():
            if v.strip() != '?':
                to_parse[k] = v
        
        if not to_parse:
            return

        self.upd_local_dict()

        max_iterations = len(to_parse) + 5 
        iteration = 0
        
        while to_parse and iteration < max_iterations:
            iteration += 1
            progress_made = False
            
            # 尝试解析剩余的量
            failed_this_round = {}
            for k, v in list(to_parse.items()):
                try:
                    parsed_expr = sp.sympify(v, locals=self.local_dict)
                    # 用实际表达式替换符号占位符
                    self.sp_derived_quantities[k] = parsed_expr
                    self.local_dict[k] = parsed_expr  # 更新字典中的定义
                    print(f"Successfully parsed derived quantity '{k}': {v}")
                    del to_parse[k]  # 从待解析列表中移除
                    progress_made = True
                except Exception as e:
                    failed_this_round[k] = (v, str(e))
            
            # 如果这轮没有任何进展，说明有循环依赖或无法解析的表达式
            if not progress_made:
                print("Warning: Some derived quantities could not be parsed:")
                for k, (v, error) in failed_this_round.items():
                    print(f"  '{k}': {v} - Error: {error}")
                break
        
        if iteration >= max_iterations:
            print("Warning: Maximum iterations reached in derived quantities parsing")

    def _build_local_dict(self):
        """
        Build a dictionary with all symbols, functions, and utilities for parsing expressions.
        """
        self.local_dict.update(self.sp_spatial_vars)
        self.local_dict.update(self.sp_field_funcs)
        self.local_dict.update(self.sp_constants)
        self.local_dict.update(self.sp_derived_quantities)
        self.local_dict.update(self.sp_unknown_quantities)

        if hasattr(self, 'sp_temporal_vars'):
            self.local_dict.update(self.sp_temporal_vars)

        if hasattr(self, 'V'):
            self.local_dict['V'] = self.V

        self.local_dict['grad'] = self.grad
        self.local_dict['norm'] = self.norm
        self.local_dict['div'] = lambda a: div(a, self.X)
        self.local_dict['dot'] = dot
        self.local_dict['ddot'] = ddot
        self.local_dict['conserve'] = conserve
        self.local_dict['transpose'] = sp.transpose
        self.local_dict['Array'] = sp.Array
        self.local_dict['sqrt'] = sp.sqrt
        self.local_dict['sin'] = sp.sin
        self.local_dict['cos'] = sp.cos
        self.local_dict['exp'] = sp.exp
        self.local_dict['log'] = sp.log
        self.local_dict['Derivative'] = sp.Derivative
        self.local_dict['X_dim'] = self.X_dim

        # Add differential operators that need time variable
        if self.has_time:
            self.local_dict['DDt'] = lambda f: DDt(f, self.V, self.X, self.t)
            # Update conserve to use time variable when available
            self.local_dict['conserve'] = lambda f: conserve(f, self.V, self.X, self.t)
    
    # TODO: 优化，这里现在太慢了
    def upd_local_dict(self):
        # 迭代更新导出量
        symbols = {sp.Symbol(k): v for k, v in self.sp_derived_quantities.items() 
               if isinstance(v, (int, float, sp.Basic))}
    
        changed = True
        while changed:
            changed = False
            for key, value in self.sp_derived_quantities.items():
                if isinstance(value, sp.Basic):
                    new_value = value.subs(symbols)
                    if new_value != value:  # 如果替换后值变化
                        self.sp_derived_quantities[key] = new_value
                        symbols[sp.Symbol(key)] = new_value
                        changed = True

        self.local_dict.update(self.sp_derived_quantities)
        self.local_dict.update(self.sp_unknown_quantities)

    def get_sp_equation(self):
        """
        Parse and set all equations.
        All parameters should be settled here.
        """
        if len(self.sp_equation) > 0:
            return
        self.upd_local_dict()
        for eq_str in self.equation_buffer:
            try:
                # Parse the equation string
                eq = sp.sympify(eq_str, locals=self.local_dict)
                eq = sp_simplify_with_timeout(eq)
                self.sp_equation.append(eq)
            except Exception as e:
                print(f"Error parsing equation '{eq_str}': {e}")
        
    def get_new_equation(self, equation_str):
        self.equation_buffer.append(equation_str)
        
    def add_constraint(self, constraint_str, var_name, verbose=False):
        """
        Get a constraint function from a string.
        Returns sympy expression.
        """
        self.upd_local_dict()
        try:
            constraint = [{"type":"ineq", "fun":sp.sympify(constraint_str, locals=self.local_dict) - self.tol}]

            self.constraints[var_name] = constraint
            if verbose:
                print(f"Added constraint for variable '{var_name}': {constraint_str}")
            return constraint
        except Exception as e:
            print(f"Error parsing constraint '{constraint_str}': {e}")
            return None
        
    def get_constr_dict_list(self):
        constr_dict_list = list()
        for pa, constr_dicts in self.constraints.items():
            for cd in constr_dicts:
                c = {"type":cd["type"], "fun":sp.lambdify([self.sp_unknown_quantities.values()], cd["fun"]), "name":pa}
                constr_dict_list.append(c)
        return constr_dict_list

    def load_args(self):
        """
        Get the arguments for the residual functions.
        """
        # Map remaining scalar field variables
        for var in self.field_vars:
            self.args_symbols.append(self.sp_field_funcs[var])

            # First-order spatial derivatives
            for i in range(len(self.space_axis)):
                self.args_symbols.append(sp.Derivative(self.sp_field_funcs[var], self.X[i]))
            
            # Second-order spatial derivatives
            for i in range(len(self.space_axis)):
                for j in range(len(self.space_axis)):
                    self.args_symbols.append(sp.Derivative(self.sp_field_funcs[var], self.X[i], self.X[j]))

            # Time derivative if time is included
            if self.has_time:
                self.args_symbols.append(sp.Derivative(self.sp_field_funcs[var], self.t))
        
        self.args_data = self.data_loader.get_args_data(verbose=self.config.verbose)
        def pre_process(arr):
            # clip boundary
            n_clip = 5
            if self.dim == 3:
                arr = arr[n_clip: -n_clip, n_clip: -n_clip, n_clip: -n_clip]
            elif self.dim == 2:
                arr = arr[n_clip: -n_clip, n_clip: -n_clip]
            elif self.dim == 1:
                arr = arr[n_clip: -n_clip]
            elif self.dim == 4:
                arr = arr[n_clip: -n_clip, n_clip: -n_clip, n_clip: -n_clip, n_clip: -n_clip]
            return arr
        self.args_data = list(map(pre_process, self.args_data))

    def gen_np_func(self, sp_res_func_list, verbose=False, src_strs=None):
        """
        Convert sympy residual expressions to numpy functions for numerical evaluation.
        
        Args:
            sp_res_func_list: List of sympy expressions to convert
        Returns:
            res_func_list: List of numpy functions corresponding to residuals
        """
        # Set up parameters list
        params = self.sp_unknown_quantities.values()
        # Convert each sympy expression to a numpy function with numerical stability
        def to_np_func_stable(sp_func):
            base_func = sp.lambdify([self.args_symbols, params], sp_func, 'numpy')
            
            def stable_wrapper(args, params_vals):
                # 添加数值稳定性检查，并确保结果为可处理的浮点数组
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    try:
                        result = base_func(args, params_vals)
                    except Exception as e:
                        # 返回一个大但有限的值
                        return np.full_like(args[0], 1e10)

                    # 尝试将结果转换为浮点数组；若为复数或无法转换，则进行降级处理
                    try:
                        arr = np.asarray(result)
                        # 如果出现复数，若虚部显著则判为无效；否则取实部
                        if np.iscomplexobj(arr):
                            if np.any(np.abs(np.imag(arr)) > 1e-12):
                                return np.full_like(args[0], 1e10)
                            arr = np.real(arr)
                        arr = arr.astype(float, copy=False)
                    except Exception:
                        return np.full_like(args[0], 1e10)

                    # 对于无穷，保留符号，替换为大但有限的值；NaN 置 0
                    if np.any(~np.isfinite(arr)):
                        sign = np.sign(arr)
                        arr = np.where(np.isinf(arr), sign * 1e10, arr)
                        arr = np.where(np.isnan(arr), 0.0, arr)
                    return arr
            
            return stable_wrapper
        
        res_func_list = []
        residual_labels = []  # one label per flattened residual function
        res_idx_list = []
        idx = 0
        
        built = 0
        for eq_i, sp_res in enumerate(sp_res_func_list):
            # pick a readable source string (original equation if provided, else str of sympy expr)
            src_label = None
            try:
                if src_strs is not None and eq_i < len(src_strs):
                    src_label = str(src_strs[eq_i])
            except Exception:
                src_label = None
            if src_label is None:
                try:
                    src_label = str(sp_res)
                except Exception:
                    src_label = f"expr[{eq_i}]"

            if hasattr(sp_res, "shape"):  # array variable (vector/tensor)
                # flatten over first axis length
                res_func_list += [to_np_func_stable(i) for i in sp_res]
                # label each component
                comp_n = len(sp_res)
                residual_labels += [f"{src_label} [comp {k}]" for k in range(comp_n)]
                res_idx_list.append(list(range(idx, idx+comp_n)))
                idx += comp_n
                built += comp_n
            else:  # scalar variable
                res_func_list.append(to_np_func_stable(sp_res))
                residual_labels.append(src_label)
                res_idx_list.append([idx,])
                idx += 1
                built += 1
        
        return res_func_list
    
    def get_loss_func(self, deci_list_len, reg_scale=1, pool_size=5, mode="train", sample_ratio=1, seed=42, debug_eval=False, eval_time_budget=None):
        """
        Args:
        deci_list_len: int, length of decision dictionary for regularization.
        reg_scale: float, regularization scale factor.
        pool_size: int, pooling size for spatial dimensions.
        mode: str, "train" or "valid" mode.
        sample_ratio: float, ratio of spatial points to sample (0 < sample_ratio <= 1).
        seed: int, random seed for reproducible sampling.
        
        Returns:
        loss_func: function, the loss function to minimize.
        loss_func_list: function, returns a list of mean squared errors for each residual.
        """
        self.get_sp_equation()
        tot_count_ops = sum( [r.count_ops() for r in self.sp_equation if hasattr(r, "count_ops")] )
        reg_coefs = reg_scale * np.array([1e-5, 1e-5, 1e-7])  # reg_coefs of [len(deci_list), len(params), tot_count_ops]
        reg_list = lambda params: [deci_list_len, len(params), tot_count_ops]
        reg_func = lambda params: reg_coefs.dot(np.array(reg_list(params)))

        if mode == "train":
            args_data = self.train_args
        else:
            args_data = self.valid_args
            
        np.random.seed(seed)
        if len(args_data) > 0:
            if self.has_time:
                # 形状: (x, y, t, ...) -> 前两个是空间维度，第三个是时间维度
                spatial_shape = args_data[0].shape[:2]  # (nx, ny)
                time_steps = args_data[0].shape[2]  # nt
            else:
                # 形状: (x, y, ...) -> 前两个是空间维度
                spatial_shape = args_data[0].shape[:2]  # (nx, ny)
                time_steps = None
            
            # 计算要采样的空间区域数量
            total_spatial_area = spatial_shape[0] * spatial_shape[1]
            target_spatial_area = int(total_spatial_area * sample_ratio)
            
            # 每个区域的大小（保证能进行pooling）
            region_height = max(pool_size * 2, spatial_shape[0] // 4)  # 至少是pooling size的2倍
            region_width = max(pool_size * 2, spatial_shape[1] // 4)
            region_area = region_height * region_width
            
            # 计算需要采样的区域数量
            n_regions = max(1, target_spatial_area // region_area)
            n_regions = min(n_regions, (spatial_shape[0] // region_height) * (spatial_shape[1] // region_width))
            
            # 生成随机的空间区域位置
            spatial_regions = []
            for _ in range(n_regions):
                start_x = np.random.randint(0, max(1, spatial_shape[0] - region_height + 1))
                start_y = np.random.randint(0, max(1, spatial_shape[1] - region_width + 1))
                spatial_regions.append((slice(start_x, start_x + region_height), 
                                      slice(start_y, start_y + region_width)))
            
            # 时间步采样（如果有时间维度）
            if self.has_time and time_steps > 1:
                # 采样部分时间步
                n_time_samples = max(1, int(time_steps * sample_ratio * 0.3))
                time_indices = np.random.choice(time_steps, size=n_time_samples, replace=False)
                time_indices = np.sort(time_indices)  # 保持时间顺序
            else:
                time_indices = None
        
        def sample_args(args):
            """对args进行多区域和时间步采样"""
            if len(args) == 0:
                return args
            sampled_args = []
            
            for arg in args:
                # 从所有空间区域中采样数据
                region_samples = []
                for region_slice in spatial_regions:
                    if self.has_time:
                        # 形状: (x, y, t, ...) -> 先采样空间区域，再采样时间步
                        if time_indices is not None:
                            # 采样空间区域和时间步
                            region_data = arg[region_slice]  # (region_h, region_w, t, ...)
                            region_data = region_data[:, :, time_indices]  # (region_h, region_w, n_time_samples, ...)
                        else:
                            # 只采样空间区域
                            region_data = arg[region_slice]  # (region_h, region_w, t, ...)
                    else:
                        # 形状: (x, y, ...) -> 只采样空间区域
                        region_data = arg[region_slice]  # (region_h, region_w, ...)
                    
                    region_samples.append(region_data)
                
                # 将多个区域的数据合并（沿着第一个维度拼接）
                if len(region_samples) > 1:
                    sampled_arg = np.concatenate(region_samples, axis=0)
                else:
                    sampled_arg = region_samples[0]
                
                sampled_args.append(sampled_arg)
            
            return sampled_args

        t0_gen = time.time()
        # Pass original equation strings to keep labels aligned
        try:
            src_eqs = list(self.equation_buffer)
        except Exception:
            src_eqs = None
        res_func_list = self.gen_np_func(self.sp_equation, verbose=True, src_strs=src_eqs)
        
        # 预采样一次，避免每次评估都重新切片与拼接大数组
        sampled_args = sample_args(args_data)

        def mse_list_sampled_noresample(params):
            # 直接使用预采样的 sampled_args 来计算每个 residual 的均方
            residuals = []
            t_eval0 = time.time()
            labels = getattr(self, "_last_residual_labels", None)
            for i, res in enumerate(res_func_list):
                t_i0 = time.time()
                res_values = res(sampled_args, params)
                # 对采样区域进行pooling，保持pool_size不变
                if len(sampled_args) > 0:
                    pooled_res = pooling(res_values, (pool_size, pool_size))
                else:
                    pooled_res = res_values
                residuals.append((pooled_res**2).mean())
                if debug_eval:
                    dt_i = time.time() - t_i0
                    # Only print if notably slow to avoid noise
                    if dt_i > 0.2:
                        prefix = f"[Calc][loss][debug] residual {i}/{len(res_func_list)}"
                        if isinstance(labels, list) and i < len(labels):
                            try:
                                lab = labels[i]
                                lab = lab if len(lab) < 200 else (lab[:197] + '...')
                                prefix += f" | {lab}"
                            except Exception:
                                pass
                        print(f"{prefix} dt={dt_i:.3f}s shape={getattr(res_values, 'shape', None)}")
                if eval_time_budget is not None and (time.time() - t_eval0) > eval_time_budget:
                    print(f"[Calc][loss][debug] eval budget exceeded after residual {i}")
                    break
            return residuals
        
        # pooling is applied to the first two axises of res, i.e. x and y axis
        mse_func_noresample = lambda params: sum(mse_list_sampled_noresample(params))
        
        if mode == "train":
            loss_func = lambda params: mse_func_noresample(params) + reg_func(params)
            loss_func_list = lambda params: mse_list_sampled_noresample(params)
        else:
            loss_func = lambda params: mse_func_noresample(params) + reg_func(params)
            loss_func_list = lambda params: mse_list_sampled_noresample(params)
        return loss_func, loss_func_list
    
    # def get_loss_func(self, deci_list_len, reg_scale=1, pool_size=5, mode="train"):
    #     """
    #     Args:
    #     deci_list_len: int, length of decision dictionary for regularization.
        
    #     Returns:
    #     loss_func: function, the loss function to minimize.
    #     loss_func_list: function, returns a list of mean squared errors for each residual.
    #     """
    #     self.get_sp_equation()
    #     tot_count_ops = sum( [r.count_ops() for r in self.sp_equation if hasattr(r, "count_ops")] )
    #     reg_coefs = reg_scale * np.array([1e-5, 1e-5, 1e-7])  # reg_coefs of [len(deci_list), len(params), tot_count_ops]
    #     reg_list = lambda params: [deci_list_len, len(params), tot_count_ops]
    #     reg_func = lambda params: reg_coefs.dot(np.array(reg_list(params)))

    #     res_func_list = self.gen_np_func(self.sp_equation, verbose=False)
    #     mse_list = lambda args, params: [(pooling(res(args, params), (pool_size, pool_size))**2).mean()  for res in res_func_list]
    #     # pooling is applied to the first two axises of res, i.e. x and y axis
    #     mse_func = lambda args, params: sum(mse_list(args, params))
        
    #     if mode == "train":
    #         loss_func = lambda params: mse_func(self.train_args, params) + reg_func(params)
    #         loss_func_list = lambda params: mse_list(self.train_args, params)
    #     else:
    #         loss_func = lambda params: mse_func(self.valid_args, params) + reg_func(params)
    #         loss_func_list = lambda params: mse_list(self.valid_args, params)
            
    #     return loss_func, loss_func_list