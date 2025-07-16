import numpy as np
from pix.utils.numpy_utils import np_grad

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.spatial_vars = [] # str
        self.field_vars = []   # str - variable names
        self.field_data = []   # numpy arrays - actual data
        self.temporal_vars = []
        self.u = None
        self.grids = None
    
    def get_raw_data(self, dataset_path, datasource="COMSOL", verbose=False):
        '''
        Automatically detects which variables are spatial coordinates (x/y/z/...) and which are field variables.
        All spatial coordinates are put into grids, t is always separated and put at the end if present.
        '''
        variables = self.config.problem['variables']
        if datasource == "COMSOL":
            data = np.load(dataset_path)
            grids = []
            t_array = None
            for var in variables:
                if var not in data:
                    print(f"Warning: variable '{var}' not found in data, skipping")
                    continue
                if var == 't':
                    t_array = data[var].reshape(-1)
                    continue
                arr = data[var]
                if var == 'x' or var == 'y' or var == 'z':
                    if arr.ndim == 3:
                        if var == 'x':
                            grids.append(arr[:, 0, 0])
                        elif var == 'y':
                            grids.append(arr[0, :, 0])
                        elif var == 'z':
                            grids.append(arr[0, 0, :])
                    elif arr.ndim == 2:
                        if var == 'x':
                            grids.append(arr[:, 0])
                        elif var == 'y':
                            grids.append(arr[0, :])
                        elif var == 'z':
                            grids.append(arr[0, :])
                    elif arr.ndim == 1:
                        grids.append(arr)
                    else:
                        raise ValueError(f"Unsupported dimension for spatial coordinate '{var}': {arr.ndim}")
                    self.spatial_vars.append(var)
                else:
                    # 不是空间坐标，作为 field variable
                    self.field_vars.append(var)  # Store the variable name
                    self.field_data.append(arr)  # Store the actual data

            if len(self.field_data) == 0:
                raise ValueError("No valid variables besides x and t found in data")

            self.u = np.stack(self.field_data, axis=-1)
            # t 单独放最后
            if t_array is not None:
                grids.append(t_array)
                self.temporal_vars.append('t')
            if verbose:
                loaded_vars = [var for var in variables if var in data and var not in self.spatial_vars and var != 't']
                print(f"Loaded variables: {loaded_vars}")
                print(f"Data shape: {self.u.shape}")
                print(f"Grid shapes: {[g.shape for g in grids]}")
        else:
            raise ValueError(f"Dataset source not supported: {datasource}")
        self.grids = tuple(grids)
        
        n_clip = 5
        if len(self.u.shape)==4:  # 3 dimensional data (2 space dim, 1 temporal dim)
            if n_clip > 0:
                self.u = self.u[n_clip: -n_clip, n_clip: -n_clip,...]
                self.grids = tuple(g[n_clip: -n_clip] for g in self.grids[:-1]) + (self.grids[-1],)

    def get_args_data(self, verbose=False):
        field_data = {}
        for i, var in enumerate(self.field_vars):
            if i < self.u.shape[-1]:
                field_data[var] = self.u[..., i]
            else:
                print(f"Warning: Variable '{var}' not found in data, using default value (ones)")
                field_data[var] = np.ones_like(self.u[..., 0])

        args_data = []
        for var, data in field_data.items():
            args_data.append(data)  # 原始变量
            if len(self.grids) > 1:
                grad_ = np_grad([data], self.grids, is_time_grad=False)  # 空间一阶导数
                grad_grad_ = np_grad(grad_, self.grids, is_time_grad=False)  # 空间二阶导数
                dt_ = np_grad([data], self.grids, is_time_grad=True)  # 时间导数
                
                args_data.extend(grad_)      
                args_data.extend(grad_grad_) 
                args_data.extend(dt_)                
        
        return args_data
