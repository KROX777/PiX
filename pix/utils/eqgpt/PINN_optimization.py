"""
PINN-based coefficient optimization for PiX.

This module provides Physics-Informed Neural Network (PINN) optimization
that directly uses the symbolic PDEs from Calculator.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import sympy as sp
from torch.autograd import Variable
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()
    def forward(self, x):
        return torch.sin(x)


class Rational(torch.nn.Module):
    def __init__(self, Data_Type=torch.float32, Device=torch.device('cpu')):
        super(Rational, self).__init__()
        self.a = torch.nn.parameter.Parameter(
            torch.tensor((1.1915, 1.5957, 0.5, .0218),
                        dtype=Data_Type, device=Device))
        self.a.requires_grad_(True)
        self.b = torch.nn.parameter.Parameter(
            torch.tensor((2.3830, 0.0, 1.0),
                        dtype=Data_Type, device=Device))
        self.b.requires_grad_(True)

    def forward(self, X: torch.tensor):
        a = self.a
        b = self.b
        N_X = a[0] + X*(a[1] + X*(a[2] + a[3]*X))
        D_X = b[0] + X*(b[1] + b[2]*X)
        return N_X/D_X


class NN(torch.nn.Module):
    def __init__(self,
                 Num_Hidden_Layers: int = 3,
                 Neurons_Per_Layer: int = 20,
                 Input_Dim: int = 1,
                 Output_Dim: int = 1,
                 Data_Type: torch.dtype = torch.float32,
                 Device: torch.device = torch.device('cpu'),
                 Activation_Function: str = "Tanh",
                 Batch_Norm: bool = False):
        super(NN, self).__init__()
        self.Input_Dim: int = Input_Dim
        self.Output_Dim: int = Output_Dim
        self.Num_Hidden_Layers: int = Num_Hidden_Layers
        self.Batch_Norm: bool = Batch_Norm
        self.Layers = torch.nn.ModuleList()
        if Batch_Norm == True:
            self.Norm_Layer = torch.nn.BatchNorm1d(num_features=Input_Dim, dtype=Data_Type, device=Device)
        self.Layers.append(torch.nn.Linear(in_features=Input_Dim, out_features=Neurons_Per_Layer, bias=True).to(dtype=Data_Type, device=Device))
        for i in range(1, Num_Hidden_Layers):
            self.Layers.append(torch.nn.Linear(in_features=Neurons_Per_Layer, out_features=Neurons_Per_Layer, bias=True).to(dtype=Data_Type, device=Device))
        self.Layers.append(torch.nn.Linear(in_features=Neurons_Per_Layer, out_features=Output_Dim, bias=True).to(dtype=Data_Type, device=Device))
        
        if Activation_Function == "Tanh" or Activation_Function == "Rational":
            Gain = 5./3. if Activation_Function == "Tanh" else 1.41
            for i in range(self.Num_Hidden_Layers + 1):
                torch.nn.init.xavier_normal_(self.Layers[i].weight, gain=Gain)
                torch.nn.init.zeros_(self.Layers[i].bias)
        elif Activation_Function == "Sin":
            import math
            a = 3./math.sqrt(Neurons_Per_Layer)
            for i in range(0, self.Num_Hidden_Layers + 1):
                torch.nn.init.uniform_(self.Layers[i].weight, -a, a)
                torch.nn.init.zeros_(self.Layers[i].bias)
        
        self.Activation_Functions = torch.nn.ModuleList()
        if Activation_Function == "Tanh":
            for i in range(Num_Hidden_Layers): 
                self.Activation_Functions.append(torch.nn.Tanh())
        elif Activation_Function == "Sin":
            for i in range(Num_Hidden_Layers): 
                self.Activation_Functions.append(Sin())
        elif Activation_Function == "Rational":
            for i in range(Num_Hidden_Layers): 
                self.Activation_Functions.append(Rational(Data_Type=Data_Type, Device=Device))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.Batch_Norm == True: 
            X = self.Norm_Layer(X)
        for i in range(0, self.Num_Hidden_Layers): 
            X = self.Activation_Functions[i](self.Layers[i](X))
        return self.Layers[self.Num_Hidden_Layers](X)


class MultiFieldNet(nn.Module):
    """Neural networks for multiple field variables."""
    def __init__(self, field_names, input_dim, hidden_layers=5, neurons=40):
        super(MultiFieldNet, self).__init__()
        self.field_names = field_names
        self.nets = nn.ModuleDict()
        for field in field_names:
            self.nets[field] = NN(
                Num_Hidden_Layers=hidden_layers,
                Neurons_Per_Layer=neurons,
                Input_Dim=input_dim,
                Output_Dim=1,
                Data_Type=torch.float32,
                Device=device,
                Activation_Function='Tanh'
            )
    
    def forward(self, coords, field_name=None):
        if field_name is not None:
            return self.nets[field_name](coords)
        return {f: self.nets[f](coords) for f in self.field_names}


class PDEResidualComputer:
    """
    Computes PDE residuals from symbolic equations using neural networks.
    """
    def __init__(self, calculator, field_nets, coords, var_names, spatial_vars, temporal_vars):
        self.calculator = calculator
        self.field_nets = field_nets
        self.coords = coords
        self.var_names = var_names
        self.spatial_vars = spatial_vars
        self.temporal_vars = temporal_vars
        self.n_points = coords.shape[0]
        
        # Map variable names to indices
        self.var_idx = {v: i for i, v in enumerate(var_names)}
        
        # Cache for derivatives
        self.derivative_cache = {}
        
    def get_field_value(self, field_name):
        """Get neural network prediction for a field."""
        return self.field_nets.nets[field_name](self.coords)
    
    def compute_derivative(self, field_name, var_name, order=1):
        """Compute derivative of field w.r.t. variable."""
        cache_key = (field_name, var_name, order)
        if cache_key in self.derivative_cache:
            return self.derivative_cache[cache_key]
        
        field_net = self.field_nets.nets[field_name]
        var_idx = self.var_idx[var_name]
        
        # First derivative
        output = field_net(self.coords)
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=self.coords,
            create_graph=True,
            retain_graph=True
        )[0]
        result = grad[:, var_idx:var_idx+1]
        
        # Higher order derivatives
        for _ in range(1, order):
            grad = torch.autograd.grad(
                outputs=result.sum(),
                inputs=self.coords,
                create_graph=True,
                retain_graph=True
            )[0]
            result = grad[:, var_idx:var_idx+1]
        
        self.derivative_cache[cache_key] = result
        return result
    
    def evaluate_expr(self, expr, coeff_values=None):
        """
        Evaluate a sympy expression using neural network values.
        
        Args:
            expr: SymPy expression
            coeff_values: Dict of coefficient name -> value
        
        Returns:
            Torch tensor of evaluated expression
        """
        # Handle different expression types
        if expr.is_Number:
            return torch.full((self.n_points, 1), float(expr), device=device)
        
        # Handle symbols and function applications
        if expr.is_Symbol:
            # Check if it's a coefficient
            if coeff_values and str(expr) in coeff_values:
                return torch.full((self.n_points, 1), coeff_values[str(expr)], device=device)
            # Check if it's a coordinate variable
            if str(expr) in self.var_idx:
                idx = self.var_idx[str(expr)]
                return self.coords[:, idx:idx+1]
            # Unknown symbol - return zero
            return torch.zeros(self.n_points, 1, device=device)
        
        # Handle function applications like u(x,y,t), rho(x,y,t)
        if hasattr(expr, 'func') and hasattr(expr, 'args') and len(expr.args) > 0:
            func_name = str(expr.func)
            # Check if it's a field variable
            if func_name in self.field_nets.nets:
                return self.get_field_value(func_name)
        
        if expr.__class__.__name__ == 'Derivative':
            # Extract field and differentiation variables
            field_func = expr.args[0]
            diff_specs = expr.args[1:]  # Each is a Tuple: (var, order) or just var
            
            # Get field name from function
            if hasattr(field_func, 'func'):
                field_name = str(field_func.func)
            else:
                field_name = str(field_func)
            
            # Parse differentiation specifications
            deriv_specs = []
            for spec in diff_specs:
                if hasattr(spec, '__iter__') and len(spec) >= 1:
                    var_name = str(spec[0])
                    order = int(spec[1]) if len(spec) > 1 else 1
                else:
                    var_name = str(spec)
                    order = 1
                deriv_specs.append((var_name, order))
            
            # Compute derivative step by step
            result = self.get_field_value(field_name)
            for var_name, order in deriv_specs:
                var_idx = self.var_idx.get(var_name)
                if var_idx is None:
                    raise ValueError(f"Unknown variable '{var_name}' in derivative")
                
                # Apply derivative order times
                for _ in range(order):
                    grad = torch.autograd.grad(
                        outputs=result.sum(),
                        inputs=self.coords,
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    result = grad[:, var_idx:var_idx+1]
                    # Clip to prevent explosion
                    result = torch.clamp(result, -1e6, 1e6)
            
            return result
        
        if expr.__class__.__name__ == 'Add':
            result = torch.zeros(self.n_points, 1, device=device)
            for arg in expr.args:
                term = self.evaluate_expr(arg, coeff_values)
                # Check for nan/inf
                if torch.isnan(term).any() or torch.isinf(term).any():
                    continue
                result = result + torch.clamp(term, -1e6, 1e6)
            return result
        
        if expr.__class__.__name__ == 'Mul':
            result = torch.ones(self.n_points, 1, device=device)
            for arg in expr.args:
                factor = self.evaluate_expr(arg, coeff_values)
                if torch.isnan(factor).any() or torch.isinf(factor).any():
                    continue
                result = result * torch.clamp(factor, -1e6, 1e6)
            return result
        
        if expr.__class__.__name__ == 'Pow':
            base = self.evaluate_expr(expr.args[0], coeff_values)
            exp = float(expr.args[1])
            # Clamp base to prevent numerical issues
            # For fractional exponents (like sqrt), ensure base >= 0
            if exp < 1.0 and exp > 0:
                base = torch.clamp(base, min=1e-8)
            elif exp < 0:
                base = torch.clamp(base, min=1e-8, max=1e8)
            else:
                base = torch.clamp(base, -1e8, 1e8)
            return torch.pow(base, exp)
        
        if expr.__class__.__name__ == 'sqrt':
            arg = self.evaluate_expr(expr.args[0], coeff_values)
            # Ensure non-negative inside sqrt
            arg = torch.clamp(arg, min=1e-8)
            return torch.sqrt(arg)
        
        # Handle Array/ImmutableDenseNDimArray
        if expr.__class__.__name__ in ('Array', 'ImmutableDenseNDimArray'):
            # Flatten array and compute each element
            elements = []
            # Use flat() or iterate through shape
            if hasattr(expr, 'flatten'):
                flat_exprs = expr.flatten()
            else:
                # Manually iterate
                flat_exprs = [expr[i] for i in range(len(expr))]
            for e in flat_exprs:
                elements.append(self.evaluate_expr(e, coeff_values))
            # Stack and compute mean of squares (sum of squared residuals)
            stacked = torch.stack(elements, dim=1)
            return (stacked ** 2).mean(dim=1, keepdim=True)
        
        # Handle Matrix
        if expr.__class__.__name__ == 'Matrix':
            elements = []
            for i in range(expr.shape[0]):
                for j in range(expr.shape[1]):
                    elements.append(self.evaluate_expr(expr[i, j], coeff_values))
            return torch.stack(elements, dim=1).mean(dim=1, keepdim=True)
        
        # Handle common functions
        if expr.__class__.__name__ == 'sin':
            return torch.sin(self.evaluate_expr(expr.args[0], coeff_values))
        if expr.__class__.__name__ == 'cos':
            return torch.cos(self.evaluate_expr(expr.args[0], coeff_values))
        if expr.__class__.__name__ == 'exp':
            return torch.exp(self.evaluate_expr(expr.args[0], coeff_values))
        if expr.__class__.__name__ == 'log':
            return torch.log(self.evaluate_expr(expr.args[0], coeff_values))
        if expr.__class__.__name__ == 'Abs':
            return torch.abs(self.evaluate_expr(expr.args[0], coeff_values))
        
        # Handle negative numbers
        if expr.__class__.__name__ == 'NegativeOne' or (hasattr(expr, 'is_number') and float(expr) < 0):
            return torch.full((self.n_points, 1), float(expr), device=device)
        
        # Default: try to evaluate numerically
        try:
            val = float(expr.evalf())
            return torch.full((self.n_points, 1), val, device=device)
        except:
            print(f"Warning: Unknown expression type {type(expr).__name__}: {str(expr)[:50]}")
            return torch.zeros(self.n_points, 1, device=device)
    
    def compute_residual(self, eq, coeff_values=None):
        """Compute residual for a single equation."""
        return self.evaluate_expr(eq, coeff_values)


def optimize_with_pinn_impl(calculator, cfg, deci_list, init_params, mse_func=None):
    """
    PINN-based coefficient optimization using calculator's symbolic equations.
    """
    start_time = time.time()
    
    # Get config
    pinn_cfg = cfg.get("pinn_config", {})
    epochs = pinn_cfg.get("epochs", 100)
    lr = pinn_cfg.get("learning_rate", 1e-3)
    pde_weight = pinn_cfg.get("pde_loss_weight", 1.0)
    data_weight = pinn_cfg.get("data_loss_weight", 0.1)
    max_points = pinn_cfg.get("max_points", 5000)
    
    # Ensure equations are parsed
    if len(calculator.sp_equation) == 0:
        calculator.get_sp_equation()
    
    # Get data loader info
    dl = calculator.data_loader
    field_vars = dl.field_vars
    spatial_vars = dl.spatial_vars
    temporal_vars = dl.temporal_vars
    var_names = spatial_vars + temporal_vars
    
    input_dim = len(var_names)
    n_coeffs = len(init_params)
    
    print(f"PINN: Fields: {field_vars}")
    print(f"PINN: Vars: {var_names}")
    print(f"PINN: {len(calculator.sp_equation)} equations")
    print(f"PINN: {n_coeffs} coefficients: {list(calculator.sp_unknown_quantities.keys())}")
    
    # Create multi-field neural network
    field_nets = MultiFieldNet(
        field_names=field_vars,
        input_dim=input_dim,
        hidden_layers=pinn_cfg.get("hidden_layers", 5),
        neurons=pinn_cfg.get("neurons", 40)
    ).to(device)
    
    # Prepare training data
    grid_arrays = dl.grids
    mesh = np.meshgrid(*grid_arrays, indexing='ij')
    flat_coords = [m.flatten() for m in mesh]
    database_np = np.column_stack(flat_coords)
    
    # Normalize coordinates
    coord_mins = database_np.min(axis=0)
    coord_maxs = database_np.max(axis=0)
    coord_ranges = coord_maxs - coord_mins
    coord_ranges[coord_ranges == 0] = 1
    database_np = 2 * (database_np - coord_mins) / coord_ranges - 1
    
    # Sample if too large
    n_total = len(database_np)
    if n_total > max_points:
        print(f"PINN: Sampling {max_points} from {n_total} points")
        np.random.seed(42)
        indices = np.random.choice(n_total, max_points, replace=False)
        database_np = database_np[indices]
        field_data_full = dl.u
        field_data_sampled = {}
        for i, field in enumerate(field_vars):
            flat_data = field_data_full[..., i].flatten()
            field_data_sampled[field] = flat_data[indices]
    else:
        field_data_sampled = {}
        for i, field in enumerate(field_vars):
            field_data_sampled[field] = dl.u[..., i].flatten()
    
    # Prepare tensors
    coords = torch.from_numpy(database_np).float().to(device)
    coords.requires_grad = True
    
    field_targets = {}
    for field in field_vars:
        field_targets[field] = torch.from_numpy(
            field_data_sampled[field].reshape(-1, 1)
        ).float().to(device)
    
    # Setup coefficients
    if n_coeffs > 0:
        coeffs = Variable(torch.tensor(init_params, dtype=torch.float32, device=device), requires_grad=True)
        coeff_names = list(calculator.sp_unknown_quantities.keys())
        params_to_optimize = [{'params': field_nets.parameters(), 'lr': lr},
                              {'params': [coeffs], 'lr': lr}]
    else:
        coeffs = None
        coeff_names = []
        params_to_optimize = [{'params': field_nets.parameters(), 'lr': lr}]
    
    optimizer = torch.optim.Adam(params_to_optimize)
    mse_loss = nn.MSELoss()
    
    # Create residual computer
    residual_computer = PDEResidualComputer(
        calculator, field_nets, coords, var_names, spatial_vars, temporal_vars
    )
    
    # Training loop
    print(f"PINN: Training for {epochs} epochs...")
    min_loss = float('inf')
    best_coeffs = init_params.copy() if n_coeffs > 0 else np.array([])
    
    for epoch in tqdm(range(epochs), desc="PINN"):
        optimizer.zero_grad()
        
        # Clear derivative cache each epoch
        residual_computer.derivative_cache = {}
        
        # Get current coefficient values
        coeff_values = {}
        if n_coeffs > 0:
            coeff_values = {name: coeffs[i].item() for i, name in enumerate(coeff_names)}
        
        # Data loss
        loss_data = torch.tensor(0.0, device=device)
        for field in field_vars:
            pred = field_nets(coords, field)
            loss_data += mse_loss(pred, field_targets[field])
        loss_data = loss_data / len(field_vars)
        
        # Physics loss (PDE residual)
        loss_pde = torch.tensor(0.0, device=device)
        valid_eqs = 0
        eq_losses = []
        
        for i, eq in enumerate(calculator.sp_equation):
            try:
                residual = residual_computer.compute_residual(eq, coeff_values)
                eq_loss = torch.mean(residual**2)
                eq_losses.append(eq_loss.item())
                loss_pde += eq_loss
                valid_eqs += 1
            except Exception as e:
                if epoch == 0:
                    print(f"  Warning: EQ {i} failed: {e}")
                continue
        
        if valid_eqs > 0:
            loss_pde = loss_pde / valid_eqs
        
        if epoch == 0:
            print(f"  Valid equations: {valid_eqs}/{len(calculator.sp_equation)}")
            print(f"  EQ losses: {[f'{l:.2e}' for l in eq_losses]}")
        
        # Total loss
        loss = data_weight * loss_data + pde_weight * loss_pde
        
        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            if epoch == 0:
                print(f"Warning: Invalid loss at epoch {epoch}, using data loss only")
            loss = loss_data  # Fall back to data loss only
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Data loss also invalid at epoch {epoch}, skipping")
                continue
        
        loss.backward()
        
        # Gradient clipping (more aggressive)
        torch.nn.utils.clip_grad_norm_(field_nets.parameters(), 0.1)
        if coeffs is not None:
            torch.nn.utils.clip_grad_norm_([coeffs], 0.1)
        
        optimizer.step()
        
        if loss.item() < min_loss:
            min_loss = loss.item()
            if n_coeffs > 0:
                best_coeffs = coeffs.detach().cpu().numpy().copy()
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            coeff_str = f"coeffs={best_coeffs.round(4)}" if n_coeffs > 0 else ""
            tqdm.write(f"Epoch {epoch}: data={loss_data.item():.2e}, pde={loss_pde.item():.2e}, total={loss.item():.2e} {coeff_str}")
    
    # Final evaluation
    elapsed = time.time() - start_time
    
    # Use mse_func if provided for consistent comparison
    if mse_func is not None and n_coeffs > 0:
        try:
            final_loss = mse_func(best_coeffs)
        except:
            final_loss = min_loss
    else:
        final_loss = min_loss
    
    print(f"PINN: Completed in {elapsed:.2f}s")
    print(f"PINN: Best coefficients: {best_coeffs}")
    print(f"PINN: Final loss: {final_loss:.4e}")
    
    return {
        "x": best_coeffs,
        "fun": final_loss,
        "pinn_loss": min_loss,
        "nit": epochs,
        "status": "Success",
        "time": elapsed
    }
