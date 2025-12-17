"""
Test hippylib NS inverse solver with synthetic data.

Generate synthetic data from known viscosity field, then try to recover it.
"""

import numpy as np
import dolfin as dl
import sys
sys.path.append('pix')
from methods.hippylib_ns_inverse import run_solver_hippylib

def generate_synthetic_ns_data(nx=50, ny=50, mu_true_func=None):
    """
    Generate synthetic NS data with known viscosity field.
    
    Returns:
    --------
    u, v, p : (nx, ny, 1) arrays - velocity and pressure
    mu_true : (nx, ny) array - true viscosity field
    """
    print("[Synthetic] Generating NS data with known viscosity...")
    
    # Create mesh
    mesh = dl.UnitSquareMesh(32, 32)
    
    # Function spaces (Taylor-Hood P2-P1)
    P2 = dl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = dl.FunctionSpace(mesh, TH)
    
    # True viscosity field (spatially varying)
    if mu_true_func is None:
        # Simple test: constant + sinusoidal variation
        import ufl
        V_scalar = dl.FunctionSpace(mesh, "CG", 1)
        mu_true = dl.Function(V_scalar)
        
        class MuExpression(dl.UserExpression):
            def eval(self, values, x):
                # mu = 0.1 + 0.05*sin(2πx)*sin(2πy)
                values[0] = 0.1 + 0.05 * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
            def value_shape(self):
                return ()
        
        mu_expr = MuExpression(degree=2)
        mu_true = dl.interpolate(mu_expr, V_scalar)
    else:
        mu_true = mu_true_func
    
    print(f"[Synthetic] True μ range: [{mu_true.vector().min():.4f}, {mu_true.vector().max():.4f}]")
    
    # Solve forward NS with true viscosity
    # Using driven cavity flow: u=1 on top, u=0 elsewhere
    
    # Boundary conditions
    def top_boundary(x, on_boundary):
        return on_boundary and dl.near(x[1], 1.0)
    
    def other_boundaries(x, on_boundary):
        return on_boundary and not dl.near(x[1], 1.0)
    
    bc_top = dl.DirichletBC(W.sub(0), dl.Constant((1.0, 0.0)), top_boundary)
    bc_other = dl.DirichletBC(W.sub(0), dl.Constant((0.0, 0.0)), other_boundaries)
    bcs = [bc_top, bc_other]
    
    # Define variational problem
    up = dl.Function(W)
    u, p = dl.split(up)
    v, q = dl.TestFunctions(W)
    
    def epsilon(w):
        return 0.5 * (dl.grad(w) + dl.grad(w).T)
    
    rho = 1.0
    F_body = dl.Constant((0.0, 0.0))
    
    # Steady Stokes equations (no convection - for stability)
    # F = (
    #     rho * dl.inner(dl.dot(u, dl.nabla_grad(u)), v) * dl.dx +  # Convection term REMOVED
    #     2.0 * mu_true * dl.inner(epsilon(u), epsilon(v)) * dl.dx -
    #     p * dl.div(v) * dl.dx -
    #     dl.inner(F_body, v) * dl.dx +
    #     dl.div(u) * q * dl.dx
    # )
    
    # Use Stokes instead of full NS for stability
    F = (
        2.0 * mu_true * dl.inner(epsilon(u), epsilon(v)) * dl.dx -
        p * dl.div(v) * dl.dx -
        dl.inner(F_body, v) * dl.dx +
        dl.div(u) * q * dl.dx
    )
    
    # Solve
    print("[Synthetic] Solving forward NS problem...")
    try:
        dl.solve(F == 0, up, bcs,
                solver_parameters={
                    "newton_solver": {
                        "maximum_iterations": 50,
                        "relative_tolerance": 1e-5,
                        "absolute_tolerance": 1e-5,
                        "relaxation_parameter": 0.8
                    }
                })
        print("[Synthetic] ✓ Forward solve successful")
    except Exception as e:
        print(f"[Synthetic] ✗ Forward solve failed: {e}")
        return None, None, None, None
    
    u_sol, p_sol = up.split(deepcopy=True)
    
    # Interpolate to regular grid
    from scipy.interpolate import griddata
    
    # Get mesh coordinates
    coords = mesh.coordinates()
    
    # Evaluate solution at mesh vertices
    u_vals = u_sol.compute_vertex_values(mesh).reshape(2, -1).T  # (nvertices, 2)
    p_vals = p_sol.compute_vertex_values(mesh)  # (nvertices,)
    
    # Create regular grid
    x_grid = np.linspace(0, 1, nx)
    y_grid = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Interpolate to grid
    u_grid = griddata(coords, u_vals[:, 0], (X, Y), method='linear', fill_value=0.0)
    v_grid = griddata(coords, u_vals[:, 1], (X, Y), method='linear', fill_value=0.0)
    p_grid = griddata(coords, p_vals, (X, Y), method='linear', fill_value=0.0)
    
    # Get true mu on grid
    mu_vals = mu_true.compute_vertex_values(mesh)
    mu_grid = griddata(coords, mu_vals, (X, Y), method='linear', fill_value=0.1)
    
    # Reshape to (nx, ny, 1) for compatibility
    u_data = u_grid[:, :, np.newaxis]
    v_data = v_grid[:, :, np.newaxis]
    p_data = p_grid[:, :, np.newaxis]
    
    print(f"[Synthetic] Data shapes: u={u_data.shape}, v={v_data.shape}, p={p_data.shape}")
    print(f"[Synthetic] Velocity range: u∈[{u_data.min():.3f}, {u_data.max():.3f}], v∈[{v_data.min():.3f}, {v_data.max():.3f}]")
    print(f"[Synthetic] Pressure range: p∈[{p_data.min():.3f}, {p_data.max():.3f}]")
    
    return u_data, v_data, p_data, mu_grid


def test_inverse_problem():
    """Test inverse problem with synthetic data."""
    
    print("="*80)
    print("Testing hippylib NS Inverse Problem with Synthetic Data")
    print("="*80)
    
    # Generate synthetic data
    u_obs, v_obs, p_obs, mu_true = generate_synthetic_ns_data(nx=50, ny=50)
    
    if u_obs is None:
        print("✗ Failed to generate synthetic data")
        return
    
    print(f"\n[Test] True viscosity: μ ∈ [{mu_true.min():.4f}, {mu_true.max():.4f}], mean={mu_true.mean():.4f}")
    
    # Add small noise
    noise_level = 0.01
    u_obs += noise_level * np.random.randn(*u_obs.shape)
    v_obs += noise_level * np.random.randn(*v_obs.shape)
    p_obs += noise_level * np.random.randn(*p_obs.shape)
    
    # Prepare args for run_solver_hippylib
    args = [u_obs] + [np.zeros_like(u_obs)]*7 + \
           [v_obs] + [np.zeros_like(v_obs)]*7 + \
           [p_obs] + [np.zeros_like(p_obs)]*7 + \
           [np.ones_like(u_obs)]
    
    # Initial guess (constant viscosity - intentionally wrong)
    nx, ny, nt = u_obs.shape
    mu_init = 0.05 * np.ones((nx, ny, nt))  # Underestimate
    
    print(f"\n[Test] Running inverse solver...")
    print(f"[Test] Initial guess: μ=0.05 (intentionally low)")
    print(f"[Test] True mean: μ={mu_true.mean():.4f}")
    
    # Run inverse solver
    mu_recovered = run_solver_hippylib(
        args,
        t_min=0,
        t_max=0,
        mu_init=mu_init,
        gamma=1e-4,
        maxiter=10,
        gtol=1e-4,
        verbose=True,
        Fx=0.0,
        Fy=0.0
    )
    
    # Extract result (only first timestep, first component)
    mu_result = mu_recovered[:, :, 0, 0]
    
    print(f"\n[Test] Results:")
    print(f"  True μ:      [{mu_true.min():.4f}, {mu_true.max():.4f}], mean={mu_true.mean():.4f}")
    print(f"  Recovered μ: [{mu_result.min():.4f}, {mu_result.max():.4f}], mean={mu_result.mean():.4f}")
    
    # Compute error
    rel_error = np.linalg.norm(mu_result - mu_true) / np.linalg.norm(mu_true)
    print(f"  Relative L2 error: {rel_error:.4f}")
    
    # Visualize
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        vmin, vmax = mu_true.min(), mu_true.max()
        
        im1 = axes[0].imshow(mu_true.T, origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title('True μ')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(mu_result.T, origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title('Recovered μ')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow((mu_result - mu_true).T, origin='lower', cmap='RdBu_r')
        axes[2].set_title(f'Error (rel={rel_error:.3f})')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig('hippylib_inverse_test.png', dpi=150)
        print(f"\n[Test] Plot saved to: hippylib_inverse_test.png")
        
    except ImportError:
        print("\n[Test] matplotlib not available, skipping visualization")
    
    # Check if we improved over initial guess
    error_init = np.linalg.norm(0.1 - mu_true) / np.linalg.norm(mu_true)
    print(f"\n[Test] Comparison:")
    print(f"  Initial guess error: {error_init:.4f}")
    print(f"  Final error:         {rel_error:.4f}")
    
    if rel_error < error_init:
        print(f"  ✓ Improvement: {(1 - rel_error/error_init)*100:.1f}%")
    else:
        print(f"  ✗ No improvement (optimization failed)")
    
    return rel_error < error_init


if __name__ == "__main__":
    success = test_inverse_problem()
    exit(0 if success else 1)
