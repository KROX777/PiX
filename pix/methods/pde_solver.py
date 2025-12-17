"""
A · ∇μ + μb = Q

A: Matrix field (2x2)
b: 2D Vector field
Q: 2D Vector field

Definitions aligned with diagnose_mu_constant and raw args (no re-differentiation):
    - A = 2S, where S_xx = u_x, S_xy = (u_y + v_x)/2, S_yy = v_y
    - b = ∇·(2S) with components using provided second derivatives directly:
                b_x = 2*u_xx + u_yy + v_xy
                b_y = u_xy + v_xx + 2*v_yy
    - Q = rho*(du/dt + (u·∇)u + u (∇·u)) + ∇p + F, component-wise

Args index mapping (from Calculator/DataLoader args_data; order per var = [var, x, y, xx, yx, xy, yy, t]):
        For u block (start=0):
            0:u, 1:u_x, 2:u_y, 3:u_xx, 4:u_yx, 5:u_xy, 6:u_yy, 7:u_t
        For v block (start=8):
            8:v, 9:v_x, 10:v_y, 11:v_xx, 12:v_yx, 13:v_xy, 14:v_yy, 15:v_t
        For rho block (start=16):
            16:rho, 17:rho_x, 18:rho_y, 19:rho_xx, 20:rho_yx, 21:rho_xy, 22:rho_yy, 23:rho_t
        For p block (start=24):
            24:p, 25:p_x, 26:p_y, 27:p_xx, 28:p_yx, 29:p_xy, 30:p_yy, 31:p_t
"""
import numpy as np
import logging
import time
from typing import Optional

from scipy.sparse import lil_matrix, csr_matrix, eye as sparse_eye, vstack as sparse_vstack, block_diag, hstack
from scipy.sparse import diags as sparse_diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import lsq_linear
import warnings
from ..utils.finite_diff import FiniteDiffVand

# Deprecated
# def solve_pde(A, b, Q, dx, dy, dt=None, mu_x0=None, mu_y0=None, mu_xN=None, mu_yN=None, t_min=0, t_max=5, regularize_eps=0.0):
#     """
#     Solve the steady-state PDE: A · ∇μ + μb = Q using finite differences.
    
#     Parameters:
#     - A: (nx, ny, nt, 2, 2) matrix field
#     - b: (nx, ny, nt, 2) vector field
#     - Q: (nx, ny, nt, 2) source term
#     - dx, dy: spatial step sizes
#     - dt: ignored (kept for compatibility)
#     - mu_x0: (ny, nt, 2) boundary values at x=0
#     - mu_y0: (nx, nt, 2) boundary values at y=0
#     - mu_xN: (ny, nt, 2) boundary values at x=nx-1
#     - mu_yN: (nx, nt, 2) boundary values at y=ny-1
#     - t_min, t_max: inclusive time index range to solve. Values are clamped to [0, nt-1].
#                     If t_min > t_max after clamping, a ValueError is raised.
#     - regularize_eps: small diagonal regularization added to the operator (>=0)
    
#     Returns:
#     - mu: (nx, ny, nt, 2) solution field
#     """
#     nx, ny, nt, _ = Q.shape
#     # Clamp t range
#     if t_min is None:
#         t_min = 0
#     if t_max is None:
#         t_max = nt - 1
#     t_min = max(0, int(t_min))
#     t_max = min(nt - 1, int(t_max))
#     if t_min > t_max:
#         raise ValueError(f"Invalid t range: t_min={t_min} > t_max={t_max}")
#     mu = np.zeros((nx, ny, nt, 2), dtype=np.float64)

#     # For each time step (subset for speed)
#     for t in range(t_min, t_max + 1):
#         # Initialize solution array for current time step
#         mu_t = np.zeros((nx, ny, 2), dtype=np.float64)
        
#         # Solve for each component (x and y)
#         for comp in [0, 1]:  # x (0) and y (1) components
#             # Construct finite difference operator and right-hand side
#             L, rhs = construct_operator(
#                 A[..., t, :, :], 
#                 b[..., t, :], 
#                 Q[..., t, comp], 
#                 dx, dy, 
#                 comp,
#                 mu_x0[:, t, comp] if mu_x0 is not None else None,
#                 mu_y0[:, t, comp] if mu_y0 is not None else None,
#                 mu_xN[:, t, comp] if mu_xN is not None else None,
#                 mu_yN[:, t, comp] if mu_yN is not None else None
#             )
            
#             # Convert to CSR format for efficient solving
#             L_csr = L.tocsr()

#             # Optional regularization
#             if regularize_eps and regularize_eps > 0.0:
#                 L_csr = L_csr + regularize_eps * sparse_eye(L_csr.shape[0], format="csr")

#             # Fast structural check: any zero rows means underdetermined at those nodes
#             row_nnz = np.diff(L_csr.indptr)
#             has_zero_rows = np.any(row_nnz == 0)

#             try:
#                 if has_zero_rows:
#                     raise RuntimeError("operator has structurally zero rows")
#                 mu_flat = spsolve(L_csr, rhs)
#                 # Guard against NaNs/Infs
#                 if not np.all(np.isfinite(mu_flat)):
#                     raise RuntimeError("non-finite solution from spsolve")
#             except Exception as e:
#                 # Retry with a small diagonal regularization
#                 try:
#                     reg_eps = max(regularize_eps, 1e-8)
#                     L_reg = L_csr + reg_eps * sparse_eye(L_csr.shape[0], format="csr")
#                     mu_flat = spsolve(L_reg, rhs)
#                     if not np.all(np.isfinite(mu_flat)):
#                         raise RuntimeError("non-finite solution after regularization")
#                     warnings.warn(f"Solved with regularization eps={reg_eps} at t={t}, comp={comp}: {e}")
#                 except Exception as e2:
#                     warnings.warn(f"Solver failed at time step {t}, component {comp}: {str(e2)}. Using zero solution.")
#                     mu_flat = np.zeros(rhs.shape)
            
#             mu_t[..., comp] = mu_flat.reshape((nx, ny))
        
#         mu[..., t, :] = mu_t
    
#     return mu

def solve_pde_scalar_ls(
    A,
    b,
    Q,
    dx,
    dy,
    dt=None,
    mu_x0=None,
    mu_y0=None,
    mu_xN=None,
    mu_yN=None,
    t_min: int = 0,
    t_max: Optional[int] = None,
    reg_ls: float = 1e-8,
    rho: Optional[np.ndarray] = None,
    estimate_g_global: bool = False,
    reg_g: float = 1e-8,
    g_bounds: Optional[tuple] = None,
):
    """
    Solve for a single scalar μ that best fits BOTH momentum components in a
    least-squares sense at each time step:
        minimize ||L0 μ - Q0||^2 + ||L1 μ - Q1||^2
    which leads to normal equations (L^T L + λI) μ = L^T rhs.

    Returns μ replicated across 2 components with shape (nx, ny, nt, 2) for
    downstream compatibility.
    """
    nx, ny, nt, _ = Q.shape

    # Clamp t range
    if t_min is None:
        t_min = 0
    if t_max is None:
        t_max = nt - 1
    t_min = max(0, int(t_min))
    t_max = min(nt - 1, int(t_max))
    if t_min > t_max:
        raise ValueError(f"Invalid t range: t_min={t_min} > t_max={t_max}")

    mu_out = np.zeros((nx, ny, nt, 2), dtype=np.float64)

    # If user requests a global g estimate across time, assemble a global
    # augmented linear system where g is a single scalar unknown shared across
    # all solved time slices. Otherwise solve per-time independently (existing behavior).
    times = list(range(t_min, t_max + 1))
    if estimate_g_global:
        if rho is None:
            raise ValueError("rho must be provided to estimate global g")

        # Collect per-time blocks
        L_blocks = []
        rhs_blocks = []
        rho_cols = []
        for t in times:
            L0, rhs0 = construct_operator(
                A[..., t, :, :],
                b[..., t, :],
                Q[..., t, 0],
                dx,
                dy,
                0,
                x0_bc=mu_x0[:, t, 0] if mu_x0 is not None else None,
                y0_bc=mu_y0[:, t, 0] if mu_y0 is not None else None,
                xN_bc=mu_xN[:, t, 0] if mu_xN is not None else None,
                yN_bc=mu_yN[:, t, 0] if mu_yN is not None else None,
                use_4th_boundary=False,
            )
            L1, rhs1 = construct_operator(
                A[..., t, :, :],
                b[..., t, :],
                Q[..., t, 1],
                dx,
                dy,
                1,
                x0_bc=mu_x0[:, t, 1] if mu_x0 is not None else None,
                y0_bc=mu_y0[:, t, 1] if mu_y0 is not None else None,
                xN_bc=mu_xN[:, t, 1] if mu_xN is not None else None,
                yN_bc=mu_yN[:, t, 1] if mu_yN is not None else None,
                use_4th_boundary=False,
            )

            L_stack = sparse_vstack([L0.tocsr(), L1.tocsr()], format="csr")
            rhs_stack = np.concatenate([rhs0, rhs1], axis=0)

            # rho contribution: zeros for comp0 rows, rho_flat for comp1 rows
            rho_flat = rho[..., t].reshape(nx * ny)
            rho_stack = np.concatenate([np.zeros_like(rho_flat), rho_flat], axis=0)

            L_blocks.append(L_stack)
            rhs_blocks.append(rhs_stack)
            rho_cols.append(rho_stack)

        # Block-diagonal assembly for L and concatenation for rhs and rho
        L_block = block_diag(L_blocks, format="csr")
        rhs_total = np.concatenate(rhs_blocks, axis=0)
        rho_total = np.concatenate(rho_cols, axis=0)

        # Augment L with a single column for g
        rho_col = csr_matrix(rho_total).reshape((-1, 1))
        A_aug = hstack([L_block, rho_col], format="csr")

        # Solve bounded least-squares using scipy.optimize.lsq_linear if available.
        # We implement Tikhonov regularization by appending scaled identity rows.
        A_csr = A_aug.tocsr()
        rhs_total = rhs_total

        # Prepare augmentation for regularization (sparse) if requested
        aug_As = [A_csr]
        aug_bs = [rhs_total]
        n_unknowns = A_csr.shape[1]
        mu_unknowns = n_unknowns - 1

        if reg_ls and reg_ls > 0.0:
            sqrt_mu = float(np.sqrt(reg_ls))
            rows_mu = sparse_diags([sqrt_mu] * mu_unknowns, 0, shape=(mu_unknowns, mu_unknowns), format='csr')
            pad = csr_matrix((mu_unknowns, 1), dtype=np.float64)
            regA_mu = hstack([rows_mu, pad], format='csr')
            aug_As.append(regA_mu)
            aug_bs.append(np.zeros(mu_unknowns, dtype=np.float64))

        if reg_g and reg_g > 0.0:
            sqrt_g = float(np.sqrt(reg_g))
            reg_row = csr_matrix((1, n_unknowns), dtype=np.float64)
            reg_row[0, -1] = sqrt_g
            aug_As.append(reg_row)
            aug_bs.append(np.array([0.0], dtype=np.float64))

        A_for_lsq = sparse_vstack(aug_As, format='csr')
        rhs_for_lsq = np.concatenate(aug_bs, axis=0)

        # Build bounds for unknowns: default unbounded, g bounded if provided
        lb = -np.inf * np.ones(n_unknowns, dtype=np.float64)
        ub = np.inf * np.ones(n_unknowns, dtype=np.float64)
        if g_bounds is not None:
            lb[-1] = float(g_bounds[0])
            ub[-1] = float(g_bounds[1])

        x = None
        try:
            # lsq_linear accepts sparse A in recent SciPy versions; use 'trf' method
            res = lsq_linear(A_for_lsq, rhs_for_lsq, bounds=(lb, ub), method='trf', verbose=0)
            if not res.success:
                raise RuntimeError("lsq_linear failed")
            x = res.x
        except Exception:
            # Fallback to normal equations solve (previous behavior)
            Lt = A_csr.transpose().tocsr()
            Normal = Lt @ A_csr
            rhs_normal = Lt @ rhs_total
            # diagonal regularization
            diag_reg = np.ones(n_unknowns, dtype=np.float64) * float(max(reg_ls, 0.0))
            diag_reg[-1] = float(max(reg_g, 0.0))
            if np.any(diag_reg > 0.0):
                Normal = Normal + sparse_diags(diag_reg, 0, format="csr")
            try:
                x = spsolve(Normal, rhs_normal)
            except Exception:
                Normal = Normal + 1e-12 * sparse_eye(Normal.shape[0], format="csr")
                x = spsolve(Normal, rhs_normal)

        # Extract mu and g
        g_est = float(x[-1])
        mu_all = x[:-1]
        # reshape and place into output
        for ti, t in enumerate(times):
            start = ti * (nx * ny)
            stop = start + (nx * ny)
            mu_t_flat = mu_all[start:stop]
            mu_t_scalar = mu_t_flat.reshape((nx, ny))
            mu_out[..., t, 0] = mu_t_scalar
            mu_out[..., t, 1] = mu_t_scalar

        # Return both mu and the estimated global g when requested.
        return mu_out, g_est

    # Fallback: per-time independent solve (original behavior)
    for t in range(t_min, t_max + 1):
        # Build operators for both components sharing the SAME μ unknown vector
        L0, rhs0 = construct_operator(
            A[..., t, :, :],
            b[..., t, :],
            Q[..., t, 0],
            dx,
            dy,
            0,
            x0_bc=mu_x0[:, t, 0] if mu_x0 is not None else None,
            y0_bc=mu_y0[:, t, 0] if mu_y0 is not None else None,
            xN_bc=mu_xN[:, t, 0] if mu_xN is not None else None,
            yN_bc=mu_yN[:, t, 0] if mu_yN is not None else None,
            use_4th_boundary=False,
        )
        L1, rhs1 = construct_operator(
            A[..., t, :, :],
            b[..., t, :],
            Q[..., t, 1],
            dx,
            dy,
            1,
            x0_bc=mu_x0[:, t, 1] if mu_x0 is not None else None,
            y0_bc=mu_y0[:, t, 1] if mu_y0 is not None else None,
            xN_bc=mu_xN[:, t, 1] if mu_xN is not None else None,
            yN_bc=mu_yN[:, t, 1] if mu_yN is not None else None,
            use_4th_boundary=False,
        )

        # Stack the two components: L_stack * μ ≈ rhs_stack
        L0_csr = L0.tocsr()
        L1_csr = L1.tocsr()
        L_stack = sparse_vstack([L0_csr, L1_csr], format="csr")
        rhs_stack = np.concatenate([rhs0, rhs1], axis=0)

        # Normal equations with Tikhonov regularization
        Lt = L_stack.transpose().tocsr()
        Normal = Lt @ L_stack
        if reg_ls and reg_ls > 0.0:
            Normal = Normal + reg_ls * sparse_eye(Normal.shape[0], format="csr")
        rhs_normal = Lt @ rhs_stack

        try:
            mu_flat = spsolve(Normal, rhs_normal)
            if not np.all(np.isfinite(mu_flat)):
                raise RuntimeError("non-finite solution LS")
        except Exception as _:
            # fall back to tiny regularization bump
            mu_flat = spsolve(Normal + 1e-10 * sparse_eye(Normal.shape[0], format="csr"), rhs_normal)

        mu_t_scalar = mu_flat.reshape(nx, ny)
        # replicate across 2 components for compatibility
        mu_out[..., t, 0] = mu_t_scalar
        mu_out[..., t, 1] = mu_t_scalar

    return mu_out

def construct_operator(A, b, Q_comp, dx, dy, component, x0_bc=None, y0_bc=None, xN_bc=None, yN_bc=None, use_4th_boundary: bool = False, conservative: bool = False):
    """
    Construct the sparse finite difference operator L and right-hand side vector
    for the PDE, with proper boundary condition handling.
    
    Parameters:
    - A: (nx, ny, 2, 2) matrix field at a single time step
    - b: (nx, ny, 2) vector field at a single time step
    - Q_comp: (nx, ny) source term for current component
    - dx, dy: spatial step sizes
    - component: 0 (x) or 1 (y) for which equation to solve
    - x0_bc: (ny,) boundary values at x=0 for current component
    - y0_bc: (nx,) boundary values at y=0 for current component
    - xN_bc: (ny,) boundary values at x=nx-1 for current component
    - yN_bc: (nx,) boundary values at y=ny-1 for current component
    
    Returns:
    - L: sparse matrix representing the discretized PDE
    - rhs: right-hand side vector
    """
    nx, ny = A.shape[:2]
    N = nx * ny
    L = lil_matrix((N, N), dtype=np.float64)  # Use LIL for efficient construction
    rhs = np.zeros(N, dtype=np.float64)
    
    # Create boundary mask and values
    bc_mask = np.zeros((nx, ny), dtype=bool)
    bc_values = np.zeros((nx, ny))
    
    # Apply boundary conditions with priority: left > bottom > right > top
    # Left boundary (x=0)
    if x0_bc is not None:
        for j in range(ny):
            bc_mask[0, j] = True
            bc_values[0, j] = x0_bc[j]
    
    # Bottom boundary (y=0)
    if y0_bc is not None:
        for i in range(nx):
            # Skip if already set by left boundary
            if not bc_mask[i, 0]:
                bc_mask[i, 0] = True
                bc_values[i, 0] = y0_bc[i]
    
    # Right boundary (x=nx-1)
    if xN_bc is not None:
        for j in range(ny):
            # Skip if already set by bottom boundary
            if not bc_mask[nx-1, j]:
                bc_mask[nx-1, j] = True
                bc_values[nx-1, j] = xN_bc[j]
    
    # Top boundary (y=ny-1)
    if yN_bc is not None:
        for i in range(nx):
            # Skip if already set by left or right boundary
            if not bc_mask[i, ny-1]:
                bc_mask[i, ny-1] = True
                bc_values[i, ny-1] = yN_bc[i]
    
    # Warn if not all boundaries are defined
    boundary_defined = [
        x0_bc is not None, 
        y0_bc is not None, 
        xN_bc is not None, 
        yN_bc is not None
    ]
    
    if not all(boundary_defined):
        warnings.warn(
            "Incomplete boundary conditions provided. "
            "Unspecified boundaries will use Neumann conditions implicitly.",
            RuntimeWarning
        )
    
    # For all points
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            
            # If boundary point, apply Dirichlet condition
            if bc_mask[i, j]:
                L[idx, idx] = 1.0
                rhs[idx] = bc_values[i, j]
                continue
                
            # For non-boundary points, apply finite differences
            if conservative:
                # Divergence form: r_i = ∂x(ax * μ) + ∂y(ay * μ), where ax=A[i,j,comp,0], ay=A[i,j,comp,1]
                # This implicitly incorporates the μ b term in discrete form and often reduces truncation error.
                ax_c = A[i, j, component, 0]
                ay_c = A[i, j, component, 1]

                # x-direction contribution
                if i == 0:
                    # forward difference: (ax[i+1]*μ[i+1] - ax[i]*μ[i]) / dx
                    L[idx, i * ny + j] += -(ax_c) / dx
                    L[idx, (i+1) * ny + j] += A[i+1, j, component, 0] / dx
                elif i == nx - 1:
                    # backward difference: (ax[i]*μ[i] - ax[i-1]*μ[i-1]) / dx
                    L[idx, (i-1) * ny + j] += -A[i-1, j, component, 0] / dx
                    L[idx, i * ny + j] += ax_c / dx
                else:
                    # central difference: (ax[i+1]*μ[i+1] - ax[i-1]*μ[i-1]) / (2dx)
                    L[idx, (i+1) * ny + j] += A[i+1, j, component, 0] / (2.0 * dx)
                    L[idx, (i-1) * ny + j] += -A[i-1, j, component, 0] / (2.0 * dx)

                # y-direction contribution
                if j == 0:
                    L[idx, i * ny + j] += -(ay_c) / dy
                    L[idx, i * ny + (j+1)] += A[i, j+1, component, 1] / dy
                elif j == ny - 1:
                    L[idx, i * ny + (j-1)] += -A[i, j-1, component, 1] / dy
                    L[idx, i * ny + j] += ay_c / dy
                else:
                    L[idx, i * ny + (j+1)] += A[i, j+1, component, 1] / (2.0 * dy)
                    L[idx, i * ny + (j-1)] += -A[i, j-1, component, 1] / (2.0 * dy)
            else:
                # Non-conservative (A·∇μ + μ b) discretization
                # x-derivative term
                ax = A[i, j, component, 0]
                if use_4th_boundary and i == 0 and nx >= 5:
                    # 4th-order forward at very first column
                    w = (-25.0/12.0, 4.0, -3.0, 4.0/3.0, -0.25)
                    for k, wk in enumerate(w):
                        L[idx, (i+k) * ny + j] += ax * (wk / dx)
                elif use_4th_boundary and i == nx - 1 and nx >= 5:
                    # 4th-order backward at very last column
                    w = (25.0/12.0, -4.0, 3.0, -4.0/3.0, 0.25)
                    for k, wk in enumerate(w):
                        L[idx, (i-k) * ny + j] += ax * (wk / dx)
                else:
                    # Interior: 4th-order central when possible, else 2nd-order central
                    if i >= 2 and i <= nx - 3:
                        c_im2 = 1.0 / (12.0 * dx)
                        c_im1 = -8.0 / (12.0 * dx)
                        c_ip1 = 8.0 / (12.0 * dx)
                        c_ip2 = -1.0 / (12.0 * dx)
                        L[idx, (i-2) * ny + j] += ax * c_im2
                        L[idx, (i-1) * ny + j] += ax * c_im1
                        L[idx, (i+1) * ny + j] += ax * c_ip1
                        L[idx, (i+2) * ny + j] += ax * c_ip2
                    else:
                        # Edges: stable 2nd-order one-sided
                        if i == 0 and nx >= 3:
                            c0 = -3.0 / (2.0 * dx)
                            c1 = 4.0 / (2.0 * dx)
                            c2 = -1.0 / (2.0 * dx)
                            L[idx, idx] += ax * c0
                            L[idx, (i+1) * ny + j] += ax * c1
                            L[idx, (i+2) * ny + j] += ax * c2
                        elif i == nx - 1 and nx >= 3:
                            c0 = 3.0 / (2.0 * dx)
                            c1 = -4.0 / (2.0 * dx)
                            c2 = 1.0 / (2.0 * dx)
                            L[idx, idx] += ax * c0
                            L[idx, (i-1) * ny + j] += ax * c1
                            L[idx, (i-2) * ny + j] += ax * c2
                        else:
                            c_plus = 1.0 / (2.0 * dx)
                            c_minus = -1.0 / (2.0 * dx)
                            L[idx, (i+1) * ny + j] += ax * c_plus
                            L[idx, (i-1) * ny + j] += ax * c_minus

                # y-derivative term
                ay = A[i, j, component, 1]
                if use_4th_boundary and j == 0 and ny >= 5:
                    w = (-25.0/12.0, 4.0, -3.0, 4.0/3.0, -0.25)
                    for k, wk in enumerate(w):
                        L[idx, i * ny + (j+k)] += ay * (wk / dy)
                elif use_4th_boundary and j == ny - 1 and ny >= 5:
                    w = (25.0/12.0, -4.0, 3.0, -4.0/3.0, 0.25)
                    for k, wk in enumerate(w):
                        L[idx, i * ny + (j-k)] += ay * (wk / dy)
                else:
                    if j >= 2 and j <= ny - 3:
                        c_jm2 = 1.0 / (12.0 * dy)
                        c_jm1 = -8.0 / (12.0 * dy)
                        c_jp1 = 8.0 / (12.0 * dy)
                        c_jp2 = -1.0 / (12.0 * dy)
                        L[idx, i * ny + (j-2)] += ay * c_jm2
                        L[idx, i * ny + (j-1)] += ay * c_jm1
                        L[idx, i * ny + (j+1)] += ay * c_jp1
                        L[idx, i * ny + (j+2)] += ay * c_jp2
                    else:
                        if j == 0 and ny >= 3:
                            c0 = -3.0 / (2.0 * dy)
                            c1 = 4.0 / (2.0 * dy)
                            c2 = -1.0 / (2.0 * dy)
                            L[idx, idx] += ay * c0
                            L[idx, i * ny + (j+1)] += ay * c1
                            L[idx, i * ny + (j+2)] += ay * c2
                        elif j == ny - 1 and ny >= 3:
                            c0 = 3.0 / (2.0 * dy)
                            c1 = -4.0 / (2.0 * dy)
                            c2 = 1.0 / (2.0 * dy)
                            L[idx, idx] += ay * c0
                            L[idx, i * ny + (j-1)] += ay * c1
                            L[idx, i * ny + (j-2)] += ay * c2
                        else:
                            c_plus = 1.0 / (2.0 * dy)
                            c_minus = -1.0 / (2.0 * dy)
                            L[idx, i * ny + (j+1)] += ay * c_plus
                            L[idx, i * ny + (j-1)] += ay * c_minus

                # Diagonal term (μb contribution)
                L[idx, idx] += b[i, j, component]
            
            # Right-hand side
            rhs[idx] = Q_comp[i, j]
    
    return L, rhs

def is_invertible(A, tol=1e-12):
    """
    Deprecated expensive check. Kept for backward-compatibility in case of external calls.
    Now replaced by a cheap structural check and solver try/except in solve_pde.
    """
    # Avoid expensive SVD on large systems; just check there are no empty rows.
    A = A.tocsr()
    row_nnz = np.diff(A.indptr)
    return not np.any(row_nnz == 0)

def compute_pde_residual_loss(
    mu: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    Q: np.ndarray,
    dx: float,
    dy: float,
    t_min: int = 0,
    t_max: int = None,
    mu_x0: Optional[np.ndarray] = None,
    mu_y0: Optional[np.ndarray] = None,
    mu_xN: Optional[np.ndarray] = None,
    mu_yN: Optional[np.ndarray] = None,
    n_clip: int = 0,
    use_mu_as_bc: bool = True,
):
    """
    Compute PDE residual loss by plugging solved mu into the same discretized operator:
        r = (A · ∇mu + mu b) - Q  ≈  L(mu) - rhs

    Returns a dict with aggregated metrics across the selected time window and per-component breakdown.
    """
    nx, ny, nt, _ = Q.shape
    if t_max is None:
        t_max = nt - 1
    t_min = max(0, int(t_min))
    t_max = min(nt - 1, int(t_max))
    if t_min > t_max:
        raise ValueError(f"Invalid t range for residual: t_min={t_min} > t_max={t_max}")

    # Optional belt mask to exclude boundaries (align with SON n_clip)
    belt = max(0, int(n_clip))
    mask2d = None
    if belt > 0 and belt * 2 < nx and belt * 2 < ny:
        mask2d = np.zeros((nx, ny), dtype=bool)
        mask2d[belt:nx - belt, belt:ny - belt] = True

    totals = {"all": {"sum_sq": 0.0, "count": 0, "max_abs": 0.0}, 0: {"sum_sq": 0.0, "count": 0, "max_abs": 0.0}, 1: {"sum_sq": 0.0, "count": 0, "max_abs": 0.0}}

    for t in range(t_min, t_max + 1):
        # Prepare boundary slices for time t if provided
        x0_t = mu_x0[:, t, :] if mu_x0 is not None else None
        y0_t = mu_y0[:, t, :] if mu_y0 is not None else None
        xN_t = mu_xN[:, t, :] if mu_xN is not None else None
        yN_t = mu_yN[:, t, :] if mu_yN is not None else None

        # Optionally use the provided mu as Dirichlet boundary values for residual evaluation
        if use_mu_as_bc:
            # mu is (nx, ny, nt, 2); use comp 0 for scalar bc values consistently
            mu_t0 = mu[:, :, t, 0]
            if x0_t is None:
                x0_t = np.stack([mu_t0[0, :], mu_t0[0, :]], axis=-1)
            if xN_t is None:
                xN_t = np.stack([mu_t0[-1, :], mu_t0[-1, :]], axis=-1)
            if y0_t is None:
                y0_t = np.stack([mu_t0[:, 0], mu_t0[:, 0]], axis=-1)
            if yN_t is None:
                yN_t = np.stack([mu_t0[:, -1], mu_t0[:, -1]], axis=-1)

        for comp in (0, 1):
            # Build operator consistent with solver
            L, rhs = construct_operator(
                A[..., t, :, :],
                b[..., t, :],
                Q[..., t, comp],
                dx,
                dy,
                comp,
                x0_bc=x0_t[:, comp] if x0_t is not None else None,
                y0_bc=y0_t[:, comp] if y0_t is not None else None,
                xN_bc=xN_t[:, comp] if xN_t is not None else None,
                yN_bc=yN_t[:, comp] if yN_t is not None else None,
                use_4th_boundary=False,
                conservative=True,
            )

            mu_flat = mu[:, :, t, comp].reshape(nx * ny)
            r_full = L.tocsr().dot(mu_flat) - rhs  # residual vector
            if mask2d is not None:
                r = r_full.reshape(nx, ny)[mask2d].ravel()
            else:
                r = r_full

            # Update stats
            sum_sq = float(np.dot(r, r))
            count = r.size
            max_abs = float(np.max(np.abs(r))) if count > 0 else 0.0

            totals[comp]["sum_sq"] += sum_sq
            totals[comp]["count"] += count
            totals[comp]["max_abs"] = max(totals[comp]["max_abs"], max_abs)

            totals["all"]["sum_sq"] += sum_sq
            totals["all"]["count"] += count
            totals["all"]["max_abs"] = max(totals["all"]["max_abs"], max_abs)

    # Aggregate
    def finalize(stats):
        mse = stats["sum_sq"] / max(stats["count"], 1)
        l2 = float(np.sqrt(stats["sum_sq"]))
        return {"mse": float(mse), "l2": l2, "max_abs": float(stats["max_abs"]) }

    return {
        "all": finalize(totals["all"]),
        "comp0": finalize(totals[0]),
        "comp1": finalize(totals[1]),
        "t_range": (int(t_min), int(t_max)),
    }

def compute_pde_residual_pointwise(
    mu_scalar: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    Q: np.ndarray,
    dx: float,
    dy: float,
    t_min: int = 0,
    t_max: Optional[int] = None,
    n_clip: int = 0,
):
    """
    Compute residual r = A·∇mu + mu b − Q pointwise by differentiating mu with
    FiniteDiffVand (same routine used to build args), to reduce discrepancies
    due to the assembled sparse operator stencils.

    mu_scalar: (nx, ny, nt)
    A: (nx, ny, nt, 2, 2), b: (nx, ny, nt, 2), Q: (nx, ny, nt, 2)
    """
    if FiniteDiffVand is None:
        raise RuntimeError("FiniteDiffVand not available; cannot compute pointwise residual")

    nx, ny, nt = mu_scalar.shape
    if t_max is None:
        t_max = nt - 1
    t_min = max(0, int(t_min))
    t_max = min(nt - 1, int(t_max))
    if t_min > t_max:
        raise ValueError(f"Invalid t range for residual: t_min={t_min} > t_max={t_max}")

    # interior mask similar to SON cropping
    belt = max(0, int(n_clip))
    if belt * 2 >= nx or belt * 2 >= ny:
        belt = 0
    if belt > 0:
        mask2d = np.zeros((nx, ny), dtype=bool)
        mask2d[belt:nx - belt, belt:ny - belt] = True
    else:
        mask2d = None

    totals = {"all": {"sum_sq": 0.0, "count": 0, "max_abs": 0.0}, 0: {"sum_sq": 0.0, "count": 0, "max_abs": 0.0}, 1: {"sum_sq": 0.0, "count": 0, "max_abs": 0.0}}

    for t in range(t_min, t_max + 1):
        mu_t = mu_scalar[..., t]
        dmu_dx = FiniteDiffVand(mu_t, dx=dx, d=1, axis=0)
        dmu_dy = FiniteDiffVand(mu_t, dx=dy, d=1, axis=1)
        for comp in (0, 1):
            r = (
                A[..., t, comp, 0] * dmu_dx
                + A[..., t, comp, 1] * dmu_dy
                + mu_t * b[..., t, comp]
                - Q[..., t, comp]
            )
            if mask2d is not None:
                r = r[mask2d]
            r = r.ravel()
            sum_sq = float(np.dot(r, r))
            count = r.size
            max_abs = float(np.max(np.abs(r))) if count > 0 else 0.0
            totals[comp]["sum_sq"] += sum_sq
            totals[comp]["count"] += count
            totals[comp]["max_abs"] = max(totals[comp]["max_abs"], max_abs)
            totals["all"]["sum_sq"] += sum_sq
            totals["all"]["count"] += count
            totals["all"]["max_abs"] = max(totals["all"]["max_abs"], max_abs)

    def finalize(stats):
        mse = stats["sum_sq"] / max(stats["count"], 1)
        l2 = float(np.sqrt(stats["sum_sq"]))
        return {"mse": float(mse), "l2": l2, "max_abs": float(stats["max_abs"]) }

    return {
        "all": finalize(totals["all"]),
        "comp0": finalize(totals[0]),
        "comp1": finalize(totals[1]),
        "t_range": (int(t_min), int(t_max)),
    }

def assemble_A_b_Q_from_args(args):
    """Assemble (A, b, Q) directly from args with the same formulas as diagnose_mu_constant.

    Returns:
        A: (nx, ny, nt, 2, 2), b: (nx, ny, nt, 2), Q: (nx, ny, nt, 2)
    """
    # First-order fields (per-var block = [var, x, y, xx, yx, xy, yy, t])
    # u block
    u   = args[0]
    u_x = args[1]
    u_y = args[2]
    u_xx = args[3]
    u_yx = args[4]
    u_xy = args[5]
    u_yy = args[6]
    u_t  = args[7]

    # v block
    v   = args[8]
    v_x = args[9]
    v_y = args[10]
    v_xx = args[11]
    v_yx = args[12]
    v_xy = args[13]
    v_yy = args[14]
    v_t  = args[15]

    # rho block
    rho   = args[16]
    rho_x = args[17]
    rho_y = args[18]
    rho_xx = args[19]
    rho_yx = args[20]
    rho_xy = args[21]
    rho_yy = args[22]
    rho_t  = args[23]

    # p block
    p   = args[24]
    p_x = args[25]
    p_y = args[26]
    p_xx = args[27]
    p_yx = args[28]
    p_xy = args[29]
    p_yy = args[30]
    p_t  = args[31]

    nx, ny, nt = u.shape

    # Strain-rate tensor S and divergence
    S_xx = u_x
    S_xy = 0.5 * (u_y + v_x)
    S_yy = v_y
    div_u = u_x + v_y

    # A = 2S
    A = np.empty((nx, ny, nt, 2, 2), dtype=np.float64)
    A[..., 0, 0] = 2.0 * S_xx
    A[..., 0, 1] = 2.0 * S_xy
    A[..., 1, 0] = 2.0 * S_xy
    A[..., 1, 1] = 2.0 * S_yy

    # b from direct second derivatives
    b = np.empty((nx, ny, nt, 2), dtype=np.float64)
    b[..., 0] = 2.0 * u_xx + u_yy + v_xy
    b[..., 1] = u_xy + v_xx + 2.0 * v_yy

    # Q = rho*(du/dt + (u·∇)u + u ∇·u) + ∇p
    Q = np.empty((nx, ny, nt, 2), dtype=np.float64)
    Q[..., 0] = rho * (u_t + u * u_x + v * u_y + u * div_u) + p_x
    Q[..., 1] = rho * (v_t + u * v_x + v * v_y + v * div_u) + p_y

    return A, b, Q


def run_solver(
    args,
    t_min: int = 0,
    t_max: Optional[int] = None,
    dx: float = 0.0025,
    dy: float = 0.0025,
    dt: float = 0.004,
    mu_x0: np.ndarray = None,
    mu_y0: np.ndarray = None,
    mu_xN: np.ndarray = None,
    mu_yN: np.ndarray = None,
    regularize_eps: float = 1e-8,
    gt: np.ndarray = None,
    solve_mode: str = "ls",  # "ls" (joint least squares) or "compwise"
    verbose: bool = False,
    estimate_g_global: bool = False
):
    """Assemble A, b, Q from the provided fields in `args`.

    Expects the same args layout used elsewhere in this repo (SON style):
    Mapping aligned with diagnose_mu_constant:
      0:u, 1:v,
      2:u_x, 3:v_x, 4:u_y, 5:v_y,
      6:u_xx,7:v_xx,10:u_xy,11:v_xy,12:u_yy,13:v_yy,
      14:u_t,15:v_t,
      16:p_x,17:p_y,
      19:rho.

    Returns:
        mu: array shape (nx, ny, nt, 2) solution of A·∇μ + μb = Q.
    """
    logger = logging.getLogger('sr4mdl.pde_solver')
    logger.info("[pde_solver] Starting PDE solving")
    t0 = time.perf_counter()
    # Assemble fields using the diagnose-consistent helper
    A, b, Q = assemble_A_b_Q_from_args(args)
    t1 = time.perf_counter()

    # infer grid sizes from Q shape
    nx, ny, nt, _ = Q.shape

    # Resolve t range locally (solve_pde will clamp and validate again)
    if t_max is None:
        t_max = nt - 1

    # logger.info(
    #     "[pde_solver] Assembled A,b,Q with shape A=%s, b=%s, Q=%s; grid=(%d,%d,%d); t_range=[%d,%d]; mode=%s; eps=%.1e; dt=%.4f",
    #     tuple(A.shape), tuple(b.shape), tuple(Q.shape), nx, ny, nt, int(t_min), int(t_max), solve_mode, regularize_eps, dt
    # )
    # logger.info("[pde_solver] Assembly time: %.3f s", (t1 - t0))

    # Solve requested time slices
    t2 = time.perf_counter()
    if solve_mode == "ls":
        # Extract rho from args (args layout: rho at index 16)
        rho_arg = args[16]

        mu_res = solve_pde_scalar_ls(
            A, b, Q,
            dx=dx, dy=dy, dt=dt,
            mu_x0=mu_x0, mu_y0=mu_y0, mu_xN=mu_xN, mu_yN=mu_yN,
            t_min=t_min,
            t_max=t_max,
            reg_ls=max(regularize_eps, 1e-8),
            rho=rho_arg,
            estimate_g_global=estimate_g_global,
            reg_g=max(regularize_eps, 1e-12),
        )

        # solve_pde_scalar_ls returns (mu, g) when estimate_g_global=True
        if isinstance(mu_res, tuple) and len(mu_res) == 2:
            mu, g_est = mu_res
            logger.info("[pde_solver] Estimated global g = %g", float(g_est))
        else:
            mu = mu_res
    else:
        raise NotImplementedError("Unsupported solve_mode '%s'" % solve_mode)
        # mu = solve_pde(
        #     A, b, Q,
        #     dx=dx, dy=dy, dt=dt,
        #     mu_x0=mu_x0, mu_y0=mu_y0, mu_xN=mu_xN, mu_yN=mu_yN,
        #     t_min=t_min,
        #     t_max=t_max,
        #     regularize_eps=regularize_eps,
        # )
    t3 = time.perf_counter()
    logger.info("[pde_solver] Solve time: %.3f s (mode=%s)", (t3 - t2), solve_mode)

    if verbose:
        # Diagnostics only over solved time window
        sl = slice(max(0, t_min), min(nt - 1, t_max) + 1)
        mu_sub = mu[:, :, sl, 0]
        gt_sub = gt[:, :, sl] if gt is not None else None
        if gt_sub is not None:
            l2_abs = float(np.linalg.norm((mu_sub - gt_sub).ravel()))
            denom = float(np.linalg.norm(gt_sub.ravel()) + 1e-12)
            l2_rel = float(l2_abs / denom)
        else:
            l2_abs = float(np.linalg.norm(mu_sub.ravel()))
            l2_rel = float('nan')
        min_v = float(np.min(mu_sub))
        max_v = float(np.max(mu_sub))
        mean_v = float(np.mean(mu_sub))

        msg = (
            f"[solver3] t in [{t_min},{t_max}] -> mu(comp0) L2 abs={l2_abs:.3e}, rel={l2_rel:.3e}, "
            f"min={min_v:.3e}, max={max_v:.3e}, mean={mean_v:.3e}; shape={mu.shape}"
        )
        print(msg)
        logger.info(msg)

    # Matrix-operator residuals for solved mu
    if verbose:
        res_mu = compute_pde_residual_loss(
            mu=mu,
            A=A,
            b=b,
            Q=Q,
            dx=dx,
            dy=dy,
            t_min=t_min,
            t_max=t_max,
            mu_x0=mu_x0,
            mu_y0=mu_y0,
            mu_xN=mu_xN,
            mu_yN=mu_yN,
            n_clip=5,
        )
        msg = (
            f"[solver3] Residual (operator) using solved mu: ALL mse={res_mu['all']['mse']:.3e}, l2={res_mu['all']['l2']:.3e}, max|r|={res_mu['all']['max_abs']:.3e}; "
            f"comp0 mse={res_mu['comp0']['mse']:.3e}, comp1 mse={res_mu['comp1']['mse']:.3e}"
        )
        print(msg)
        logger.info(msg)

    # Also report residuals when plugging GT
    if verbose and gt is not None:
        res_gt = compute_pde_residual_loss(
            mu=np.stack([gt, gt], axis=-1),
            A=A,
            b=b,
            Q=Q,
            dx=dx,
            dy=dy,
            t_min=t_min,
            t_max=t_max,
            mu_x0=mu_x0,
            mu_y0=mu_y0,
            mu_xN=mu_xN,
            mu_yN=mu_yN,
            n_clip=5,
        )
        msg = (
            f"[solver3] Residual (operator) using GT:     ALL mse={res_gt['all']['mse']:.3e}, l2={res_gt['all']['l2']:.3e}, max|r|={res_gt['all']['max_abs']:.3e}; "
            f"comp0 mse={res_gt['comp0']['mse']:.3e}, comp1 mse={res_gt['comp1']['mse']:.3e}"
        )
        print(msg)
        logger.info(msg)

    # Pointwise residual (SON-style) for GT for reference
    if verbose and FiniteDiffVand is not None and gt is not None:
        alt = compute_pde_residual_pointwise(
            mu_scalar=gt,
            A=A,
            b=b,
            Q=Q,
            dx=dx,
            dy=dy,
            t_min=t_min,
            t_max=t_max,
            n_clip=5,
        )
        msg = (
            f"[solver3] Residual (pointwise) using GT:  ALL mse={alt['all']['mse']:.3e}, l2={alt['all']['l2']:.3e}, max|r|={alt['all']['max_abs']:.3e}; "
            f"comp0 mse={alt['comp0']['mse']:.3e}, comp1 mse={alt['comp1']['mse']:.3e}"
        )
        print(msg)
        logger.info(msg)

    # Optional: save a representative slice if index available
    if verbose:
        slice_t = 85
        if 0 <= slice_t < nt:
            np.savetxt('mu_slice.txt', mu[:, :, slice_t, 0], fmt='%.6f', header=f'PDE Solved (mu[:, :, {slice_t}, 0])')
            if gt is not None and gt.shape[2] > slice_t:
                np.savetxt('gt_slice.txt', gt[:, :, slice_t], fmt='%.6f', header=f'GT for comparison (gt[:, :, {slice_t}])')
            logger.info("[solver3] Saved mu_slice.txt and gt_slice.txt for t=%d", slice_t)
    # If a global g was estimated above we may have it in locals(); return it
    if 'g_est' in locals():
        return mu, float(g_est)
    return mu