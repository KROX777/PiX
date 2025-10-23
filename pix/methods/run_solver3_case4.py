"""Quick runner to test solver3 with the COMSOL case used in diagnose_mu_constant.

Additionally, this script visualizes the resulting mu field:
- Heatmaps for mu[..., comp] at a few time slices
- Histograms of values for each component
- A centerline slice plot for quick sanity

Plots are saved under PhysPDE/output/solver3_plots.
"""
import importlib.util, sys, os
import numpy as np
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pde_solver import run_solver, assemble_A_b_Q_from_args

if __name__ == "__main__":
    root = os.path.dirname(__file__)
    target = os.path.join(root, "4_0.0796.py")
    spec = importlib.util.spec_from_file_location("case_4_0796", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["case_4_0796"] = mod
    spec.loader.exec_module(mod)

    dataname_tuple = ('Untitled_ggg_0_n_1.3_mu0_0.1_v_up_0.5_v_bottom_0.5_2d_grid', 'COMSOL')
    args, _ = mod.Fl.load_np_data(dataname_tuple)
    
    dx = dy = 0.0025
    dt = 0.004

    # Calculate gamma = sp.sqrt(2 * Fl.ddot(Fl.S, Fl.S))
    u = args[0]
    v = args[1]
    u_x = args[2]
    v_x = args[3]
    u_y = args[4]
    v_y = args[5]

    # Second-order fields
    u_xx = args[6]
    v_xx = args[7]
    u_xy = args[10]
    v_xy = args[11]
    u_yy = args[12]
    v_yy = args[13]

    S_xx = u_x
    S_xy = 0.5 * (u_y + v_x)
    S_yy = v_y
    S_ddot_S = S_xx**2 + 2 * S_xy**2 + S_yy**2
    gamma = np.sqrt(2 * S_ddot_S)
    # gt = 0.01 + 0.01 * np.cos(gamma) / gamma + 0.01 * np.log(gamma)
    gt = 0.1 * (gamma ** 0.3)
    
    A, b, Q = assemble_A_b_Q_from_args(args)

    # Choose time range to solve (example: full range)
    nx, ny, nt = u.shape
    t_min, t_max = 81, 90
    
    mu_x0 = np.stack([gt[0, :, :], gt[0, :, :]], axis=-1)  # Shape: (ny, nt, 2)
    mu_y0 = np.stack([gt[:, 0, :], gt[:, 0, :]], axis=-1)  # Shape: (nx, nt, 2)
    mu_xN = np.stack([gt[-1, :, :], gt[-1, :, :]], axis=-1)    # (ny, nt, 2)
    mu_yN = np.stack([gt[:, -1, :], gt[:, -1, :]], axis=-1)    # (nx, nt, 2)
    
    # Solve for mu with Dirichlet BC set to gt over chosen time window
    mu = run_solver(args, t_min=t_min, t_max=t_max, dx=dx, dy=dy, dt=dt, gt=gt, solve_mode="ls", verbose=True)
    # Compute metrics on the same solved window
    sl = slice(t_min, t_max + 1)
    mu_sub = mu[:, :, sl, 0]
    gt_sub = gt[:, :, sl]
    l2_abs = float(np.linalg.norm((mu_sub - gt_sub).ravel()))
    denom = float(np.linalg.norm(gt_sub.ravel()) + 1e-12)
    l2_rel = float(l2_abs / denom)
    print(f"[runner] mu(comp0) L2 abs={l2_abs:.3e}, rel={l2_rel:.3e}")

    # ---- Visualization ----
    out_dir = os.path.join(root, "output", "solver3_plots")
    os.makedirs(out_dir, exist_ok=True)

    nx, ny, nt, _ = mu.shape
    # Use the same solved time range for plotting
    t_solved = t_max - t_min + 1
    # Reference scalar for plots (since gt is a field)
    gt_ref = float(np.mean(gt_sub))

    # Heatmaps for each component and a few time slices
    for comp in (0, 1):
        for t in range(t_solved):
            field = mu[:, :, t_min + t, comp]
            plt.figure(figsize=(6, 5))
            im = plt.imshow(field.T, origin="lower", aspect="equal", cmap="viridis")
            plt.colorbar(im, fraction=0.046, pad=0.04, label=f"mu comp{comp}")
            plt.title(f"mu comp{comp} heatmap (t={t})")
            plt.xlabel("x index")
            plt.ylabel("y index")
            fname = os.path.join(out_dir, f"mu_heatmap_comp{comp}_t{t}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()

        # Histogram across all solved time slices
        vals = mu[:, :, t_min:t_min + t_solved, comp].ravel()
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=50, color="#4c78a8", alpha=0.9)
        plt.axvline(gt_ref, color="red", linestyle="--", label=f"gt≈{gt_ref:.4f}")
        plt.title(f"mu comp{comp} histogram (first {t_solved} t-slices)")
        plt.xlabel("mu value")
        plt.ylabel("count")
        plt.legend()
        fname = os.path.join(out_dir, f"mu_hist_comp{comp}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    # Centerline slice plot at mid y for both components over x for t=0
    mid_j = ny // 2
    t0 = 0
    plt.figure(figsize=(7, 4))
    plt.plot(mu[:, mid_j, t0, 0], label="comp0")
    plt.plot(mu[:, mid_j, t0, 1], label="comp1")
    plt.axhline(gt_ref, color="red", linestyle="--", label=f"gt≈{gt_ref:.4f}")
    plt.title("mu centerline at y=mid, t=0")
    plt.xlabel("x index")
    plt.ylabel("mu")
    plt.legend()
    fname = os.path.join(out_dir, "mu_centerline_t0.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

    # Scatter plot: every point in mu[:, :, sl, 0]
    mu_sub = mu[:, :, t_min:t_min + t_solved, 0]
    vals = mu_sub.ravel()
    idx = np.arange(vals.size)
    plt.figure(figsize=(10, 4))
    plt.scatter(idx, vals, s=1, alpha=0.6, edgecolors="none")
    plt.axhline(gt_ref, color="red", linestyle="--", linewidth=1, label=f"gt≈{gt_ref:.4f}")
    plt.title(f"Scatter of all points: mu[:, :, {t_min}:{t_min + t_solved}, 0] (N={vals.size})")
    plt.xlabel("flattened index")
    plt.ylabel("mu (comp0)")
    plt.legend(loc="upper right")
    fname = os.path.join(out_dir, "mu_scatter_all_points_comp0.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f"[runner] Saved plots to: {out_dir}")
