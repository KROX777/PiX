import numpy as np
from scipy.signal import savgol_filter


def FiniteDiffVand(u, dx, d=1, axis=0, poly_degree=6, use_smooth=False):
    """Differentiate a uniform-grid array with a local polynomial stencil.

    The historical name is retained for API compatibility. The requested
    derivative is computed directly instead of recursively differentiating
    already noisy derivatives. SciPy constructs the stencil in unit
    coordinates and applies ``dx`` only as the final scale, avoiding the
    ill-conditioned physical-coordinate Vandermonde matrix used previously.
    """
    u = np.asarray(u)
    if not isinstance(d, (int, np.integer)) or d < 0:
        raise ValueError(f"Derivative order must be a non-negative integer, got {d!r}")
    if not np.isfinite(dx) or dx == 0:
        raise ValueError(f"Grid spacing must be finite and nonzero, got {dx!r}")
    if axis < 0:
        axis += u.ndim
    if axis < 0 or axis >= u.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array with {u.ndim} dimensions")
    n = u.shape[axis]
    if n < 3:
        raise ValueError(f"At least 3 grid points are required, got {n}")

    max_window = n if n % 2 == 1 else n - 1
    requested_window = poly_degree + 1 if poly_degree % 2 == 0 else poly_degree + 2
    if use_smooth:
        requested_window = max(requested_window, 11)
    window_length = min(max_window, requested_window)
    # Third derivatives amplify grid-scale errors especially strongly. A cubic
    # local fit is more robust there; orders 1, 2 and 4 retain the high-order
    # fit needed for low truncation error on smooth fields.
    fit_degree = 3 if d == 3 else poly_degree
    effective_degree = min(max(d, fit_degree), poly_degree, window_length - 1)
    if d > effective_degree:
        raise ValueError(
            f"Derivative order {d} exceeds polynomial degree {effective_degree} "
            f"for axis length {n}"
        )

    return savgol_filter(
        u,
        window_length=window_length,
        polyorder=effective_degree,
        deriv=d,
        delta=float(dx),
        axis=axis,
        mode="interp",
    )


def get_diff(u, dt, dx, dy, axis_list=(0, 1, 2), diff_func=FiniteDiffVand):
    """Return first time/space and second pure-space derivatives."""
    u_t = diff_func(u, dt, d=1, axis=axis_list[0])
    u_x = diff_func(u, dx, d=1, axis=axis_list[1])
    u_y = diff_func(u, dy, d=1, axis=axis_list[2])
    u_xx = diff_func(u, dx, d=2, axis=axis_list[1])
    u_yy = diff_func(u, dy, d=2, axis=axis_list[2])
    return u_t, u_x, u_y, u_xx, u_yy
