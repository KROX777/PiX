"""
NumPy utility functions for numerical computation.

Provides utilities for:
    - Gradient computation via finite differences
    - Array pooling and downsampling
    - Numerical derivatives and smoothing
"""

from typing import List, Union, Tuple
import numpy as np

from pix.utils.finite_diff import FiniteDiffVand


def _spectral_diff(arr, grid, axis=0):
    """
    Compute derivative using spectral (Fourier) differentiation.
    Best for periodic data; has boundary artifacts for non-periodic.
    """
    n = arr.shape[axis]
    L = grid[-1] - grid[0] + (grid[1] - grid[0])  # approximate period
    # wave numbers
    k = 2 * np.pi * np.fft.fftfreq(n, d=(grid[1] - grid[0]))
    
    # move axis to front
    trans_idx = list(range(arr.ndim))
    trans_idx[axis] = 0
    trans_idx[0] = axis
    u = arr.transpose(*trans_idx)
    shape = u.shape
    u = u.reshape(n, -1)
    
    ux = np.zeros_like(u)
    for i in range(u.shape[1]):
        u_hat = np.fft.fft(u[:, i])
        du_hat = 1j * k * u_hat
        ux[:, i] = np.fft.ifft(du_hat).real
    
    ux = ux.reshape(shape)
    ux = ux.transpose(*trans_idx)
    return ux


def _numpy_gradient(arr, dx, axis=0):
    """
    Compute derivative using np.gradient (2nd-order central difference).
    Uses scalar spacing for robustness (avoids grid/arr shape mismatch after clipping).
    """
    # np.gradient with scalar spacing works regardless of boundary clipping mismatches
    grads = np.gradient(arr, dx, axis=axis)
    return grads


def np_grad(
    arr_list: Union[np.ndarray, List[np.ndarray]],
    grids: Tuple[np.ndarray, ...],
    is_time_grad: bool = False,
    method: str = "polynomial"
) -> List[np.ndarray]:
    """ 
    Compute spatial or temporal gradient of NumPy arrays.
    
    Supports multiple differentiation methods:
    - 'polynomial': Polynomial interpolation via Vandermonde (PiX default, smooth)
    - 'numpy':      Standard np.gradient (central finite difference, faithful)
    - 'spectral':   Fourier spectral differentiation (accurate for smooth/periodic data)
    
    Args:
        arr_list: Single or list of N-dimensional arrays to differentiate.
                  Each array shape: (nx, ny, [nz], nt)
        grids: Tuple of 1-D coordinate arrays for each dimension.
               Example: (x_grid, y_grid, t_grid) with shapes (nx,), (ny,), (nt,)
        is_time_grad: If True, compute temporal derivatives only.
                      If False, compute spatial derivatives only.
        method: Differentiation method. One of 'polynomial', 'numpy', 'spectral'.
    
    Returns:
        List of gradient arrays. For each input array and each gradient axis,
        one output array of same shape as input.
        
    Examples:
        >>> u = np.random.rand(10, 10, 5)  # 2D field at 5 time steps
        >>> grids = (np.linspace(0,1,10), np.linspace(0,1,10), np.linspace(0,1,5))
        >>> du_dx, du_dy = np_grad(u, grids, is_time_grad=False, method='numpy')
        >>> du_dt = np_grad(u, grids, is_time_grad=True, method='numpy')[0]
    """
    if not isinstance(arr_list, list):
        arr_list = [arr_list]
    ret = []
    
    for axis_idx, grid in enumerate(grids):
        # Skip temporal derivatives if not requested, skip spatial if temporal requested
        if is_time_grad ^ (axis_idx == len(grids)-1):
            continue
        dx = grid[1] - grid[0]
        
        for arr in arr_list:
            if method == "polynomial":
                ret.append(FiniteDiffVand(arr, dx=dx, d=1, axis=axis_idx))
            elif method == "numpy":
                ret.append(_numpy_gradient(arr, dx, axis=axis_idx))
            elif method == "spectral":
                ret.append(_spectral_diff(arr, grid, axis=axis_idx))
            else:
                raise ValueError(f"Unknown differentiation method: {method}. "
                                 f"Supported: 'polynomial', 'numpy', 'spectral'")
    return ret

def np_grad_all(arr_list, grids, method="polynomial"):
    """ 
    np_grad() wrapper, get [grad_, grad_grad_, dt_] in one call.
    """
    dt_ = np_grad(arr_list, grids, is_time_grad=True, method=method)
    grad_ = np_grad(arr_list, grids, is_time_grad=False, method=method)
    grad_grad_  = np_grad(grad_, grids, is_time_grad=False, method=method)
    return [grad_, grad_grad_, dt_]


def pooling(mat, ksize, method='mean', pad=False):
    '''
    Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''
    if not hasattr(mat, "shape"):
        return np.zeros((1,))
    
    m, n = mat.shape[:2]
    ky, kx = ksize

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m,ky)
        nx = _ceil(n,kx)
        size = (ny * ky, nx * kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[: m, : n,...] = mat
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = mat[: ny * ky, :nx * kx, ...]

    new_shape = (ny, ky, nx, kx) + mat.shape[2:]

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1,3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1,3))

    return result


def np_ms(a):
    """numpy array mean square"""
    b = a ** 2
    if isinstance(b, np.ndarray):
        b = b.mean()
    return b