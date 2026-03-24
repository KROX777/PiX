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


def np_grad(
    arr_list: Union[np.ndarray, List[np.ndarray]],
    grids: Tuple[np.ndarray, ...],
    is_time_grad: bool = False
) -> List[np.ndarray]:
    """ 
    Compute spatial or temporal gradient of NumPy arrays.
    
    Uses finite difference methods to compute derivatives along coordinate axes.
    
    Args:
        arr_list: Single or list of N-dimensional arrays to differentiate.
                  Each array shape: (nx, ny, [nz], nt)
        grids: Tuple of 1-D coordinate arrays for each dimension.
               Example: (x_grid, y_grid, t_grid) with shapes (nx,), (ny,), (nt,)
        is_time_grad: If True, compute temporal derivatives only.
                      If False, compute spatial derivatives only.
    
    Returns:
        List of gradient arrays. For each input array and each gradient axis,
        one output array of same shape as input.
        
    Examples:
        >>> u = np.random.rand(10, 10, 5)  # 2D field at 5 time steps
        >>> grids = (np.linspace(0,1,10), np.linspace(0,1,10), np.linspace(0,1,5))
        >>> du_dx, du_dy = np_grad(u, grids, is_time_grad=False)  # spatial gradients
        >>> du_dt = np_grad(u, grids, is_time_grad=True)[0]  # temporal gradient
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
            ret.append(FiniteDiffVand(arr, dx=dx, d=1, axis=axis_idx))
    return ret

def np_grad_all(arr_list, grids):
    """ 
    np_grad() wrapper, get [grad_, grad_grad_, dt_] in one call.
    """
    dt_ = np_grad(arr_list, grids, is_time_grad=True)
    grad_ = np_grad(arr_list, grids)
    grad_grad_  = np_grad(grad_, grids)
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