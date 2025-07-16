import numpy as np
from .finite_diff import FiniteDiffVand

def np_grad(arr_list, grids, is_time_grad=False):
    """ 
    Spacial or temporal gradient of np arrays.
    Input:
        arr_list: A single or a list of len(grids)-dim np.ndarray, the matrices to take gradient, each arr.shape=(nx, ny, (nz), nt)
        grids: list of 1-dim np.ndarray, spacial and temporal grids. grids=[x,y,(z),t], x.shape = (nx,).
        is_time_grad: Boolean. True->return temporal gradient only, False->return spacial gradient only
    Output:
        ret: list of len(grids)-dim np.ndarray, resulting gradients, length=len(arr_list)*(len(grids)-1).

    e.g. Input: arr_list =[u, v], grids=(x,y,t), is_time_grad=False
        Output: [Derivative(u(x, y, t), x), Derivative(v(x, y, t), x), Derivative(u(x, y, t), y), Derivative(v(x, y, t), y)]
    e.g. Input: arr_list =[u, v], grids=(x,y,t), is_time_grad=True
        Output: [Derivative(u(x, y, t), t), Derivative(v(x, y, t), t)]
    """
    if not isinstance(arr_list, list): #for single array.
        arr_list = [arr_list]
    ret = []
    
    for axis_idx, grid in enumerate(grids):
        if is_time_grad ^ (axis_idx == len(grids)-1): #skip time or spacial gradients.
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