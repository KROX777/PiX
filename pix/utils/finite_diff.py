import numpy as np
import numpy.linalg as la
from scipy.signal import savgol_filter

def FiniteDiffVand(u, dx, d=1, axis=0, poly_degree=6, use_smooth=False):
    """
    Takes dth derivative data using finite difference and Vandermonde Matrices (polynomial interpolation)
    """
    # diff order higher than 1  is recursively solved
    if d > 1:
        u = FiniteDiffVand(u, dx, d-1, axis, poly_degree)

    # transpose u, such that the axis to be diff is the first axis.
    trans_idx = [i for i in range(len(u.shape))]
    trans_idx[axis] = 0
    trans_idx[0] = axis
    u = u.transpose(*trans_idx)

    # smoothing u
    if use_smooth:
        u = savgol_filter(u,window_length=11, polyorder=3, axis=0)

    n = u.shape[0]
    shape = u.shape
    ux = np.zeros_like(u)

    # Check if we have enough points for the polynomial degree
    if n <= poly_degree:
        poly_degree = max(1, n - 1)
        print(f"Warning: Reduced polynomial degree to {poly_degree} due to insufficient grid points")

    # Vandermonde Matrices with numerical stability
    nodes = np.array([i*dx for i in range(poly_degree+1) ])
    
    if dx == 0 or np.abs(dx) < 1e-12:
        raise ValueError(f"Warning: Grid spacing too small: {dx}")
    
    def monomial_deriv(i, x):
        if i == 0:
            return 0*x
        else:
            return i*nodes**(i-1)

    V = np.array([nodes**i for i in range(poly_degree+1)]).T
    Vprime = np.array([ monomial_deriv(i, nodes)for i in range(poly_degree+1)]).T
    
    # Use pseudo-inverse for numerical stability
    try:
        # Try regular inverse first
        cond_number = la.cond(V)
        if cond_number > 1e12:
            D = Vprime @ la.pinv(V)
        else:
            D = Vprime @ la.inv(V)
    except la.LinAlgError:
        print("Warning: Singular Vandermonde matrix, using pseudo-inverse")
        D = Vprime @ la.pinv(V)

    # --- convolution ---  i.e.  ux = convolve(u, D[poly_degree//2][::-1])
    # step0. reshape
    u = u.reshape(n, -1)
    ux = ux.reshape(n, -1)
    # step1. convolve at the left and right boundary
    left = poly_degree//2
    right = poly_degree - left
    
    for i in range(left):
        ux[i] = np.matmul(D[i], u[:poly_degree+1])
    for i in range(right):
        ux[-1-i] = np.matmul(D[-1-i], u[-poly_degree-1:])
    
    # step2. convolve in the middle
    for i in range(left, n-right):
        ux[i] = np.matmul(D[poly_degree//2], u[i-poly_degree//2 : i-poly_degree//2 + poly_degree +1])

    # step3. reshape back
    u = u.reshape(shape)
    ux = ux.reshape(shape)
    # --- convolution ends --- 

    #transpose back
    ux = ux.transpose(*trans_idx)
    return ux

def get_diff(u, dt, dx, dy, axis_list=[0,1,2], diff_func=FiniteDiffVand):
    '''
    axis_list=[t axis idx, x axis idx, y axis idx]
    '''
    u_t  = diff_func(u, dt, d=1, axis=axis_list[0])  
    u_x  = diff_func(u, dx, d=1, axis=axis_list[1])  
    u_y  = diff_func(u, dy, d=1, axis=axis_list[2])
    u_xx = diff_func(u, dx, d=2, axis=axis_list[1])  
    u_yy = diff_func(u, dy, d=2, axis=axis_list[2])
    return u_t, u_x, u_y, u_xx, u_yy