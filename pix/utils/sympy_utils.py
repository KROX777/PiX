import signal
import sympy as sp
from sympy import derive_by_array as ts_grad  #gradient
from sympy import tensorcontraction as ts_contr  #summation
from sympy import tensorproduct as ts_prod 
from sympy import transpose as ts_trans

class TimeoutError(Exception):
    pass

def handler(signum, frame):
    raise TimeoutError("Simplification process took too long")

def sp_simplify_with_timeout(expr, timeout=3):
    """ 
    Simplification for both Int and sp.expr, with time limitation 
    and also catch NotImplementedError (err msg: Improve MV Derivative support in collect)
    """
    if hasattr(expr, "simplify"):   
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            simplified = expr.simplify()
            signal.alarm(0)
            return simplified
        except (NotImplementedError, TimeoutError) as error:
            signal.alarm(0)
            return expr
    return expr

def dot(tensor1, tensor2):
    ''' 
    Dot production between two sympy tensors.
    Contraction of the last dim of input tensors.
    Input:
        tensor1:  rank(n) tensor.
        tensor2:  rank(m) tensor
    Output:
        rank(n+m-2) tensor.
    '''
    n = len(tensor1.shape)
    return ts_contr(ts_prod(tensor1, tensor2), (n-1, n))


def ddot(tensor1, tensor2):
    ''' 
    double Dot production between two sympy tensors.
    Contraction of the last two dims of input tensros.
    E.g. if both tensors are rank(2), A:B = a_{ij}b_{ij}
    Input:
        tensor1:  rank(n) tensor, n>=2
        tensor2:  rank(m) tensor, m>=2
    Output:
        rank(n+m-4) tensor.
    '''
    n = len(tensor1.shape)
    tmp = ts_prod(tensor1, tensor2)
    tmp = ts_contr(tmp, (n-2, n))
    tmp = ts_contr(tmp, (n-2, n-1))
    return tmp

def div(f, x):
    '''
    Divergence := \grad_{x} \cdot (f).
    Input:
        f: sympy.Array, any field, rank(n) tensor
        x: sympy.Array, Euler coord
    Output: Divergence, rank(n-1) tensor
    '''
    return ts_contr(ts_grad(f, x), (0, 1))

def conserve(f, vel, x, t):
    '''
    conservation form := df/dt + div(f*vel, x).
    Input:
        f: sympy.Array, any field, rank(n) tensor
        vel: sympy.Array, velocity field, rank(1) tensor
        x: sympy.Array, Euler coord
        t: sympy.Symbol, time
    Output:
        rank(n) tensor
    '''
    return f.diff(t) + div(ts_prod(f, vel), x)


def DDt(f, vel, x, t):
    '''
    Total gradient Df/Dt := df/dt + vel \dot grad(f).
    Input:
        f: sympy.Array, any field, rank(n) tensor
        vel: sympy.Array, velocity field, rank(1) tensor
        x: sympy.Array, Euler coord
        t: sympy.Symbol, time
    Output:
        total gradient, rank(n) tensor
    '''
    return f.diff(t) + dot(vel, ts_grad(f, x))

def ts_1d_list(tensor):
    """
    Flatten a sympy tensor to list.
    """
    if not hasattr(tensor, 'tolist'): # rank(0) tensor
        return [tensor]
    else:
        lst = tensor.tolist()
        while isinstance(lst[0], list): #recursively de-nest the list
            lst = sum(lst, [])
        return lst

def ts_flatten(tensor): 
    """
    Flatten a sp.tensor to a 1d sp.tensor.
    """
    return sp.Array(ts_1d_list(tensor))

def ts_grad_all(tensor, x, t): 
    """
    warpper of ts_grad(), get [grad_, grad_grad_, dt_] in one call.
    """
    dt_ = ts_grad(tensor, t)
    grad_ = ts_grad(tensor, x)
    grad_grad_ = ts_grad(ts_flatten(grad_), x)
    return [grad_, grad_grad_, dt_]

def sp_maximum(a, b):
    """
    maximum of 2 symbols.
    Ref: https://stackoverflow.com/questions/60723841/how-do-i-use-sympy-lambdify-with-max-function-to-substitute-numpy-maximum-instea
    """
    return sp.Piecewise((b, a < b), (a, True))