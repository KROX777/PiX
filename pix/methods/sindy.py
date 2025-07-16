"""
Original codes by Mingquan Feng: https://github.com/FengMingquan-sjtu/PhysPDE
"""
import os
import signal
import time
from collections import namedtuple
import itertools
import copy
import pprint

from ordered_set import OrderedSet
import scipy
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import sympy as sp
from sympy import derive_by_array as ts_grad  #gradient
from sympy import tensorcontraction as ts_contr  #summation
from sympy import tensorproduct as ts_prod 
from sympy import transpose as ts_trans
import pysindy as ps
import argparse

from ..utils.finite_diff import get_diff, FiniteDiffVand
from ..utils import switch
from SONv4 import get_raw_data

#---- Load Data and Test Sindy (Begin)----
def sindy(u, grids, problem, datasource, use_weakform=True):
    """ 
    sindy pde discovery.
    Input:
        u: np.ndarray, u.shape=(nx, nt, u_dim) for 2d problems; u.shape=(nx, ny, nt, u_dim) for 3d problems.
        grids: tuple of np.ndarray;  grids=(x,t) for 2d problems,  grids=(x, y, t) for 2d problems,  
        Note that t is only used to obtain dt = t[1]-t[0], it's okay if nt of u != len(t).
        problem, datasource: strings
        use_weakform: bool, whether use weak-sindy for noise and shock pde.
    Output:
        Equation and Score are printed out, no return value.
    """
    
    if len(grids) == 3: # 3d data
        x, y, t = grids
        dx = x[1] - x[0]
        dt = t[1] - t[0]
        if len(t) != u.shape[-2]: 
            # len(t) maybe not equal to nt of u, need re-calculate
            t = np.arange(u.shape[-2]) * dt  

        if problem == "2d_incomp_viscose_newton_ns" or problem == "2d_comp_viscose_newton_ns":
            # eps = 1e-10
            # library_functions= [lambda x: x, lambda x: 1/(abs(x)+eps)]
            # function_names=[lambda x: x, lambda x: x+"^-1"]
            library_functions= [lambda x: x]
            function_names=[lambda x: x]

            if use_weakform:
                spatiotemporal_grid = np.zeros((u.shape[0], u.shape[1], u.shape[2], 3))
                spatiotemporal_grid[:, :, :, 0] = x[:, np.newaxis, np.newaxis]
                spatiotemporal_grid[:, :, :, 1] = y[np.newaxis, :, np.newaxis]
                spatiotemporal_grid[:, :, :, 2] = t[np.newaxis, np.newaxis, :]
                
                pde_lib = ps.WeakPDELibrary(
                    library_functions=library_functions,
                    spatiotemporal_grid=spatiotemporal_grid,
                    function_names=function_names,
                    derivative_order=2,
                    is_uniform=True,
                    periodic=False,
                    include_interaction=True,
                    multiindices=np.array([[1,0],[0,1],[2,0],[0,2]]),  #only allow [_x, _y, _xx, _yy]
                    K=1000,
                )

                threshold = 0.0001
                alpha = 1e-20

                print(f"{use_weakform=},{threshold=},{alpha=}")
                optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True, max_iter=200)
                model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)

                model.fit(u[...,:3])
                model.print()
                score = model.score(u[...,:3], t=dt, metric=mean_squared_error)
                print("MSE(wsindy)=", score)

            else: 
                # strong form sindy
                X, Y = np.meshgrid(x, y)
                spatial_grid = np.asarray([X, Y]).T  #spatial_grid.shape= (325, 170, 2) = (x,y,2)
                pde_lib = ps.PDELibrary(
                    library_functions=library_functions,
                    function_names=function_names,
                    derivative_order=2,
                    spatial_grid=spatial_grid,
                    include_bias=False,
                    is_uniform=True,
                    include_interaction=True,
                )
                threshold = 0.1
                alpha = 50

                print(f"{use_weakform=}, {threshold=}, {alpha=}")
                optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True, max_iter=200)
                model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)

                u_dot = np.zeros_like(u)
                u_dot[...,0] = FiniteDiffVand(u[...,0], dt, d=1, axis=2) #LHS[0] = ux_t
                u_dot[...,1] = FiniteDiffVand(u[...,1], dt, d=1, axis=2) #LHS[1] = uy_t
                u_dot[...,2] = FiniteDiffVand(u[...,0], dx, d=1, axis=0) #LHS[3] = ux_x
                                                                        
                model.fit(u, x_dot=u_dot)
                model.print()
                score = model.score(u, x_dot=u_dot, metric=mean_squared_error)
                print("MSE(sindy)=", score)
        elif problem == "2d_comp_viscose_new_non_newton":
            eps = 1e-10
            library_functions= [lambda x: x, lambda x: 1/(abs(x)+eps)]
            function_names=[lambda x: x, lambda x: x+"^-1"]
            # strong form sindy
            X, Y = np.meshgrid(x, y)
            spatial_grid = np.asarray([X, Y]).T  #spatial_grid.shape= (325, 170, 2) = (x,y,2)
            pde_lib = ps.PDELibrary(
                library_functions=library_functions,
                function_names=function_names,
                derivative_order=2,
                spatial_grid=spatial_grid,
                include_bias=False,
                is_uniform=True,
                include_interaction=True,
                multiindices=np.array([[1,0],[0,1],[2,0],[0,2]]),  #only allow [_x, _y, _xx, _yy]
            )
            threshold = 0.1
            alpha = 50

            print(f"{use_weakform=}, {threshold=}, {alpha=}")
            optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True, max_iter=200)
            model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)

            u_dot = np.zeros_like(u)
            u_dot[...,0] = FiniteDiffVand(u[...,0], dt, d=1, axis=2) #LHS[0] = ux_t
            u_dot[...,1] = FiniteDiffVand(u[...,1], dt, d=1, axis=2) #LHS[1] = uy_t
            u_dot[...,2] = FiniteDiffVand(u[...,0], dx, d=1, axis=0) #LHS[3] = ux_x
                                                                    
            model.fit(u, x_dot=u_dot)
            model.print()
            score = model.score(u, x_dot=u_dot, metric=mean_squared_error)
            print("MSE(sindy)=", score)
        elif problem == "2d_heat_comp_v2":
            eps = 1e-10
            library_functions= [lambda x: x]
            function_names=[lambda x: x]
            # strong form sindy
            X, Y = np.meshgrid(x, y)
            spatial_grid = np.asarray([X, Y]).T  #spatial_grid.shape= (325, 170, 2) = (x,y,2)
            pde_lib = ps.PDELibrary(
                library_functions=library_functions,
                function_names=function_names,
                derivative_order=2,
                spatial_grid=spatial_grid,
                include_bias=True,
                is_uniform=True,
                include_interaction=True,
                multiindices=np.array([[1,0],[0,1],[2,0],[0,2]]),  #only allow [_x, _y, _xx, _yy]
            )
            threshold = 0.01
            alpha = 1e-8

            print(f"{use_weakform=}, {threshold=}, {alpha=}")
            optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True, max_iter=200)
            model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)

            u_dot = np.zeros_like(u)
            u_dot[...,0] = FiniteDiffVand(u[...,0], dt, d=1, axis=2) #LHS[0] = ux_t
            u_dot[...,1] = FiniteDiffVand(u[...,1], dt, d=1, axis=2) #LHS[1] = uy_t
            u_dot[...,2] = FiniteDiffVand(u[...,0], dx, d=1, axis=0) #LHS[3] = ux_x
            u_dot[...,3] = FiniteDiffVand(u[...,4], dx, d=1, axis=2) #LHS[3] = T_t
                                                                    
            model.fit(u, x_dot=u_dot)
            model.print()
            score = model.score(u, x_dot=u_dot, metric=mean_squared_error)
            print("MSE(sindy)=", score)
    else:
        raise ValueError

def test_sindy(problem, dataSource, useWeakForm=False):
    print("---start to retrieve data---")
    U, grid = get_raw_data(problem, dataSource)
    print("---start to do sindy---")
    sindy(U, grid, problem, dataSource, useWeakForm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_type", type=int, default=0)
    args = parser.parse_args()
    problems = ["2d_comp_viscose_newton_ns", '2d_comp_viscose_new_non_newton', '2d_heat_comp_v2']
    problem = problems[args.prob_type]
    dataSource =  "COMSOL"
    test_sindy(problem, dataSource)