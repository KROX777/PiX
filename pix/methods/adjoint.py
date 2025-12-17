import numpy as np
import scipy.integrate as integrate
import time

# Forward model with an expansion of all possible terms
def forward_m(params, ps, ds, f0, dx, dt, Nt):
    ## Inputs:
    #  params (numPDE, len(ds), P)
    #  ps is a list of powers for f
    #  e.g. [(0,0), (1,0), (2,1)] is 1, f_0, f_0^2*f_1
    #  ds is a list of derivative index tuples.
    #  e.g. [(0,0), (1,0), (2,0)] for f, df/dx0, d^2f/dx0^2
    #  f0 (numPDE, Nx1, Nx2, ...)
    #  dx (dx0, dx1, ...)
    ## Output:
    #  f (numPDE, Nt, Nx1, Nx2, ...)
    numPDE = f0.shape[0]
    grid_shape = f0[0,:].shape
    f = np.zeros( (numPDE,) + (Nt,) + grid_shape )
    f[:,0,:] = f0
    for i in range(Nt-1):
        f[:,i+1,:] = f[:,i,:]
        for i_pde in range(numPDE):
            for i_p, p in enumerate(ps):
                for i_d, d in enumerate(ds): ## for each d of ds, compute nabla^{d} [f^p] contribution
                    temp = np.ones_like(f[0, i, :])
                    for i_pi, pi in enumerate(p):
                        temp *= f[i_pi, i, :] ** pi
                    for i_di, di in enumerate(d): ## for each element di of d=(d0,d1,...)
                        for _ in range(1, di+1): ## take di derivatives
                            temp = np.gradient(temp, axis=i_di)/dx[i_di]
                    # multiply to its parameter, then add to the whole expression
                    f[i_pde, i+1,:] -= params[i_pde, i_d, i_p] * temp * dt
    return f

# Adjoint equation for the Lagarange multiplier
def adjoint_eq(params, ps, ds, fs, f, dx, dt, Nt):
    ## Inputs:
    #  params (numPDE, len(ds), P)
    #  ps is a list of powers for f
    #  e.g. [(0,0), (1,0), (2,1)] is 1, f_0, f_0^2*f_1
    #  ds is a list of derivative index tuples.
    #  e.g. [(0,0), (1,0), (2,0)] for f, df/dx1, d^2f/dx^2
    #  fs (numPDE, Nx1, Nx2, ...)
    #  f (numPDE, Nt, Nx1, Nx2, ...)
    #  dx (dx0, dx1, ...)
    ## Output:
    #  lam (numPDE, Nt, Nx0, Nx1, ...)
    numPDE = f.shape[0]
    lam = np.zeros(f.shape)
    lam[:, -1, :] = 2 * (fs[:, :] - f[:, -1, :])
    for i in range(Nt - 1, 0, -1):
        lam[:, i-1, :] = lam[:, i, :]
        for i_pde in range(numPDE):
            for i_d, d in enumerate(ds):## for each d of ds, compute nabla^{d}[lambda] contribution
                temp = lam[i_pde, i, :]
                for i_di, di in enumerate(d): ## for each element di of d=(d0,d1,...)
                    for _ in range(1, di+1):
                        temp = np.gradient(temp, axis=i_di)/dx[i_di]
                temp *= (-1)**(sum(d))
                for i_p, p in enumerate(ps):
                    tempf = np.ones_like(f[0, i, :])
                    for i_pi, pi in enumerate(p):
                        if i_pi != i_pde:
                            tempf *= f[i_pi, i, :] ** pi
                        else:
                            if pi>0: # take derivative if f_i has power > 0
                                tempf *= pi *  f[i_pi, i, :] ** (pi-1)
                            else:
                                tempf *= 0. ## set derivative to zero when f^p does not have any term with f_i
                    lam[i_pde, i-1,:] -= params[i_pde, i_d, i_p] * tempf * temp * dt
    return lam

# Finding the parameters of the PDE using adjoint method
def AdjointFindPDE(fs, x, dx, data_dt=1, Nt_fine=2, nt=None, dt=None, avg=False, gamma=1e-3, epochs=100, epthr=None, tol_thr=None, beta = 0.01, learning_rates=None, ds=np.array([[1], [2], [3]]), ps=np.array([[1], [2], [3]]), V = 1., tol = 1e-12, eps0=1e-12):
    if epthr is None:
        epthr = epochs
    if tol_thr is None:
        tol_thr = tol*10
    if nt is None:
        nt = [Nt_fine for _ in range(fs.shape[1])]
        dt = [data_dt/(Nt_fine-1) for _ in range(fs.shape[1])]
    numPDE = fs.shape[0]
    D = np.max(np.sum(ds,axis=1))
    P = np.max(np.sum(ps,axis=1))
    params = np.zeros((numPDE,len(ds),len(ps)))
    if learning_rates is None:
        learning_rates = np.ones((numPDE, len(ds),len(ps)))
        for i, d in enumerate(ds):
            for j, p in enumerate(ps):
                learning_rates[:,i,j] = beta * ( min(dx)**(sum(d)-D) )
    estimated_params = np.zeros((epochs+1, numPDE, len(ds),len(ps)))
    estimated_params[0,:] = params

    bool_thr = False
    params0 = params
    losses = []
    for ep in range(1, epochs+1):
        dC_dparams = np.zeros((numPDE, len(ds),len(ps)))
        loss = 0
        nloss = 0
        for i in range(len(nt)-1):
            fst = fs[:, i:i+2,:]
            ft0 = fst[:, 0, :]
    
            ft  = forward_m(params, ps, ds, ft0, dx, dt[i], nt[i])

            loss += np.sum((ft-fst)**2)
            nloss += np.sum((fst)**2)
            lam = adjoint_eq(params, ps, ds, fst[:, -1,:], ft, dx, dt[i], nt[i])

            if avg is False:
                dC_dparams = np.zeros((numPDE, len(ds),len(ps)))
            for i_pde in range(numPDE):
                for i_d, d in enumerate(ds):
                    temp = np.array([lam[i_pde, ii,:] for ii in range(nt[i])])
                    for i_di, di in enumerate(d):
                        for rep in range(1, di+1):
                            temp = np.array([np.gradient(temp[ii,:], axis=i_di) for ii in range(nt[i])] ) / dx[i_di]
                    temp *= (-1)**sum(d)
                    for i_p, p in enumerate(ps):
                        tempf = np.ones_like(ft[0,:]) # (Nt, Nx0, Nx1,...)
                        for i_pi, pi in enumerate(p):
                            tempf *= ft[i_pi, :]**pi
                        integ = np.array( [ tempf[ii,:] * temp[ii,:] for ii in range(nt[i])] )
                        for xi in range(len(dx)-1, -1, -1):
                            integ = np.array( [integrate.trapezoid(integ[ii,:], x[xi], axis=xi) for ii in range(nt[i])] )
                        dC_dparams[i_pde, i_d, i_p] += np.sum(integ) * dt[i] / (nt[i]*dt[i]) / V + 2*eps0*params[i_pde, i_d, i_p]
            if avg is False:
                # update parameters afte seeing each data points
                params = params - dC_dparams * learning_rates
            #if ep>epthr: # thresholding
            if bool_thr is False and (np.linalg.norm(params-params0) < tol_thr or ep>epthr):
                bool_thr = True
            if bool_thr is True:
                params[abs(params)<gamma] = 0.
        if avg is True:
            # update parameters only after seeing all the data points
            params = params - dC_dparams/len(nt) * learning_rates
        #if ep>epthr:# thresholding
        if bool_thr is True:
            params[abs(params)<gamma] = 0.
        estimated_params[ep,:] = params
        if np.linalg.norm(params-params0) < tol:
            epochs = ep
            break
        params0 = params
        losses.append(loss / nloss / len(nt) )
    
    estimated_params = estimated_params[:epochs,:]
    eps = [i for i in range(epochs)]
    return estimated_params, eps, losses
    
