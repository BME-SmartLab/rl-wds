# -*- coding: utf-8 -*-
from scipy.optimize import minimize as optimize

def minimize(target, dims, maxfev=1000, init_guess=None):
    """Minimizing the target function with SciPy's the Nelder-Mead method.
       The result depends on the initial guess, no random behaviour is presented."""
    if not init_guess:
        init_guess  = dims * [1.]
    options     = {
                    'maxfev': maxfev,
                    'xatol' : .005,
                    'fatol' : .01}
    result      = optimize(target, init_guess, method='Nelder-Mead', options=options)
    return result.x, result.fun, result.nit
