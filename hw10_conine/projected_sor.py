# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 17:41:27 2017

@author: Grant
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:33:29 2017

@author: Grant

An implementation of the Successive Over-Relaxation algorithm in Python, using 
just-in-time compiling from numba for additional speed
"""

import numpy as np
from numba import njit, f8, u8

def solve(v0, b, g, omega, tol, theta, lamb):
    '''Wrapper function to use SOR algorithm to solve Ax = b
    
    Parameters
    ===========
    xv: numpy array
        First guess for solution
    b: numpy array
        vector to represent A * w
    g: numpy array
        vector representing early excercise values
    omega: float
        relaxation factor
    tol: float
        error tolerancd for stopping iteration (error for convergence)
    theta: float
        parameter controlling what discretization method is being used
    lamb: float
        lambda parameter from option pricing model
    '''
    
    # Format dtype of all parameters to numba double precision float
    # this helps @njit work correctly
    v0 = f8(v0)
    b = f8(b)
    g = f8(g)
    omega = f8(omega)
    M = u8(10 ** 6)
    tol = f8(tol)
    theta = f8(theta)
    lamb = f8(lamb)
    
    # call actual SOR algorithm (need the numba dtypes to allow njit compile)
    x = solve_body(v0, b, g, omega, tol, theta, lamb, M)
    
    return x

@njit
def solve_body(v0, b, g, omega, tol, theta, lamb, M):
    '''Projected SOR function. Iterates until errror is less than specified, or M 
    iterations, whichever comes first. 
    '''
   
      # set initial guess
    err = f8(tol + 1000000) # initial error level
    v = v0
    
    # While loop. Main loop exit after M iterations, but has a secondary break
    # that stops when the observed error is less than selected tolerance
    
    for k in range(0,M):
        v_new = SOR_iter(b, v, g, omega, tol, theta, lamb) # perform SOR iteration
        err = resid(v_new, v) # compute new residual
        v = v_new
        
        # second break parameter (under error tolerance)
        if err < tol:
            break
        
    
    return v
    

@njit
def SOR_iter(b, v, g, omega, tol, theta, lamb):
    ''' Helper function to perform SOR iteration step.
    '''    
    N = len(b)

    v_new = np.zeros(N) #initialize x_new (outside of loop)
    
    # first step of SOR algorithm (haven't yet computed any of x_new)
    v_new[0] = 0
    
    for i in range(1,N):
        # Compute new xi using SOR algorithm
        v_new[i] = np.maximum(g[i+1], 
                              v[i] + omega / (1 + 2 * theta * lamb) *
                              (b[i] + theta * lamb * v_new[i-1] -
                               (1 + 2 * theta * lamb) * v[i] + theta * lamb * v[i+1]))
    
    return v_new

@njit
def resid(x_new, x):
    ''' Calculate L2-norm of x and x_new'''
    
    # need to be numba double precision float so njit will work in body of
    # solve function
    err = f8(np.linalg.norm(x_new - x))
    
    return err