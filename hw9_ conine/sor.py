# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:33:29 2017

@author: Grant

An implementation of the Successive Over-Relaxation algorithm in Python, using 
just-in-time compiling from numba for additional speed
"""

import numpy as np
from numba import jit, njit, f8, u8

def solve(x0, A, b, omega, M, tol):
    '''Wrapper function to use SOR algorithm to solve Ax = b
    
    Parameters
    ===========
    x0: numpy array
        First guess for solution
    A: numpy array
        matrix describing linear system
    b: numpy array
        vector of constants
    omega: float
        relaxation factor
    M: integer
        maximum number of steps while seeking convergence
    tol: float
        error tolerancd for stopping iteration (error for convergence)
    '''
    
    # Format dtype of all parameters to numba double precision float
    # this helps @njit work correctly
    x0 = f8(x0)
    A = f8(A)
    b = f8(b)
    omega = f8(omega)
    M = u8(M)
    tol = f8(tol)
    
    # call actual SOR algorithm (need the numba dtypes to allow njit compile)
    x = solve_body(x0, A, b, omega, M, tol)
    
    return x

@njit
def solve_body(x0, A, b, omega, M, tol):
    '''SOR function. Iterates until errror is less than specified, or M 
    iterations, whichever comes first.
    
    Parameters
    ===========
    x0: numpy array
        First guess for solution
    A: numpy array
        matrix describing linear system
    b: numpy array
        vector of constants
    omega: float
        relaxation factor
    M: integer
        maximum number of steps while seeking convergence
    tol: float
        error size for stopping iteration (error for convergence)
    '''
   
    x = x0  # set initial guess
    tolcheck = 0 # flag for tolerance vs maximum number of steps
    err = f8(tol + 1000000) # initial error level
    
    # While loop. Main loop exit after M iterations, but has a secondary break
    # that stops when the observed error is less than selected tolerance
    for i in xrange(M):
        x_new = SOR_iter(x, A, b, omega) # perform SOR iteration
        err = resid(x_new, A, b) # compute new residual
        x = x_new # reset x for new pass (or for output)
        
        # second break parameter (under error tolerance)
        if err < tol:
            # extra information on why loop was terminated (tol surpassed)
            print('Error tolerance surpassed.')
            tolcheck = 1
            break
        
    # extra information on why loop was terminated (too many steps)    
    if tolcheck == 0:
        print('Maximum steps taken and convergence not reached.')
    
    return x
    

@njit
def SOR_iter(x, A, b, omega):
    ''' Helper function to perform SOR iteration step.
    
    Parameters
    ===========
    x: Nx1 numpy array, numba double float dtype
        The previous guess for the solution of Ax = b
    A: NxN numpy array, numba double float dtype
        matrix describing linear system
    b: Nx1 numpy array, numba double float dtype
        vector of constants
    omega: float, numba double float dtype
        relaxation factor
    '''
    
    N = len(b)

    x_new = np.zeros(N) #initialize x_new (outside of loop)
    
    # first step of SOR algorithm (haven't yet computed any of x_new)
    x_new[0] = (1 - omega) * x[0] + (omega / A[0,0]) * (b[0] - 
         np.sum(np.multiply(A[0,1:N], x[1:N])))
    
    for i in xrange(1,N):
        # split up the a's and x's for SOR summation
        a_low = A[i,0:i]
        x_low = x_new[0:i]
        a_hi  = A[i,i+1:N]
        x_hi  = x[i+1:N]
        
        # Compute new xi using SOR algorithm
        x_new[i] = ((1 - omega) * x[i] + 
            (omega / A[i,i]) * (b[i] - np.sum(np.multiply(a_low, x_low)) -
            np.sum(np.multiply(a_hi, x_hi))))
    
    return x_new

@njit
def resid(x, A, b):
    ''' Calculate L2-norm of b and x dot A '''
    
    # need to be numba double precision float so njit will work in body of
    # solve function
    err = f8(np.linalg.norm(b - np.dot(A, x)))
    
    return err


    