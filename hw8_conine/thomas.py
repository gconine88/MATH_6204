# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:06:49 2017

@author: Grant

An implementation of the Thomas algorithm in Python, using just-in-time 
compiling from numba for additional speed
"""

import numpy as np
from numba import njit, f8

def full(A, d):
    '''Helper function for Thomas algorith. Breaks matrix into tridiagonal
    elements for easier processing by algorithm. '''
    
    # pass numba float64 dtype np.arrays to the solve function - need to 
    # perform this step to allow for nopython execution of thomas algorithm 
    # which yields maximum speed
    a = f8(np.diagonal(A, offset=0))
    b = f8(np.diagonal(A, offset=1))
    c = f8(np.diagonal(A, offset=-1))
    dfloat = f8(d)
    
    D = np.diag(a, 0) + np.diag(b, 1) + np.diag(c, -1) #create test matrix
    
    # test if D is 'close enough' to A - if not that means A was not 
    # tridiagonal and the function raises an exception
    if not np.allclose(A, D):
        raise Exception('The given A is not tridiagonal')
    
    # pass to thomas algorithm solver
    x = solve(a, b, c, dfloat)
    
    return x
    
# chose to use njit decorator to force nopython implementation and 
# get faster speed. Downside is I lose flexibility in input of solver, must
# wrap in another function which will format data correctly
@njit('f8[:](f8[:], f8[:], f8[:], f8[:])')
def solve(a, b, c, d):
    ''' Thomas algorithm to solve a tridiagonal system of equations
    
    INPUTS
    ========
    a: numpy array
        the diagonal entries
    b: numpy array
        the superdiagonal entries
    c: numpy array
        the subdiagonal entries
    d: numpy array
        the right-hand side of the system of equations
    
    RETURNS
    ========
    The solution for the given tri-diagonal system of equations.
    '''
    
    n = len(a) # determine number of equations in system
    
    #initialize
    alpha = np.zeros(n)
    beta = np.zeros(n)
    alpha[0] = a[0]
    beta[0] = d[0]
    
    # first (forward) loop to zero c[i]'s
    for i in range(1, n, 1):
        # in python, c's index is from 0 to n-2, not 1 to n-1, have to subtract 1
        alpha[i] = a[i] - (b[i-1] * c[i-1]) / alpha[i-1]
        beta[i] = d[i] - (beta[i-1] * c[i-1]) / alpha[i-1]
    
    #initialize and set last step
    x = np.zeros(n)
    x[n-1] = beta[n-1] / alpha[n-1]
    
    # second (backwards) loop to find solutions
    for j in range(n-2, -1, -1): #indices are weird, want to step from n-2 to 0 
        x[j] = (beta[j] - b[j-1] * x[j+1]) / alpha[j]
        
    return x
        

