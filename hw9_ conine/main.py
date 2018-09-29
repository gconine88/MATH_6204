# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:06:46 2017

@author: Grant

UNC Charlotte MS Math Finance
MATH 6204 Numerical Methods for Financial Derivatives
Prof Hwan Lin

Set of functions to solve a linear system using Successive Over Relaxation
algorithm. SOR will terminate either when a given tolerance level is beat or
when a given number of steps is surpassed.

For explaniation of algorithm see:
https://en.wikipedia.org/wiki/Successive_over-relaxation
"""

import numpy as np
import pandas as pd
import sor

# create tridiagonal matrix to solve

# first create vectors for each diagonal entry
a = np.repeat(2, 10)
b = np.repeat(1, 9)
c = np.repeat(1, 9)
d = np.repeat(10, 10)

# then paste together with np.diag
A = np.diag(a, 0) + np.diag(b, 1) + np.diag(c, -1)

x0 = np.repeat(0, 10) # initial guess for SOR
omega = 1.5 #set relaxation parameter, should be between 1 and 2
M = 1000000 # set max number of iterations
tol = 0.0000001 # set error toleraance

# call our SOR algorithm solver.  
x = sor.solve(x0, A, d, omega, M, tol)

# tabulate output for clarity
d = {'Solution': x}
df = pd.DataFrame(d, index = ['x0', 'x1', 'x2', 'x3', 'x4', 
                              'x5', 'x6', 'x7', 'x8', 'x9'])
print(df)