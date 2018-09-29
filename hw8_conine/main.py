# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:06:46 2017

@author: Grant

UNC Charlotte MS Math Finance
MATH 6204 Numerical Methods for Financial Derivatives
Prof Hwan Lin

Set of functions to solve a tridiagonal system.
"""

import numpy as np
import pandas as pd
import thomas

# create tridiagonal matrix to solve

# first create vectors for each diagonal entry
a = np.repeat(2, 10)
b = np.repeat(1, 9)
c = np.repeat(1, 9)
d = np.repeat(10, 10)

# then paste together with np.diag
A = np.diag(a, 0) + np.diag(b, 1) + np.diag(c, -1)

# call our thomas algorithm solver. I use the whole matrix solver rather than
# one which accepts only the individuals diagonal entries. 
x = thomas.full(A, d)

# tabulate output for clarity
d = {'Solution': x}
df = pd.DataFrame(d, index = ['x0', 'x1', 'x2', 'x3', 'x4', 
                              'x5', 'x6', 'x7', 'x8', 'x9'])
print(df)