# -*- coding: utf-8 -*-

#
# Author: Grant Conine
# Date: 2017-09-01
#
# UNC Charlotte MS Math Finance
# MATH 6204 Numerical Methods for Financial Derivatives
# Prof Hwan Lin
# 

"""
Created on Sun Sep 17 15:19:12 2017

@author: Grant Conine

UNC Charlotte MS Math Finance
MATH 6204 Numerical Methods for Financial Derivatives
Prof Hwan Lin

This is a set of functions which simulates the motion of a stock process
driven by geometric Brownian motion and graphs the paths taken by these 
processes. 

This uses Euler discretization to estimate the path of the geometric Brownian
motion.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def GBM_generate_paths(S0, mu, sigma, T, delta_t, I):
    '''Function to simulate geometric Brownian motion.

    Parameters
    ==========
    S0 = float
        Initial value of stock process
    mu = float
        Drift of stock process
    sigma = float
        Volatility of stock process
    T  = float
        Time horizon
    delta_t = float
        Time step size
    I  = integer
        Number of paths to generate

    Returns
    =======
    x = NumPy array
        simulated paths of stock process
    '''
    
    M = int(T / delta_t) # number of steps. Must be an integer.
    x = np.zeros((M + 1, I)) # initialize output array
    x[0] = S0
    ran = np.random.standard_normal((M + 1, I)) # prepare random numbers
    
    for t in xrange(1, M+1):
        W = ran[t] * math.sqrt(delta_t)
        x[t] = x[(t-1)] + mu * x[(t-1)] * delta_t + sigma * x[(t-1)] * W
        
    return x

def plot_paths(x):
    '''Function to plot paths from GBM_generate_paths() function.'''
    
    plt.figure(figsize=(27, 15))
    plt.plot(range(len(x)), x[:, :5])
    plt.grid()
    plt.xlabel('time (t)')
    plt.ylabel('S(t)')
    
np.random.seed(seed = 100)
s = GBM_generate_paths(100, 0.1, 0.2, 1, 0.001, 5)

plot_paths(s)
    