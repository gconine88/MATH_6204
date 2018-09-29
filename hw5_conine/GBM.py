#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:13:14 2017

@author: Grant Conine

Function to simulate Geometric Brownian Motion via Euler discretization.
Capable of computing multiple stock price processes at once.
"""

import numpy as np

def generate_paths(S0, mu, delta, sigma, T, delta_t, I):
    '''Function to simulate geometric Brownian motion.

    Parameters
    ==========
    S0 = float
        Initial value of stock process
    mu = float
        Drift of stock process
    delta = float
        Continuous dividend yield
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
        W = ran[t] * np.sqrt(delta_t) # create random noise component
        
        # Use Euler discretization to simulate process
        x[t] = x[(t-1)] + (mu - delta) * x[(t-1)] * delta_t + sigma * x[(t-1)] * W
        
    return x