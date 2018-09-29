#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 15 16:22:47 2017

@author: Grant Conine

UNC Charlotte MS Math Finance
MATH 6204 Numerical Methods for Financial Derivatives
Prof Hwan Lin

This is a set of functions to compute the price for American call and put 
options assuming a stock price process with geometric brownian motion.  
"""

import GBM
import amerOpt
import numpy as np
import pandas as pd

# Model Parameters
S0 = 100 # Initial value of stock price process
K = 100 # Strike price of option
mu = 0.08 # average return of stock price process
r = 0.03 # short rate
delta = 0.025 # dividend yield
sigma = 0.75 # volatility
T = 1.0 # time-to-maturity

# Simulation Parameters
N = 1000 # Number of paths to simulate
delta_t = 0.01 # time step size
df = np.exp(-r * delta_t)

# Stock price paths
np.random.seed(seed=123) #set seed
S = GBM.generate_paths(S0, mu, delta, sigma, T, delta_t, N)

call1 = round(amerOpt.call(S, K, df), 6)
put1 = round(amerOpt.put(S, K, df), 6)

# Simulation Parameters
N = 1000 # Number of paths to simulate
delta_t = 0.001 # time step size
df = np.exp(-r * delta_t)

# Stock price paths
np.random.seed(seed=123) #set seed
S = GBM.generate_paths(S0, mu, delta, sigma, T, delta_t, N)

call2 = round(amerOpt.call(S, K, df), 6)
put2 = round(amerOpt.put(S, K, df), 6)

d = {'American Call Option': [call1, call2],
    'American Put Option': [put1, put2]}
df = pd.DataFrame(d, index=['MC Value (delta_t = 0.01)',
                            'MC Value (delta_t = 0.001)'])
print(df)