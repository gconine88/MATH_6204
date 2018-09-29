#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 29 16:22:47 2017

@author: Grant Conine

UNC Charlotte MS Math Finance
MATH 6204 Numerical Methods for Financial Derivatives
Prof Hwan Lin

This is a set of functions to compute the price for European call and put
options using Fourier transformations. This is the driver function for this
set of functions. 
"""
import numpy as np
import pandas as pd
import FourierOpt as fo

# Model Parameters
s0 = 100. # Initial value of stock price process
K = 80. # Strike price of option
r = 0.05 # short rate
sigma = 0.50 # volatility
t0 = 0.0 # starting time
T = 1.0 # time of expiration

# Simulation Parameters
N = 1000 # Number of paths to simulate
B = 50 # upper bound for evaluation
h = B / float(N)

# compute call and put with damping parameter of 2.5. We compute both half and
# full frequency domains
alpha1 = 2.5
call_half1 = fo.fourierOpt_half(r, s0, K, alpha1, sigma, t0, T, h, N)
put_half1 = fo.fourierOpt_half(r, s0, K, -alpha1, sigma, t0, T, h, N)
call_full1 = fo.fourierOpt_full(r, s0, K, alpha1, sigma, t0, T, h, N)
put_full1 = fo.fourierOpt_full(r, s0, K, -alpha1, sigma, t0, T, h, N)

# compute call and put with damping parameter of 5. We compute both half and
# full frequency domains
alpha2 = 5
call_half2 = fo.fourierOpt_half(r, s0, K, alpha2, sigma, t0, T, h, N)
put_half2 = fo.fourierOpt_half(r, s0, K, -alpha2, sigma, t0, T, h, N)
call_full2 = fo.fourierOpt_full(r, s0, K, alpha2, sigma, t0, T, h, N)
put_full2 = fo.fourierOpt_full(r, s0, K, -alpha2, sigma, t0, T, h, N)

# compute call and put with damping parameter of 10. We compute both half and
# full frequency domains
alpha3 = 10
call_half3 = fo.fourierOpt_half(r, s0, K, alpha3, sigma, t0, T, h, N)
put_half3 = fo.fourierOpt_half(r, s0, K, -alpha3, sigma, t0, T, h, N)
call_full3 = fo.fourierOpt_full(r, s0, K, alpha3, sigma, t0, T, h, N)
put_full3 = fo.fourierOpt_full(r, s0, K, -alpha3, sigma, t0, T, h, N)

# Assemble output
d = {'Call Option (Half Frequency Domain)': [call_half1, call_half2, call_half3],
    'Put Option (Half Frequency Domain)': [put_half1, put_half2, put_half3],
    'Call Option (Full Frequency Domain)': [call_full1, call_full2, call_full3],
    'Put Option (Full Frequency Domain)': [put_full1, put_full2, put_full3]}

df = pd.DataFrame(d, index=['alpha = +/- 2.5',
                            'alpha = +/- 5',
                            'alpha = +/- 10'])

print(df)