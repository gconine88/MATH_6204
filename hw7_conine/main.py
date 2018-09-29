#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thursday Nov 2 17:45:47 2017

@author: Grant Conine

UNC Charlotte MS Math Finance
MATH 6204 Numerical Methods for Financial Derivatives
Prof Hwan Lin

This is a set of functions to compute the price for European call and put
options using Fourier transformations and the FFT algorithm. This is the
driver function for this set of functions.

Written for Python 3.6
"""

import numpy as np
import pandas as pd
import scipy.interpolate as interp
import FFT_opt as ffto

# Model Parameters
s0 = 100.  # Initial value of stock price process
K = 80.  # Strike price of option
r = 0.05  # short rate
sigma = 0.50  # volatility
t0 = 0.0  # starting time
T = 1.0  # time of expiration

# Simulation Parameters
N = 2**10  # Number of paths to simulate
B = 50  # upper bound for evaluation
h = B / float(N - 1)

m = np.arange(0, N, 1)  # m = 0,1,...,N-1
omega = m * h  # frequency grind for FFT

delta_k = 2*np.pi/(h * N)
k_min = np.log(20)
k_list = k_min + m*delta_k  # log strike grid for option

# compute call and put option prices
alpha1 = 2.5
call1 = ffto.price(r, s0, k_list, omega, alpha1, sigma, t0, T, h, N)
put1 = ffto.price(r, s0, k_list, omega, -alpha1, sigma, t0, T, h, N)

alpha2 = 5
call2 = ffto.price(r, s0, k_list, omega, alpha2, sigma, t0, T, h, N)
put2 = ffto.price(r, s0, k_list, omega, -alpha2, sigma, t0, T, h, N)

alpha3 = 10
call3 = ffto.price(r, s0, k_list, omega, alpha3, sigma, t0, T, h, N)
put3 = ffto.price(r, s0, k_list, omega, -alpha3, sigma, t0, T, h, N)

# interpolate
call1_int = interp.interp1d(k_list, call1)
call2_int = interp.interp1d(k_list, call2)
call3_int = interp.interp1d(k_list, call3)

put1_int = interp.interp1d(k_list, put1)
put2_int = interp.interp1d(k_list, put2)
put3_int = interp.interp1d(k_list, put3)

k = np.log(K)  # log-value of strike put option

# graph strike vs option prices
ffto.plot(k_list, call1, "Call option, alpha = 2.5", 1)
ffto.plot(k_list, call2, "Call option, alpha = 5", 2)
ffto.plot(k_list, call3, "Call option, alpha = 10", 3)

ffto.plot(k_list, put1, "Put option, alpha = -2.5", 4)
ffto.plot(k_list, put2, "Put option, alpha = -5", 5)
ffto.plot(k_list, put3, "Put option, alpha = -10", 6)

# Assemble output
d = {'Call Option': [call1_int(k), call2_int(k), call3_int(k)],
     'Put Option': [put1_int(k), put2_int(k), put3_int(k)]}

df = pd.DataFrame(d, index=['alpha = +/- 2.5',
                            'alpha = +/- 5',
                            'alpha = +/- 10'])

print(df)
