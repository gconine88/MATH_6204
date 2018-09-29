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

This is a set of functions which derives the value of a European option/call
by both an exact Black-Scholes-Merton formula and Monte-Carlo simulation, and 
compares the results.

This uses Euler discretization to estimate the path of the geometric Brownian
motion.
"""

import numpy as np
import math
import scipy as sp
import pandas as pd

#
# Helper Functions
#

def N(x):
    '''Probability Density Function of a standard normal random variable.'''
    
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * math.pi)

def dN(x):
    ''' Cumulative Normal Density Function of a standard normal random variable
    using continued fractions expansion of the normal CDF.
    '''
    
    
    ones = sp.less(x, 0) # determine xi <0 
    negs = -1 * ones + ~ones # create vector of -1 and 1 to reverse sign of F(|x|)
    x = abs(x)
    
    # Set the variables here for ease of reading later
    f = N(x)
    z = 1 / (1 + 0.2316419 * x)
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    
    F = 1 - f * z * ((((a5 * z + a4) * z + a3) * z + a2) * z + a1)
    F = ones + negs * F # if xi <0 F(xi) = 1 - F(-xi)
    
    return F

def d1f(S, K, r, delta, sigma, delT):
    ''' BSM d1 function. '''
    
    d1 = (math.log(S / K) + (r - delta + 0.5 * sigma ** 2) 
            * delT) / (sigma * math.sqrt(delT))
    
    return d1

def d2f(S, K, r, delta, sigma, delT):
    '''BSM d2 function.'''
    
    d1 = d1f(S, K, r, delta, sigma, delT)
    
    d2 = d1 - sigma * math.sqrt(delT)
    
    return d2

#
# Monte Carlo Functions
#

def GBM_generate_paths(S0, mu, delta, sigma, T, delta_t, I):
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
        W = ran[t] * math.sqrt(delta_t)
        x[t] = x[(t-1)] + (mu - delta) * x[(t-1)] * delta_t + sigma * x[(t-1)] * W
        
    return x


def MC_BSM_call(S0, K, mu, r, delta, sigma, T, delta_t, N):
    '''
    Parameters
    ==========
    S0: float
        Initial value of stock process
    mu: float
        Drift of stock process
    r: float
        Risk Free rate
    delta: float
        Continuous dividend yield
    sigma: float
        Volatility of stock process
    T: float
        Time horizon
    delta_t: float
        Time step size
    N: integer
        Number of paths to generate
    
    Returns
    =======
    C: Float
        Value of Call option
    '''
    
    M = int(T / delta_t) # number of steps. Must be an integer.
    
    np.random.seed(seed=10)
    S = GBM_generate_paths(S0, mu, delta, sigma, T, delta_t, N)
    
    C = round(np.exp(- r * T) * sum(np.maximum(S[M] - K, 0)) / N, 6)
    
    return C

def MC_BSM_put(S0, K, mu, r, delta, sigma, T, delta_t, N):
    '''
    Parameters
    ==========
    S0: float
        Initial value of stock process
    mu: float
        Drift of stock process
    r: float
        Risk Free rate
    delta: float
        Continuous dividend yield
    sigma: float
        Volatility of stock process
    T: float
        Time horizon
    delta_t: float
        Time step size
    N: integer
        Number of paths to generate
    
    Returns
    =======
    C: Float
        Value of put option
    '''
    
    M = int(T / delta_t) # number of steps. Must be an integer.
    
    np.random.seed(seed=10)
    S = GBM_generate_paths(S0, mu, delta, sigma, T, delta_t, N)
    
    P = round(np.exp(- r * T) * sum(np.maximum(K - S[M], 0)) / N, 6)
    
    return P

#
# BSM Functions
#

def BSM_call(S, K, r, delta, sigma, T, t=0):
    ''' Calculates the Black Scholes Merton (1973) European call option value
    
    Parameters:
    ===========
    S: float
        The current equity/stock price
    K: float
        The strike price of the option
    r: float
        The risk-free rate
    delta: float
        The continuous dividend yield
    sigma: float
        The volatility of the equity
    T: float
        Time of expiry
    t: float
        The current/valuation time
            
    Returns:
    ========
    call_value: array
        European call value at time t
    '''
    
    delT = T - t # delta t
    
    d1 = d1f(S, K, r, delta, sigma, delT)
    d2 = d2f(S, K, r, delta, sigma, delT)
    
    call_value = (S * np.exp(-delta * delT) * dN(d1) 
                  - K * np.exp(-r * delT) * dN(d2))
    
    return call_value

def BSM_put(S, K, r, delta, sigma, T, t=0):
    ''' Calculates the Black Scholes Merton (1973) European put option value
    
    Parameters:
    ===========
    S: float
        The current equity/stock price
    K: float
        The strike price of the option
    r: float
        The risk-free rate
    delta: float
        The continuous dividend yield
    sigma: float
        The volatility of the equity
    T: float
        Time of expiry
    t: float
        The current/valuation time
            
    Returns:
    ========
    put_value: array
        European put value at time t
    '''
    
    delT = T - t # delta t
    
    d1 = d1f(S, K, r, delta, sigma, delT)
    d2 = d2f(S, K, r, delta, sigma, delT)
    
    put_value = (-S * np.exp(-delta * delT) * dN(-d1) 
                 + K * np.exp(-r * delT) * dN(-d2))
    
    return put_value

call_tru = BSM_call(100, 100, 0.03, 0.025, 0.75, 1, 0)
call_mc1 = MC_BSM_call(100, 100, 0.08, 0.03, 0.025, 0.75, 1, 0.01, 1000)
e11 = round(np.abs(call_mc1 - call_tru),6)
call_mc2 = MC_BSM_call(100, 100, 0.08, 0.03, 0.025, 0.75, 1, 0.001, 1000)
e12 = round(np.abs(call_mc2 - call_tru),6)

put_tru = BSM_put(100, 100, 0.03, 0.025, 0.75, 1, 0)
put_mc1 = MC_BSM_put(100, 100, 0.08, 0.03, 0.025, 0.75, 1, 0.01, 1000)
e21 = round(np.abs(put_mc1 - put_tru), 6)
put_mc2 = MC_BSM_put(100, 100, 0.08, 0.03, 0.025, 0.75, 1, 0.001, 1000)
e22 = round(np.abs(put_mc2 - put_tru), 6)

d = {'Call Option': [call_tru, call_mc1, e11, call_mc2, e12],
    'Put Option': [put_tru, put_mc1, e21, put_mc2, e22]}
df = pd.DataFrame(d, index=['BSM Value', 
                            'MC Value (delta_t = 0.01)', 'Error (delta_t = 0.01)',
                            'MC Value (delta_t = 0.001)', 'Error (delta_t = 0.001)'])
print(df)