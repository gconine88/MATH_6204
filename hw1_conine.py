# -*- coding: utf-8 -*-
# Black-Scholes Merton (1973) European Call and Put Valuation
#
# Author: Grant Conine
# Date: 2017-09-01
#
# UNC Charlotte MS Math Finance
# MATH 6204 Numerical Methods for Financial Derivatives
# Prof Hwan Lin
# 
""" This is a set of functions to compute the Black-Schole Merton (1973) value
of a European call and European put. 

It utilizes an approximation for the normal CDF based upon work by C. Hastings 
Jr in "Approximations for Digital computers". Princton Univ, Princton, NJ, 
(1955). The author of this code also consulted examples by Yves Hilpisch in 
"Derivative Analytics for Python". Wiley, West Sussex, UK (2015).

These functions compute the exact value, using no approximation other than
that for the normal CDF.
"""

import math
import numpy as np
import scipy as sp

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
# BSM Functions
#

def BSM_call(S=100, t=0):
    ''' Calculates the Black Scholes Merton (1973) European call option value
    
    Parameters:
    ===========
    S: float
        The current equity/stock price
    t: float
        The current/valuation time
            
    Returns:
    ========
    call_value: array
        European call value at time t
    '''
    
    
    K = 100 # stike price
    r = 0.05 # risk-free rate of return
    delta = 0.025 # continuous dividend yield
    sigma = np.arange(0.1, 1.1, 0.1) # volatility
    T = 1 # expiration
    
    delT = T - t # delta t
    
    d1 = d1f(S, K, r, delta, sigma, delT)
    d2 = d2f(S, K, r, delta, sigma, delT)
    
    call_value = (S * np.exp(-delta * delT) * dN(d1) 
                  - K * np.exp(-r * delT) * dN(d2))
    
    return call_value

def BSM_put(S=100, t=0):
    ''' Calculates the Black Scholes Merton (1973) European put option value
    
    Parameters:
    ===========
    S: float
        The current equity/stock price
    t: float
        The current/valuation time
            
    Returns:
    ========
    put_value: array
        European put value at time t
    '''
    
    K = 100 # strike price
    r = 0.05 # risk-free rate of return
    delta = 0.025 # continuous dividend yield
    sigma = np.arange(0.1, 1.1, 0.1) # volatility
    T = 1 # expirayion
    
    delT = T - t # delta t
    
    d1 = d1f(S, K, r, delta, sigma, delT)
    d2 = d2f(S, K, r, delta, sigma, delT)
    
    put_value = (-S * np.exp(-delta * delT) * dN(-d1) 
                 + K * np.exp(-r * delT) * dN(-d2))
    
    return put_value
    
BSM_call(100,0)
BSM_put(100,0)
