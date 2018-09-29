#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:30:48 2017

@author: Grant Conine

Set of functions to compute value of a call option given stock price processes
"""
import numpy as np
from scipy.optimize import curve_fit as curve_fit

def func(x, a, b, c, d):
    ''' Helper function - represents third-degree polynomial.'''
    return a + b*x + c*(x**2) + d* (x**3)

def call(S, K, r, delta_t):
    ''' Computes the American call option value given a stock price process
    and the discount factor.
    
    Parameters
    ============
    S = NumPy Array
        Simulated paths of a stock price process
        
    K = float
        Strike price of option
        
    r = float
        Short rate
    
    delta_t = float
        Size of time step
    '''
    
    [t,I] = S.shape
    
    # Compute inner values (excercise values) at all points 
    h = np.maximum(S - K, 0)
    
    # Initialize variables
    g = h[-1] # Final value at expiration
    tau = np.repeat(t, I) #stopping time
    C = np.zeros(I) # continuation value
    df = np.zeros(I) # discount factor
    
    # want to use these as indices, must reduce by one because of how python indexes
    t = t-1
    I = I-1
    
    # Step backwards, computing excercise vs continuation values
    # NOTE: Would prefer np.polyfit over sp.optimize.curve_fit
    for i in xrange(t-1, 0, -1):
        df = np.exp(-r * (tau - i) * delta_t) # create discount vector
        b = S[i] > K # boolean vector, which do we need to update?
        para,_ = curve_fit(func, S[i,b], df[b] * g[b]) #regression
        C[b] = func(S[i,b], para[0], para[1], para[2], para[3]) #continuation values
        g[b] = np.where(h[i,b] >= C[b], h[i,b], g[b]) #update g when we would excercise
        tau[b] = np.where(h[i,b] >= C[b], i, tau[b]) #update tau when we would excercise
        
    df = np.exp(-r * (tau) * delta_t)
    C0 = np.sum(df * g) / I # Continuation value at time 0
    V0 = np.maximum(C0, h[0,0])
    
    return V0
    
def put(S, K, r, delta_t):
    ''' Computes the American put option value given a stock price process
    and the discount factor.
    
    Parameters
    ============
    S = NumPy Array
        Simulated paths of a stock price process
        
    K = float
        Strike price of option
        
    r = float
        Short rate
    
    delta_t = float
        Size of time step
    '''
    
    [t,I] = S.shape
    
    # Compute inner values (excercise values) at all points 
    h = np.maximum(K - S, 0)
    
     # Initialize variables
    g = h[-1] # Final value at expiration
    tau = np.repeat(t, I) #stopping time
    C = np.zeros(I) # continuation value
    df = np.zeros(I) # discount factor
    
    # want to use these as indices, must reduce by one because of how python indexes
    t = t-1
    I = I-1
    
    # Step backwards, computing excercise vs continuation values
    # NOTE: Would prefer np.polyfit over sp.optimize.curve_fit
    for i in xrange(t-1, 0, -1):
        df = np.exp(-r * (tau - i) * delta_t) # create discount vector
        b = S[i] > K # boolean vector, which do we need to update?
        para,_ = curve_fit(func, S[i,b], df[b] * g[b]) #regression
        C[b] = func(S[i,b], para[0], para[1], para[2], para[3]) #continuation values
        g[b] = np.where(h[i,b] >= C[b], h[i,b], g[b]) #update g when we would excercise
        tau[b] = np.where(h[i,b] >= C[b], i, tau[b]) #update tau when we would excercise
        
    df = np.exp(-r * (tau) * delta_t)
    C0 = np.sum(df * g) / I # Continuation value at time 0
    V0 = np.maximum(C0, h[0,0])
    
    return V0
