#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:30:48 2017

@author: Grant Conine

Set of functions to compute value of a call option given stock price processes
"""
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit as curve_fit

def func(x, a, b, c, d):
    ''' Helper function - represents third-degree polynomial.'''
    return a + b*x + c*(x**2) + d* (x**3)

def call(S, K, df):
    ''' Computes the American call option value given a stock price process
    and the discount factor.
    
    Parameters
    ============
    S = NumPy Array
        Simulated paths of a stock price process
        
    K = float
        Strike price of option
        
    df = float
         Discount factor for the time step and short rate assumed for model
    '''
    
    [t,I] = S.shape
    t = t - 1 #reduce for looping later (shape gives size, but loop starts at 0)
    I = I - 1 #reduce for looping later (shape gives size, but loop starts at 0)
    
    # Compute inner values (excercise values) at all points 
    h = np.maximum(S - K, 0)
    
    # Initialize values of option (including at expiration)
    V = h[-1]
    
    # Step backwards, computing excercise vs continuation values
    # NOTE: Would prefer np.polyfit over sp.optimize.curve_fit
    for i in xrange(t-1, 0, -1):
        para,_ = curve_fit(func, S[i], df * V) #regression
        C = func(S[i], para[0], para[1], para[2], para[3]) #continuation values
        V = np.maximum(h[i],C)
        
    V0 = df * np.sum(V) / I # Least Square estimator of call value
    
    return V0
    
def put(S, K, df):
    ''' Computes the American put option value given a stock price process
    and the discount factor.
    
    Parameters
    ============
    S = NumPy Array
        Simulated paths of a stock price process
        
    K = float
        Strike price of option
        
    df = float
         Discount factor for the time step and short rate assumed for model
    '''
    
    [t,I] = S.shape
    t = t - 1 #reduce for looping later (shape gives size, but loop starts at 0)
    I = I - 1 #reduce for looping later (shape gives size, but loop starts at 0)
    
    # Compute inner values (excercise values) at all points 
    h = np.maximum(K - S, 0)
    
    # Initialize values of option (at expiration)
    V = h[-1]
    
    # Step backwards, computing excercise vs continuation values
    # NOTE: Would prefer np.polyfit over sp.optimize.curve_fit
    for i in xrange(t-1, 0, -1):
        para,_ = curve_fit(func, S[i], df * V) #regression
        C = func(S[i], para[0], para[1], para[2], para[3]) #continuation values
        V = np.maximum(h[i],C)
        
    V0 = df * np.sum(V) / I # Least Square estimator of call value
    
    return V0
