# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 16:54:41 2017

@author: Grant

Compute American and European option values using monte carlo methods.
"""
import numpy as np
import math
from numba import jit, f8, int8
from scipy.optimize import curve_fit as fit
    
@jit
def generate_paths(S, mu, d, sigma, delta_t, N, M):
    '''Function to simulate geometric Brownian motion.

    Parameters
    ==========
    S0 = float
        Initial value of stock process
    mu = float
        Drift of stock process
    d = float
        Continuous dividend yield
    sigma = float
        Volatility of stock process
    t = float
        Current time
    T  = float
        Time horizon (expiration of option)
    delta_t = float
        Time step size
    N  = integer
        Number of paths to generate

    Returns
    =======
    x = NumPy array
        simulated paths of stock process
    '''
    
    x = np.zeros((M + 1, N)) # initialize output array
    x[0] = S
    
    rand = np.random.standard_normal((M + 1, N)) # prepare random numbers
    
    x = gen_iter(x, rand, delta_t, mu, d, sigma, M)
        
    return x

@jit
def gen_iter(x, rand, delta_t, mu, d, sigma, M):
    ''' Helper function for iterating generator. I house it seperately so 
    it is easier to use @njit decorator since the calculation of these paths 
    will take up the majority of the computing time for this function
    '''
    
    for t in range(1, M+1):
        W = rand[t, :] * math.sqrt(delta_t) #random noise component
        
        # Euler discretization
        x[t, :] = (x[(t-1), :]
                + (mu - d) * x[(t-1), :] * delta_t
                + sigma * x[(t-1), :] * W)
        
    return x

def func(x, a, b, c, d):
    ''' Helper function - represents third-degree polynomial.'''
    return a + b*x + c*(x**2) + d* (x**3)

def MC_Amer_call(S0, K, mu, r, d, sigma, t, T, delta_t, N):
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
        Value of American call option
    '''
    
    M = int((T - t) / delta_t) # number of steps. Must be an integer.
    
    S = generate_paths(S0, mu, d, sigma, delta_t, N, M)
    
    h = np.maximum(S - K, 0) #compute exercise values
    g = h[M] #set up excercised value vector
    tau = np.repeat(T, N) #set up stopping time vector
    
    for j in range(M-1, 0, -1):
        k = S[j] > K #in the money boolean vector
        x = S[j, k] # in the money points
        y = g[k] * np.exp(-(r - d) * (tau[k] - t * delta_t))
        
        a, __ = fit(func, x, y) #regression step
        C_hat = func(S[j,k], a[0], a[1], a[2], a[3]) #find estimated continuation value
        
        g[k][C_hat >= h[j, k]] = h[j, k][C_hat >= h[j, k]] #update g where excercise more than continuation
        tau[k][C_hat >= h[j, k]] = j/f8(M) * (T - t) #update optimal excercise time
    
    C_0 = np.sum(np.exp(-(r - d) * delta_t * tau).T * g) / N
    V_0 = np.maximum(C_0, h[0,0])
    
    return V_0

def MC_Amer_put(S0, K, mu, r, d, sigma, t, T, delta_t, N):
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
        Value of American put option
    '''
    
    M = int((T - t) / delta_t) # number of steps. Must be an integer.
    
    S = generate_paths(S0, mu, d, sigma, delta_t, N, M)
    
    h = np.maximum(K - S, 0) #compute exercise values
    g = h[M] #set up excercised value vector
    tau = np.repeat(T, N) #set up stopping time vector
    
    for j in range(M-1, 0, -1):
        k = S[j] < K #in the money boolean vector
        x = S[j, k] # in the money points
        y = g[k] * np.exp(-(r - d) * (tau[k] - t * delta_t))
        
        a, __ = fit(func, x, y) #regression step
        C_hat = func(S[j,k], a[0], a[1], a[2], a[3]) #find estimated continuation value
        
        g[k][C_hat >= h[j, k]] = h[j, k][C_hat >= h[j, k]] #update g where excercise more than continuation
        tau[k][C_hat >= h[j, k]] = j/f8(M) * (T - t) #update optimal excercise time
    
    C_0 = np.sum(np.exp(-(r - d) * delta_t * tau).T * g) / N
    V_0 = np.maximum(C_0, h[0,0])
    
    return V_0      

def MC_Euro_call(S, K, mu, r, d, sigma, t, T, delta_t, N):
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
    
    S = generate_paths(S, mu, d, sigma, delta_t, N, M)
    
    C = round(np.exp(- r * T) * sum(np.maximum(S[M] - K, 0)) / N, 6)
    
    return C

def MC_Euro_put(S, K, mu, r, d, sigma, t, T, delta_t, N):
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
    
    S = generate_paths(S, mu, d, sigma, delta_t, N, M)
    
    P = round(np.exp(- r * T) * sum(np.maximum(K - S[M], 0)) / N, 6)
    
    return P