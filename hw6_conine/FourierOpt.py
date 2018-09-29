#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:30:48 2017

@author: Grant Conine

Set of functions to compute value of a call and put options using Fourier 
transforms
"""
import numpy as np
import scipy as sp
from numba import jit

@jit
def nu_hat(r, x0, t0, T, sigma, alpha, omega):
    ''' Computes the inverse Fourier transform of the normalized option price.
    '''
    omega2 = omega + (alpha + 1)*1j # compute omega prime
    t = T - t0
    
    out = ((np.exp(-r*(T-t0)) * q_hat(omega2, x0, r, sigma, T - t0))/
           ((alpha - 1j * omega)*(alpha - 1j * omega + 1)))
    return out

@jit
def q_hat(omega, x0, r, sigma, t):
    ''' Computes the complex conjugate of the characteristic function of 
    normal distribution.
    '''
    
    out = np.exp(-1j * (x0 + (r - sigma**2/2)*t) * omega - 
                 sigma**2 * t/2 * omega**2)
    return out

@jit
def delta_wm(m,N,h, type):
    ''' Helper function for Fourier summand - the delta omega_m.'''
    if type=="half":
        if m == 0:
            return h/2
        if m == N:
            return h/2
        else:
            return h 
    
    if type=="full":
        if m == -N:
            return h/2
        if m == N:
            return h/2
        else:
            return h
        
@jit       
def fourier_summand(r, x0, k, alpha, sigma, t0, T, m, h, N, type):
    ''' Helper function for Fourier option pricing. Computes interior of 
    the summation in the discreetized inverse Fourier transform.
    '''
    
    omega_m = h*m #find omega_m for this step
    return (np.exp(1j * omega_m * k) * 
                  nu_hat(r, x0, t0, T, sigma, alpha, omega_m) * 
                  delta_wm(m,N,h,type))

@jit
def fourierOpt_half(r, s0, K, alpha, sigma, t0, T, h, N):
    ''' Compute Fourier transform option price using the half frequency domain
    Inputs
    ========
    r: float
        risk-free short rate
    s0: float
        starting price of underlying equity
    K: float
        option strike price
    alpha: float
        damping factor - determines call or put option
    sigma: float
        variance of equity
    t0: float
        starting time
    T: float
        time of expiration/excercise
        
    Returns
    ========
    Value of option given parameters
    '''
    x0 = np.log(s0)
    k = np.log(K)
    
    summation = 0
    # For loop to sum. Could also vectorize
    for m in xrange(0,N,1):
        next_sum = fourier_summand(r,x0, k, alpha, sigma, t0, T, m, h, N, "half")
        summation = summation + next_sum
    
    return (np.exp(-alpha*k)/np.pi) * np.real(summation)
    
@jit
def fourierOpt_full(r, s0, K, alpha, sigma, t0, T, h, N):
    ''' Compute Fourier transform option price using the half frequency domain
    Inputs
    ========
    r: float
        risk-free short rate
    s0: float
        starting price of underlying equity
    K: float
        option strike price
    alpha: float
        damping factor - determines call or put option
    sigma: float
        variance of equity
    t0: float
        starting time
    T: float
        time of expiration/excercise
        
    Returns
    ========
    Value of option given parameters
    '''
    x0 = np.log(s0)
    k = np.log(K)
    
    summation = 0
    for m in xrange(-N,N,1):
        next_sum = fourier_summand(r,x0, k, alpha, sigma, t0, T, m, h, N, "full")
        summation = summation + next_sum
    
    # Include real only because of rounding errors I am getting.
    return (np.exp(-alpha*k)/(2*np.pi)) * np.real(summation)
