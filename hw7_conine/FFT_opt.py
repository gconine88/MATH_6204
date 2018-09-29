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
set of functions to perform FFT option pricing.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit
def price(r, s0, k_list, omega, alpha, sigma, t0, T, h, N):
    """Compute price of option using Fourier Fast Transform (FFT) algorithm.

    Parameters
    ==========
    r: float
        Short rate
    s0: float
        Initial value of underlying security
    k_list: np.array
        Grid of log-strike values
    omega: np.array
        Grid of frequency values for Fourier transform.
    alpha: float
        Damping factor. Sign of alpha determines call or put option.
    sigma: float
        Volatility of underlying security
    t0: float
        Current time
    T: float
        Time of expiration
    h: float
        Step size for evaluation grid
    N: integer
        Number of simulations

    Returns
    =========
    np.array of option values
    """

    x0 = np.log(s0)
    k0 = k_list[0]

    A_vector = A(omega, alpha, x0, k0, sigma, h, r, t0, T, N)
    a = np.fft.ifft(A_vector)

    V = np.exp(-alpha * k_list) / np.pi * np.real(a)

    return V


@jit
def A(omega, alpha, x0, k0, sigma, h, r, t0, T, N):
    """Computes the A vector for option pricing. Serving as a helper
    function so that we do not have many calculations crowing the option
    pricing function.

    Parameters
    ==========
    omega: np.array
        Grid of frequency values for Fourier transform.
    alpha: float
        Damping factor. Sign of alpha determines call or put option.
    x0: float
        Log-value of underlying security
    k0: float
        Value of strike of option.
    sigma: float
        Volatility of underlying security
    h: float
        Step size for evaluation grid
    r: float
        Short rate
    t0: float
        Current time
    T: float
        Time of expiration
    N: integer
        Number of simulations

    Returns
    ========
    Returns np.array of the values of A at different values of omega
    """

    nu_vector = nu(omega, alpha, x0, sigma, h, r, t0, T, N)

    A = np.exp(1j * omega * k0) * nu_vector * h * N
    A[0] = A[0] * 0.5

    return A


@jit
def nu(omega, alpha, x0, sigma, h, r, t0, T, N):
    """ Computes the nu hat function for option pricing. Serving as a helper
    function so that we do not have many calculations crowing the option
    pricing function.

    Parameters
    ==========
    omega: np.array
        Grid of frequency values for Fourier transform.
    alpha: float
        Damping factor. Sign of alpha determines call or put option.
    x0: float
        Log-value of underlying security
    k0: float
        Value of strike of option.
    sigma: float
        Volatility of underlying security
    h: float
        Step size for evaluation grid
    r: float
        Short rate
    t0: float
        Current time
    T: float
        Time of expiration
    N: integer
        Number of simulations

    Returns
    ========
    Returns np.array of the values of nu_hat at different values of omega
    """

    omega2 = omega + (alpha + 1) * 1j  # alternate omega for computation
    t = T - t0

    nu_hat = (np.exp(-r * t) * q(omega2, x0, r, sigma, t)) / ((alpha - omega * 1j) * (alpha - omega * 1j + 1))

    return nu_hat


@jit
def q(omega, x0, r, sigma, t):
    """ Computes complex conjugate of characteristic function of option price
    """

    q_hat = np.exp(-1j * (x0 + (r - sigma ** 2 / 2) * t) * omega -
                   sigma ** 2 * t / 2 * omega ** 2)
    return q_hat


@jit
def plot(k, opt, title, i):
    """Function to plot strike prices vs option price for various options.

    Parameters
    ===========
    opt: np.array
        Array of the option values
    k: np.array
        Array of strike log-prices
    title: str
        text for title
    """

    K = np.exp(k)  # convert back to strike price from log-price

    plt.figure(i)
    plt.plot(K, opt)
    plt.grid()
    plt.xlim([20, 180])  # need to restrict window so we only get reasonable option prices
    plt.ylim([0, 80])    # otherwise we will have some prices/strike prices far too high
    plt.xlabel('Strike price K')  # labeling
    plt.ylabel('Option Price')
    plt.title(title)
    plt.show()
