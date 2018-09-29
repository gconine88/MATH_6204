# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 22:11:43 2017

@author: Grant
"""

import numpy as np
import brennan_schwartz as bsa
import projected_sor as psor
from scipy.sparse import spdiags

class option:
    '''Computes options pricing using a variety of finite differencing 
    methods.
    
     Attributes
     ============
     S: float
         initial value of equity
     K: float
         strike price of option
     T: float
         expiration time
     t: float
         current time
     r: float
         short rate
     d: float
         dividend yield
     vol: float
         volatility of equity
    '''
    
    def __init__(self, S0, K, T, t, r, d, vol, xmin, xmax, xstep, tstep):
        '''Initialize option object'''
        self.S0 = float(S0)
        self.K = float(K)
        self.T = T
        self.t = t
        self.tau = (vol ** 2) * (T - t) / 2 #convert
        self.q = 2 * r / (vol ** 2)
        self.q_d = 2 * (r - d) / (vol ** 2)
        self.r = r
        self.d = d
        self.vol = vol
        self.xmin = xmin
        self.xmax = xmax
        self.xstep = xstep
        self.tstep = tstep

        
    def put_explicit(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w & v matrix for FDM
        w = np.zeros((N-1, M))
            
        #set up initial conditions vector
        b = np.zeros((N-1)) 
        
        # create g matrix
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d - 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d + 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        
        # tau loop. Work backward to initial time 
        for i in range(0, M-1):
            b[0] = w[0, i] + lamb * (g[0, i] - 2 * w[0, i] + w[1, i])
            b[N-2] = w[N-2, i] + lamb *(w[N-3, i] - 
                                             2 * w[N-2, i] + g[N, i])
            for j in range(1,N-2):
                b[j] = w[j, i] + lamb * (w[j-1, i] - 2 * w[j, i] + w[j+1, i])
                
            w[:, i+1] = np.maximum(b, g[1:N, i])
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]
    
    def put_implicit_bsa(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w & v matrix for FDM
        w = np.zeros((N-1, M))
        
        # set up A matrix for explicit FDM
        maindiag = (1 + 2 * lamb) * np.ones(N-1)
        subdiag = -lamb * np.ones(N-1)
        supdiag = -lamb * np.ones(N-1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N-1, N-1).toarray()
        
        # create g matrix
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d - 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d + 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        
        for i in range(0, M-1):
            w[:, i+1] = bsa.solve(A,w[:,i], g[1:N, i])
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]
    
    def put_cn_bsa(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w & v matrix for FDM
        w = np.zeros((N-1, M))
            
        #set up initial conditions vector
        b = np.zeros((N-1)) 
        
        # set up A matrix for explicit FDM
        maindiag = (1 + lamb) * np.ones(N-1)
        subdiag = -0.5 * lamb * np.ones(N-1)
        supdiag = -0.5 * lamb * np.ones(N-1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N-1, N-1).toarray()
        
        # create g matrix, includes value of early excercise
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d - 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d + 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        
        for i in range(0, M-1):
            # create b vector for Crank-Nicholson
            b[0] = (w[0, i] + 0.5 * lamb * (g[0, i] - 2 * w[0, i] + w[1, i])
                      + 0.5 * lamb * g[0,i+1])
            b[N-2] = (w[N-2, i] + lamb *(w[N-3, i] - 2 * w[N-2, i] + g[N, i])
                      + 0.5 * lamb * g[N, i+1])
            for j in range(1,N-2):
                b[j] = (w[j, i] 
                            + 0.5 * lamb * (w[j-1, i] - 2 * w[j, i] + w[j+1, i]))
            
            # Use Brennan-Schwartz to solve A w(i+1) >= b(i), A w(i+1) >= g(i)
            w[:, i+1] = bsa.solve(A, b, g[1:N, i])
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]
    
    def put_implicit_psor(self, omega, tol):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w & v matrix for FDM
        w = np.zeros((N-1, M))
        
        # create g matrix
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d - 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d + 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        b = np.zeros((N-1))
        
        for i in range(0, M-1):
            b[0] = w[0, i] + lamb * g[0, i+1]
            b[N-2] = w[N-2, i] + lamb * g[N, i+1]
            for j in range(1,N-2):
                b[j] = w[j, i]
                
            
            v0 = np.maximum(w[:,i], g[1:N, i+1])
            w[:, i+1] = psor.solve(v0, b, g[1:N, i+1], omega, tol, 1, lamb)
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]
    
    def put_cn_psor(self, omega, tol):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w & v matrix for FDM
        w = np.zeros((N-1, M))
            
        #set up initial conditions vector
        b = np.zeros((N-1)) 
        
        
        # create g matrix, includes value of early excercise
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d - 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d + 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        
        for i in range(0, M-1):
            # create b vector for Crank-Nicholson
            b[0] = (w[0, i] + 0.5 * lamb * (g[0, i] - 2 * w[0, i] + w[1, i])
                      + 0.5 * lamb * g[0,i+1])
            b[N-2] = (w[N-2, i] + lamb *(w[N-3, i] - 2 * w[N-2, i] + g[N, i])
                      + 0.5 * lamb * g[N, i+1])
            for j in range(1,N-2):
                b[j] = (w[j, i] 
                            + 0.5 * lamb * (w[j-1, i] - 2 * w[j, i] + w[j+1, i]))
            
            # Use projected SOR to solve A w(i+1) >= b(i), A w(i+1) >= g(i)
            v0 = np.maximum(w[:,i], g[1:N, i+1])
            w[:, i+1] = psor.solve(v0, b, g[1:N, i+1], omega, tol, 0.5, lamb)
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]
    
    def call_explicit(self):
        lamb = self.tstep / (self.xstep ** 2) 
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w & v matrix for FDM
        w = np.zeros((N-1, M))
            
        #set up initial conditions vector
        b = np.zeros((N-1))
        
        # create g matrix
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d + 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d - 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        
        # tau loop. Work backward to initial time 
        for i in range(0, M-1):
            b[0] = w[0, i] + lamb * (g[0, i] - 2 * w[0, i] + w[1, i])
            b[N-2] = w[N-2, i] + lamb *(w[N-3, i] - 
                                             2 * w[N-2, i] + g[N, i])
            for j in range(1,N-2):
                b[j] = w[j, i] + lamb * (w[j-1, i] - 2 * w[j, i] + w[j+1, i])
                
            w[:, i+1] = np.maximum(b, g[1:N, i ])
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]
    
    def call_implicit_bsa(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N-1, M))
        
        # set up A matrix for explicit FDM
        maindiag = (1 + 2 * lamb) * np.ones(N-1)
        subdiag = -lamb * np.ones(N-1)
        supdiag = -lamb * np.ones(N-1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N-1, N-1).toarray()
            
        # create g matrix
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d + 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d - 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        
        # tau loop. Work backward to initial time 
        for i in range(0, M-1):
            w[:, i+1] = bsa.solve(A,w[:,i], g[1:N, i])
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]
    
    def call_cn_bsa(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N-1, M))
        
        # set up A matrix for explicit FDM
        maindiag = (1 + lamb) * np.ones(N-1)
        subdiag = -0.5 * lamb * np.ones(N-1)
        supdiag = -0.5 * lamb * np.ones(N-1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N-1, N-1).toarray()
            
        #set up b vector 
        b = np.zeros((N-1))
        
        # create g matrix
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d + 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d - 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        
        # tau loop. Work backward to initial time
        for i in range(0, M-1):
        # create b vector for Crank-Nicholson
            b[0] = (w[0, i] + 0.5 * lamb * (g[0, i] - 2 * w[0, i] + w[1, i])
                      + 0.5 * lamb * g[0,i+1])
            b[N-2] = (w[N-2, i] + lamb *(w[N-3, i] - 2 * w[N-2, i] + g[N, i])
                      + 0.5 * lamb * g[N, i+1])
            for j in range(1,N-2):
                b[j] = (w[j, i] 
                            + 0.5 * lamb * (w[j-1, i] - 2 * w[j, i] + w[j+1, i]))
            
            # Use Brennan-Schwartz to solve A w(i+1) >= b(i), A w(i+1) >= g(i)
            w[:, i+1] = bsa.solve(A, b, g[1:N, i])
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]
    
    def call_implicit_psor(self, omega, tol):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N-1, M))
        
        # set up A matrix for explicit FDM
        maindiag = (1 + 2 * lamb) * np.ones(N-1)
        subdiag = -lamb * np.ones(N-1)
        supdiag = -lamb * np.ones(N-1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N-1, N-1).toarray()
            
        #set up b vector 
        b = np.zeros((N-1))
        
        # create g matrix
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d + 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d - 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        b = np.zeros((N-1))
        
        for i in range(0, M-1):
            b[0] = w[0, i] + lamb * g[0, i+1]
            b[N-2] = w[N-2, i] + lamb * g[N, i+1]
            for j in range(1,N-2):
                b[j] = w[j, i]
                
            
            v0 = np.maximum(w[:,i], g[1:N, i+1])
            w[:, i+1] = psor.solve(v0, b, g[1:N, i+1], omega, tol, 1, lamb)
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]
    
    def call_cn_psor(self, omega, tol):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N-1, M))
        
        # set up A matrix for explicit FDM
        maindiag = (1 + lamb) * np.ones(N-1)
        subdiag = -0.5 * lamb * np.ones(N-1)
        supdiag = -0.5 * lamb * np.ones(N-1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N-1, N-1).toarray()
            
        #set up b vector 
        b = np.zeros((N-1))
        
        # create g matrix
        g = np.zeros((N+1, M))
        for i in range(0, M):
            tau_i = i * self.tstep
            g[:,i] = (np.exp((0.25 * (self.q_d - 1) ** 2 + self.q) * tau_i) *
                         np.maximum(np.exp(0.5 * (self.q_d + 1) * xgrid) - 
                                    np.exp(0.5 * (self.q_d - 1) * xgrid), 0))
        
        w[:,0] = g[1:N,0]
        
        for i in range(0, M-1):
            # create b vector for Crank-Nicholson
            b[0] = (w[0, i] + 0.5 * lamb * (g[0, i] - 2 * w[0, i] + w[1, i])
                      + 0.5 * lamb * g[0,i+1])
            b[N-2] = (w[N-2, i] + lamb *(w[N-3, i] - 2 * w[N-2, i] + g[N, i])
                      + 0.5 * lamb * g[N, i+1])
            for j in range(1,N-2):
                b[j] = (w[j, i] 
                            + 0.5 * lamb * (w[j-1, i] - 2 * w[j, i] + w[j+1, i]))
            
            # Use projected SOR to solve A w(i+1) >= b(i), A w(i+1) >= g(i)
            v0 = np.maximum(w[:,i], g[1:N, i+1])
            w[:, i+1] = psor.solve(v0, b, g[1:N, i+1], omega, tol, 0.5, lamb)
    
        V = np.zeros((N-1,2))
        
        for i in range(1,N):
            V[i-1,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i-1,1] = (self.K * np.exp(-0.5 * (self.q_d - 1) * 
                         (self.xmin + (i * self.xstep)) -
                     (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) 
                         * w[i-1, M-1])
        
        return V[V[:,0] == self.S0,1][0]