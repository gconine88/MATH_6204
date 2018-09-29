# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 16:03:31 2017

@author: Grant

"""

import numpy as np
import thomas
import sor
from scipy.sparse import spdiags


# Explicit FDM

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
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d - 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d + 1)),0)
            
        #set up initial conditions vector
        bounds = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 - 2 * lamb) * np.ones(N+1)
        subdiag = lamb * np.ones(N+1)
        supdiag = lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            tau_j = j * self.tstep
            bounds[0] = lamb * np.exp(0.5 * (self.q_d - 1) * self.xmin +
                          (0.25 * (self.q_d - 1) ** 2) * tau_j) 
            w[:, j+1] = np.matmul(A, w[:, j]) + bounds
          
    
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
    
    def put_implicit_thomas(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d - 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d + 1)),0)
            
        #set up initial conditions vector
        bounds = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 + 2 * lamb) * np.ones(N+1)
        subdiag = -lamb * np.ones(N+1)
        supdiag = -lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            tau_j = (j + 1) * self.tstep
            bounds[0] = -lamb * np.exp(0.5 * (self.q_d - 1) * self.xmin +
                          (0.25 * (self.q_d - 1) ** 2) * tau_j) 
            w[:, j+1] = thomas.solve(A, w[:, j] - bounds)
          
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
    
    def put_implicit_sor(self, omega, tol):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d - 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d + 1)),0)
            
        #set up initial conditions vector
        bounds = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 + 2 * lamb) * np.ones(N+1)
        subdiag = -lamb * np.ones(N+1)
        supdiag = -lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            tau_j = (j + 1) * self.tstep
            bounds[0] = -lamb * np.exp(0.5 * (self.q_d - 1) * self.xmin +
                          (0.25 * (self.q_d - 1) ** 2) * tau_j) 
            w[:, j+1] = sor.solve(w[:,j], A, w[:, j] - bounds, omega, tol)
          
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
    
    def put_cn_thomas(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d - 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d + 1)),0)
            
        #set up initial conditions vector
        boundsL = np.zeros((N+1))
        boundsR = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 +  lamb) * np.ones(N+1)
        subdiag = -0.5 * lamb * np.ones(N+1)
        supdiag = -0.5 * lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        AL = spdiags(data, triset, N+1, N+1).toarray()
        
        maindiag = (1 - lamb) * np.ones(N+1)
        subdiag = 0.5 * lamb * np.ones(N+1)
        supdiag = 0.5 * lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        AR = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            boundsL[0] = -0.5 * lamb * np.exp(0.5 * (self.q_d - 1) * self.xmin +
                          (0.25 * (self.q_d - 1) ** 2) * (j+1) * self.tstep)
            boundsR[0] = 0.5 * lamb * np.exp(0.5 * (self.q_d - 1) * self.xmin +
                          (0.25 * (self.q_d - 1) ** 2) * j * self.tstep)
            w[:, j+1] = thomas.solve(AL, np.matmul(AR, w[:, j]) - boundsL + boundsR)
          
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
    
    def put_cn_sor(self, omega, tol):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d - 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d + 1)),0)
            
        #set up initial conditions vector
        boundsL = np.zeros((N+1))
        boundsR = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 +  lamb) * np.ones(N+1)
        subdiag = -0.5 * lamb * np.ones(N+1)
        supdiag = -0.5 * lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        AL = spdiags(data, triset, N+1, N+1).toarray()
        
        maindiag = (1 - lamb) * np.ones(N+1)
        subdiag = 0.5 * lamb * np.ones(N+1)
        supdiag = 0.5 * lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        AR = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            boundsL[0] = -0.5 * lamb * np.exp(0.5 * (self.q_d - 1) * self.xmin +
                          (0.25 * (self.q_d - 1) ** 2) * (j+1) * self.tstep)
            boundsR[0] = 0.5 * lamb * np.exp(0.5 * (self.q_d - 1) * self.xmin +
                          (0.25 * (self.q_d - 1) ** 2) * j * self.tstep)
            w[:, j+1] = sor.solve(w[:,j], AL, 
                                  np.matmul(AR, w[:, j]) - boundsL + boundsR,
                                  omega, tol)
          
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
    
    def call_explicit(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d + 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d - 1)),0)
            
        #set up initial conditions vector
        bounds = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 - 2 * lamb) * np.ones(N+1)
        subdiag = lamb * np.ones(N+1)
        supdiag = lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            tau_j = j * self.tstep
            bounds[N] = lamb * np.exp(0.5 * (self.q_d + 1) * self.xmax +
                          (0.25 * (self.q_d + 1) ** 2) * tau_j) 
            w[:, j+1] = np.matmul(A, w[:, j]) + bounds
          
    
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
    
    def call_implicit_thomas(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d + 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d - 1)),0)
            
        #set up initial conditions vector
        bounds = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 + 2 * lamb) * np.ones(N+1)
        subdiag = -lamb * np.ones(N+1)
        supdiag = -lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            tau_j = (j + 1) * self.tstep
            bounds[N] = -lamb * np.exp(0.5 * (self.q_d + 1) * self.xmax +
                          (0.25 * (self.q_d + 1) ** 2) * tau_j) 
            w[:, j+1] = thomas.solve(A, w[:, j] - bounds)
          
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
        
    def call_implicit_sor(self, omega, tol):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d + 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d - 1)),0)
            
        #set up initial conditions vector
        bounds = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 + 2 * lamb) * np.ones(N+1)
        subdiag = -lamb * np.ones(N+1)
        supdiag = -lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        A = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            tau_j = (j + 1) * self.tstep
            bounds[N] = -lamb * np.exp(0.5 * (self.q_d + 1) * self.xmax +
                          (0.25 * (self.q_d + 1) ** 2) * tau_j) 
            w[:, j+1] = sor.solve(w[:,j], A, w[:, j] - bounds, omega, tol)
          
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
    
    def call_cn_thomas(self):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d + 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d - 1)),0)
            
        #set up initial conditions vector
        boundsL = np.zeros((N+1))
        boundsR = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 +  lamb) * np.ones(N+1)
        subdiag = -0.5 * lamb * np.ones(N+1)
        supdiag = -0.5 * lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        AL = spdiags(data, triset, N+1, N+1).toarray()
        
        maindiag = (1 - lamb) * np.ones(N+1)
        subdiag = 0.5 * lamb * np.ones(N+1)
        supdiag = 0.5 * lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        AR = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            boundsL[N] = -0.5 * lamb * np.exp(0.5 * (self.q_d + 1) * self.xmax +
                          (0.25 * (self.q_d + 1) ** 2) * (j+1) * self.tstep)
            boundsR[N] = 0.5 * lamb * np.exp(0.5 * (self.q_d + 1) * self.xmin +
                          (0.25 * (self.q_d + 1) ** 2) * j * self.tstep)
            w[:, j+1] = thomas.solve(AL, np.matmul(AR, w[:, j]) - boundsL + boundsR)
          
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
    
    def call_cn_sor(self, omega, tol):
        lamb = self.tstep / (self.xstep ** 2)
        N = int((self.xmax - self.xmin) / self.xstep)
        M = int(0.5 * (self.vol ** 2) * self.T / self.tstep)
        
        xgrid = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        
        #set up w matrix for FDM
        w = np.zeros((N+1,M+1))
        w[:,0] = np.maximum(np.exp(0.5 * xgrid * (self.q_d + 1)) -
                                  np.exp(0.5 * xgrid * (self.q_d - 1)),0)
            
        #set up initial conditions vector
        boundsL = np.zeros((N+1))
        boundsR = np.zeros((N+1))
        
        # set up A matrix for explicit FDM
        maindiag = (1 +  lamb) * np.ones(N+1)
        subdiag = -0.5 * lamb * np.ones(N+1)
        supdiag = -0.5 * lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        AL = spdiags(data, triset, N+1, N+1).toarray()
        
        maindiag = (1 - lamb) * np.ones(N+1)
        subdiag = 0.5 * lamb * np.ones(N+1)
        supdiag = 0.5 * lamb * np.ones(N+1)
        data = np.array([subdiag, maindiag, supdiag])
        triset = np.array([-1, 0, 1])
        AR = spdiags(data, triset, N+1, N+1).toarray()
        
        # tau loop. Work backward to initial time 
        for j in range(0, M):
            boundsL[N] = -0.5 * lamb * np.exp(0.5 * (self.q_d + 1) * self.xmax +
                          (0.25 * (self.q_d + 1) ** 2) * (j+1) * self.tstep)
            boundsR[N] = 0.5 * lamb * np.exp(0.5 * (self.q_d + 1) * self.xmin +
                          (0.25 * (self.q_d + 1) ** 2) * j * self.tstep)
            w[:, j+1] = sor.solve(w[:,j], AL, 
                                  np.matmul(AR, w[:, j]) - boundsL + boundsR,
                                  omega, tol)
          
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.S0,1][0]
