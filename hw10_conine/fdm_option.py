# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 16:03:31 2017

@author: Grant

Finite Difference Methods (or PDE methods) are toolkit of numerical methods
for finding solutions to options pricing PDEs that would otherwise not be 
solvable using analytic methods.

FDMs have the following advantages over Monte Carlo methods for solving similar
options:
    - There is a clear scaling between computational effort and accuracy 
    - Easily handle early excercise/boundaries
    - Prices options for all values of stock, not just one S0
Disadvantages:
    - The stock price process must be Markovian (ie Memoryless - cannot price
      certain Asian and Russian options)
    - Must be small dimensional problem
    
First we approximate the concept of derivative in a discrete manner. We 
consider three finite difference approximations:
                          d V(i)    V(i+1) - V(i)
    (1) Explicit Method: ------- = ---------------
                           dt         delta t
           
                         d V(i+1)    V(i+1) - V(i)
    (2) Implicit Mehod: --------- = ---------------
                            dt          delta t
           
                                 1    d V(i)     d V(i+1)      V(i+1) - V(i)
    (3) Crank-Nicholson Method: --- (-------- + ----------) = ---------------
                                 2      dt          dt           deltla t
                                 
Or in matrix form:
    (1) V(i+1) = (I + delta t A) V(i)
    (2) (I - delta t A) V(i+1) = V(i)
              delta t A                  delta t A
    (3) (I - -----------) V(i+1) = (I + -----------) V(i)
                  2                          2
                  
We will be using this approximation on the Heat equation representation of 
the Black-Scholes equation. 

    dV     sigma^2       d^2 S               dV
   ---- + --------- S^2 ------- + (r - d) S ---- - rV = 0
    dt        2           dS^2               dS
    
with terminal conditions for the call [put] option
    
    C(S,T) = (S - K)^+  [P(S,T) = (K - S)^+] 
    
and boundary conditions
    
    S -> 0, C(S,t) -> 0 [P(S,t) -> K exp(-r(T - t))]
    S -> infty, C(S,t) -> S exp(-d(T-t)) [P(S,t) -> 0]
    
If we make the following substitutions
    
    x = ln (S/K)
    tau = sigma^2/2 (T - t)
    q = 2r / sigma^2
    q_d = 2(r - d) / sigma^2
    
Then
    V(S,t) = V(Ke^x, T - 2tau/sigma^2) = v(x,tau)
    v(x,tau) = K exp{ - 1/2(q_d - 1)x - (1/4(q_d -1)^2 + q)tau} y(x,tau)
where y(x,tau) is the solution to the heat equation

       dy        dy^2                                   sigma^2 T
    ------- = ---------, -infty < x < infty, 0 < tau < -----------
     d tau     d tau^2                                      2
     
To solve for American Options, we must also include the possibility of early
excercise in our calculations, expressed as additional boundary conditions. 
Instead of the y_t = y_xx expression of the heat equation as in the European
option, we have y_t >= y_xx with the function g(x_j, tau_i) = g(j,i) serving as 
the aforementioned additional boundary conditions. 

For an American call:
    g(j,i) = (exp{(1/4(q_d - 1)^2 + q)tau_i} * 
                  max{exp{1/2(q_d + 1)x_j} - exp{1/2(q_d - 1)x_j}, 0})
and American put:
    g(j,i) = (exp{(1/4(q_d - 1)^2 + q)tau_i} * 
                  max{exp{1/2(q_d - 1)x_j} - exp{1/2(q_d + 1)x_j}, 0})
     
Once the PDE is appropriately discreetized, we may either directly solve using
direct methods (Thomas algorithm/Brennan-Schwartz algorithm) or by an
iterative solver (SOR algorithm/projected SOR)

Thomas Algorithm
    This is a direct solver for a tri-diagonal matrix. If the three diagonals
    are a_i, i = 1,...,n on the main diagonal, b_i, i = 1,...,n-1 on the
    super-diagonal, and c_i, i =2,...,n-1 on the sub-diagonal, and the solution
    for the equation is d_i, 1= 1,...,n:
        (1) ahat_1 = a_1, bdhat_1 = d_1
        (2) Forward Loop i = 2,...,n:
            (a) ahat_i = a_i - b_(i-1)(c_i / ahat_(i-1))
            (b) dhat_i = d_i - d_hat(i-1)(c_i / ahat_(i-1))
        (3) Backward loop j = n-1,...,1:
            (a) x_n = dhat_n / ahat_n
            (b) x_i = (dhat_i - b_i x_(i+1)) / ahat_i

Brennan-Schwartz Algorithm
    This is a direct solver for a tri-diagonal matrix with the additional 
    boundary conditions for pricing American options. 
        (1) (1) ahat_1 = a_1, bdhat_1 = d_1
        (2) Forward Loop i = 2,...,N-1:
            (a) ahat_i = a_i - b_(i-1)(c_i / ahat_(i-1))
            (b) dhat_i = d_i - d_hat(i-1)(c_i / ahat_(i-1))
        (3) Backward loop j = N-2,...,1:
            (a) w_(N-1) = max{g(N-1), dhat_(N-1) / ahat_(N-1)}
            (b) w_j = max{g(j), (dhat_j - b_j x_(j+1)) / ahat_j}
            
Successive Over Relaxation algorithm
    This is an iterative solver to solve a system of linear equations. We set
    a over-relaxation parameter 1 < omega < 2 and a tolerance level tol, as 
    well as a maximum number of iterations to perform. Let (a_ij) = A, Ax = d
    describe the system of linear equations
    
    (1) Enter initial guess x0 = x(1)
    (2) REPEAT:
        (a) Set x(2) = x(1)
        (b) Set x_i(1) = x_i(2) + omega/a_ii(d_i - 
                             sum_j<i a_ij x_j(1) - 
                             sum_j>i a_ij x_j(2) )
        (c) Stop if |x(1) - x(2)| < tol or the maximum iterations have been 
            completed

Projected SOR algorithm
    This is an iterative solver for a system of linear equations with the 
    additional boundary conditions for pricing American options.
    
    (1) Enter intial guess w0 = w(1)
    (2) REPEAT:
        (a) Set w(2) = w(1)
        (b) Set w_i(1) = max{g_i, w_i(2) + omega/a_ii(d_i - 
                             sum_j<i a_ij w_j(1) - a_iiw_i(2) - 
                             sum_j>i a_ij w_j(2) )}
        (c) Stop if |w(1) - w(2)| < tol or the maximum iterations have been
            completed.
            
Once a solution to the heat equation has been found, it is simply necessary to 
back into the financial solution by reversing early transformations.
"""

import numpy as np
import thomas
from scipy.sparse import spdiags


# Explicit FDM

class fdm_option:
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

        
    def euro_put_explicit(self):
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
        
        return V[V[:,0] == self.K,1][0]
    
    def euro_put_implicit_thomas(self):
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
            w[:, j+1] = thomas.full(A, w[:, j] - bounds)
          
        V = np.zeros((N+1,2))
        
        for i in range(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.K,1][0]
    
    def euro_call_explicit(self):
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
        
        return V[V[:,0] == self.K,1][0]
    
    def euro_call_implicit_thomas(self):
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
            w[:, j+1] = thomas.full(A, w[:, j] - bounds)
          
        V = np.zeros((N+1,2))
        
        for i in rangeS(0,N):
            V[i,0] = self.K * np.exp(self.xmin + i * self.xstep)
            V[i,1] = self.K * np.exp(-0.5 * (self.q_d - 1) * (self.xmin + i * self.xstep)
                - (0.25 * (self.q_d - 1) ** 2 + self.q) * self.tau ) * w[i, M]
        
        return V[V[:,0] == self.K,1][0]
        
