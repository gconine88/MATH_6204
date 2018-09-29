# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 11:59:51 2017

@author: Grant Conine

UNC Charlotte MS Math Finance
MATH 6204 Numerical Methods for Financial Derivatives
Prof Hwan Lin

################################
           Question 1
################################

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

Citations:
Seydel, Rudinger (2017). "Tools for Computational Finance" 6th Ed. Springer
    Univeritex. (2017)

################################
           Question 2
################################

Use Monte Carlo method to compute the value of European and American
calls and put options under the following assumptions:
        
    Stock price process follows a geometric Brownian motion, that is
        
        dSt = (mu - d) St dt + sigma St dWt
            
    where St is the stock price process at time t, mu is the average
    return, d is the dividend yield of the stock, sigma is the
    volatility of the stock price process, and Wt is the Weiner process.
        
    This class works by repeatedly creating a simulation of a the geometric
    Brownian motion using the Euler discreetization:
            
        S[t+1] = S[t] + (mu - d) S[t] delta_t + vol S[t] W
            
        Where W is randomly drawn from a standard normal distribution.
            
    Once the stock price processes have been simulated, we may compute the
    value of the European or American call or put options.
        
    The European call (put) option is valued by taking the average 
    discounted payoff function, i.e.:
            
                1
        Call = --- exp(-(r-d)(T-t)) sum( max(S(T) - K, 0) )
                M
                    
               1
        Put = --- exp(-(r-d)(T-t)) sum( max(K - S(T), 0) )
               M
                   
    The American options are priced using a more complicated algorithm to
    account for the possibility of early excercise. Once the paths for the
    stock price process have been calculated, we follow this algorithm for
    a call [put] option
        
    Excercise h(S_t) = max(S_t - K,0) [h(S_t) = max(K - S_t, 0)]
        
    For k = 0, 1, 2, ..., N (the number of paths simulated)
        For t = T, T-delta_t, ..., delta_t, 0
            if t = T
                g(k) = h(S_Tk)
                tau(k) = T  (the optimal excercise time for the path)
            if 0 < t < T
                For in-the-money points (where S_tk > K)
                   (xk, yk) = (S_tk, exp( -(r - d)(tau(k) - i) delta_t)g(k))
                   
                Approximate C_t(x) ~ sum_l a_l b_l(x) using basis functions
                x, x^2, x^3
                   
                If h(S_tk) >= C_t(S_tk) we should excercise. Set
                g(k) = h(S_tk), tau_k = t
            if t = 0
                Set
                           1
                    C_0 = --- sum exp( -(r - d) delta_t tau(k)) g(k))
                           N
                    V_0(S_0) =max(C_0, h(S_0))
                
Citations:
    Hilpisch, Yves (2015). "Derivative Analytics with Python." Wiley Finance 
        Series.
    Longstaff, Francis and Eduardo Schwartz (2001). "Valuing American Options 
        by Simulation: A Simple Least-Squares Approach." Review of Financial 
        Studies. 14 (1): 113-147.
    Seydel, Rudinger (2017). "Tools for Computational Finance" 6th Ed. Springer
        Univeritex. (2017)

################################
           Question 3
################################

Compute Analytic solution to Option pricing under Black-Scholes-Merton
assumptions as follows:
        
    Stock price process follows the following process:
            
        dSt = (mu - d) St dt + sigma St dWt
            
        where St is the stock price process at time t, mu is the average
        return, d is the dividend yield of the stock, sigma is the
        volatility of the stock price process, and Wt is the Weiner process
            
    It is well known the time t value V of an option is a solution to:
            
  dV(St,t)    1               d^2 V(St,t)               dV(St,t)
 --------- + --- sigma^2 S^2 ------------- + (r - d) S --------- - r V(St,t) = 0
     dt       2                   dS^2                     dS
     
For a European call option this yields:
            
    C(St, t) = N(d1) St - N(d2) K exp(-(r - d)(T - t))
            
                   1               St               vol^2
        d1 = --------------- [ ln(----) + (r - d + -------)(T - t)]
             vol sqrt(T - t)        K                  2
            
        d2 - d1 - vol sqrt(T - t)
            
        N is the cumulative distribution function of the standard normal
        
For a European put option:
            
    P(St, t) = N(-d2) K exp( - (r - d)(T - t)) - N( -d1) St
        
Citations:
Black, Fischer; Myron Scholes (1973). "The Pricing of Options and Corporate
    Liabilities". Journal of Political Economy. 81 (3): 637–654.
Hilpisch, Yves (2015). "Derivative Analytics with Python." Wiley Finance Series.
Seydel, Rudinger (2017). "Tools for Computational Finance" 6th Ed. Springer
    Univeritext.

"""

import analytic_option as ao
import pandas as pd
import mc_option as mc
import fdm_euro
import fdm_amer

# Set Model Parameters
S0 = 100 #current asset price
K = 100 #strike price
T = 1 #maturity date
t = 0 #current date
r = 0.02 #risk-free short rate
d = 0.01 #dividend yield
mu = 0.05 #mean return of asset
vol = 0.6 #volatility of asset


# Question 1 - Finite Difference methods

#Algorithmic parameters
xmin = -2.5 
xmax = 2.5
xstep = 0.05
tstep = 0.00125

omega = 1.10
tol = 10 ** -6

o1 = fdm_euro.option(S0, K, T, t, r, d, vol, xmin, xmax, xstep, tstep)

put_euro_expl = o1.put_explicit()
call_euro_expl = o1.call_explicit()
put_euro_imp_thom = o1.put_implicit_thomas()
call_euro_imp_thom = o1.call_implicit_thomas()
put_euro_imp_sor = o1.put_implicit_sor(omega, tol)
call_euro_imp_sor = o1.call_implicit_sor(omega, tol)
put_euro_cn_thom = o1.put_cn_thomas()
call_euro_cn_thom = o1.call_cn_thomas()
put_euro_cn_sor = o1.put_cn_sor(omega, tol)
call_euro_cn_sor = o1.call_cn_sor(omega, tol)

# Assemble output
df = {'Call Option': [call_euro_expl, call_euro_imp_thom, 
                     call_euro_imp_sor, call_euro_cn_thom,
                     call_euro_cn_sor],
     'Put Option': [put_euro_expl, put_euro_imp_thom, 
                    put_euro_imp_sor, put_euro_cn_thom,
                    put_euro_cn_sor]}

dff = pd.DataFrame(df, index=['Explicit Method',
                            'Implicit Method, Thomas Algorithm',
                            'Implicity Method, SOR Algorithm',
                            'Crank-Nicholson Method, Thomas Algorithm',
                            'Crank-Nicholson Method, SOR Algorithm'])
     
print('European option with finite difference method.')
print(dff)

o2 = fdm_amer.option(S0, K, T, t, r, d, vol, xmin, xmax, xstep, tstep)

put_amer_expl = o2.put_explicit()
call_amer_expl = o2.call_explicit()
put_amer_imp_bsa = o2.put_implicit_bsa()
call_amer_imp_bsa = o2.call_implicit_bsa()
put_amer_imp_sor = o2.put_implicit_psor(omega, tol)
call_amer_imp_sor = o2.call_implicit_psor(omega, tol)
put_amer_cn_bsa = o2.put_cn_bsa()
call_amer_cn_bsa = o2.call_cn_bsa()
put_amer_cn_sor = o2.put_cn_psor(omega, tol)
call_amer_cn_sor = o2.call_cn_psor(omega, tol)

# Assemble output
df = {'Call Option': [call_amer_expl, call_amer_imp_bsa, 
                     call_amer_imp_sor, call_amer_cn_bsa,
                     call_amer_cn_sor],
     'Put Option': [put_amer_expl, put_amer_imp_bsa, 
                    put_amer_imp_sor, put_amer_cn_bsa,
                    put_amer_cn_sor]}

dff = pd.DataFrame(df, index=['Explicit Method',
                            'Implicit Method, Brennan-Schwartz Algorithm',
                            'Implicity Method, Projected SOR Algorithm',
                            'Crank-Nicholson Method, Brennan-Schwartz Algorithm',
                            'Crank-Nicholson Method, Projected SOR Algorithm'])

print('')     
print('American option with finite difference method.')
print(dff)

# Question 2 - Monte Carlo methods

# Algorithmic parameters
N = 500 #number of simulations to perform
delta_t = 0.00125 #size of time step

put_Amer_MC = mc.MC_Amer_put(S0, K, mu, r, d, vol, t, T, delta_t, N)
call_Amer_MC = mc.MC_Amer_call(S0, K, mu, r, d, vol, t, T, delta_t, N)
put_Euro_MC = mc.MC_Euro_put(S0, K, mu, r, d, vol, t, T, delta_t, N)
call_Euro_MC = mc.MC_Euro_call(S0, K, mu, r, d, vol, t, T, delta_t, N)

df = {'Call Option': [call_Euro_MC, call_Amer_MC],
     'Put Option': [put_Euro_MC, call_Amer_MC]}

dff = pd.DataFrame(df, index=['European Option', 'American Option'])

print('')
print('Options with Monte-Carlo method.')
print(dff)
# Question 3 - Closed form solutions

o3 = ao.AnalyticOption(S0, K, T, t, r, d, vol)

call_Euro_Analytic = o3.Call_Value()

put_Euro_Analytic = o3.Put_Value()

df = {'Call Option': [call_Euro_Analytic],
     'Put Option':  [put_Euro_Analytic]}

dff = pd.DataFrame(df, index=['Analytic Option Price'])

print('')
print('Options with Analytic method')
print(dff)
