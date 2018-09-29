# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 11:59:51 2017

@author: Grant

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

import math
import scipy.stats as scs

# Class Definition

class AnalyticOption:
    ''' Compute Analytic solution to Option pricing under Black-Scholes-Merton
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
    Hilpisch, Yves (2015). "Derivative Analytics with Python." Wiley Finance 
        Series.
    Seydel, Rudinger (2017). "Tools for Computational Finance" 6th Ed. Springer
        Univeritext.
            
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
    
    def __init__(self, S0, K, T, t, r, d, vol):
        '''Initialize option object'''
        self.S0 = float(S0)
        self.K = float(K)
        self.ttm = T - t #instead of keeping T and t seperate, find time to maturity
        self.ex = r - d #instead of keeping r and d sepetate, find excess risk-free
        self.vol = vol
        
    def d1(self):
        '''Helper function for Black-Scholes pricing'''
        
        d1 = (math.log(self.S0 / self.K) + 
              (self.ex + 0.5 * self.vol ** 2) * self.ttm
             / (self.vol * math.sqrt(self.ttm)))
        return d1
    
    def Call_Value(self):
        ''' Black-Scholes valuation of European call option.'''
        
        d1 = self.d1()
        d2 = d1 - self.vol * math.sqrt(self.ttm)
        
        call_value = (self.S0 * scs.norm.cdf(d1, 0.0, 1.0)
                      - self.K * math.exp(-self.ex * self.ttm)
                      * scs.norm.cdf(d2, 0.0, 1.0))
        return call_value
    
    def Put_Value(self):
        ''' Black-Scholes valuation of European put option.'''
        
        d1 = self.d1()
        d2 = d1 - self.vol * math.sqrt(self.ttm)
        
        put_value = (- self.S0 * scs.norm.cdf(-d1, 0.0, 1.0)
                     + self.K * math.exp(-self.ex * self.ttm)
                     * scs.norm.cdf(-d2, 0.0, 1.0))
        
        return put_value