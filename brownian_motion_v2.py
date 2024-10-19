# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:06:05 2024

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
# Parameters
# drift coefficent
mu = 0
# number of steps
n = 25
# time in years
T = 25
# number of sims
M = 1
# initial stock price
S0 = 100
# volatility
sigma = 0.3
eps = 0.01

def GBM(mu, sigma, eps, T, n):
    # calc each time step
    dt = T/n
    # simulation using numpy arrays
    St = np.exp((mu - sigma ** 2 / 2) * dt + sigma * eps * np.random.normal(0, np.sqrt(dt), size=(M,n)).T)
    # Define time interval correctly 
    time = np.linspace(0,T,n+1)
    tt = np.full(shape=(M,n+1), fill_value=time).T
    # include array of 1's
    St = np.vstack([np.ones(M), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0). 
    St = S0 * St.cumprod(axis=0)
    return tt, St

tt,St = GBM(mu, sigma, eps, T, n)

print(St)


# Define time interval correctly 
time = np.linspace(0,T,n+1)

St1 = np.zeros(n+1)
for i in range(n+1):
    St1[i] = S0 * np.exp((mu - sigma ** 2 / 2) * time[i] + sigma * eps * np.sqrt(time[i]))
    
# Require numpy array that is the same shape as St
plt.plot(tt, St)
plt.plot(time, St1, 'r')
plt.xlabel("Years $(t)$")
plt.ylabel("Stock Price $(S_t)$")
plt.title("Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(S0, mu, sigma))
plt.show()