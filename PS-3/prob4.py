import numpy as np
import timeit

#function[] of N how the mean, variance, skewness, and kurtosis of the distribution change.

N = 100
s = 1000
x_i = np.random

def y_variate(N):
    x_i = rng.exponential(N,s)
    y = 0
    for i in range(N):
        y_i = (1/N)*(x_i)
        y = y + y_i
    return y
    
    
#make Gaussian to overlay plot
#for kurtosis, etc., use numpy et al packages to overlay 
#mean close to 1 
#skoonis close to 2
#kurtosis close to 4
    
#expectation value of y = (1/N)*sum to N of Expectation value of x_i where expectation of x_i
#equals infinite sum from 0 to infinity of x*p(x)dx = 1
#sqrt(N)(y-mu)/sigma --> Gaussian (standard limit theorem)