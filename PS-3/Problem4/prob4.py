import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

Nval = np.arange(10,1000) #array of the various N values.
s = 1000 #number of samples.

def sample(N,s):
    x = np.random.default_rng().exponential(scale=1, size=(s,N)) #array of random variates from e^-x.
    y = np.mean(x, axis=1) #y = (1/N)*sum(x_i)
    return y

#initialization of arrays for different quantities of interest.
mean_val = []
variance_val = []
skew_val = []
kurtosis_val = []

for N in Nval: #for each N value, the x variate is calculated over an average sampling of s.
    y = sample(N,s) 
    #statistical quantities of interest are collected in their respective arrays.
    mean_val.append(np.mean(y))
    variance_val.append(np.var(y))    
    skew_val.append(skew(y))  
    kurtosis_val.append(kurtosis(y))

plt.figure()
plt.plot(Nval, mean_val, label='Mean')
plt.plot(Nval, kurtosis_val, label='Kurtosis')
plt.plot(Nval, skew_val, label='Skew')
plt.plot(Nval, variance_val, label='Variance', color = 'm')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Statistic Value')
plt.legend()
plt.title('Mean, Variance, Skewness, and Kurtosis as a function of N')
plt.savefig('CLT')

