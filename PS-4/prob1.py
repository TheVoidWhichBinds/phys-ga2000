import numpy as np


V = 1000
rho = 6.022e28
theta = 428
N = 50

alpha = 9*V*rho*k*(T/theta)**3
lb = 0
ub = theta/T

def Cv(T):
    (ub**4)*(e**x)/(e**x - 1)**2
    

    
    
















def f(x):
    return x**4 - 2*x + 1

N = 3
a = 0.0
b = 2.0

x,w = np.polynomial.legendre.leggauss(N)
xp = 0.5*(b-a)*x + 0.5*(b+a)
wp = 0.5*(b-a)*w

s = 0.0
for k in range(N):
    s += wp[k]*f(xp[k])

print(s)