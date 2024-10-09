import numpy as np
import matplotlib.pyplot as plt

def integrand(x,a):
    ig = x**(a-1)*np.exp(-x)
    return ig

#graph settings.
plt.figure(figsize=(10, 6)) 
plt.title('Gamma Function Integrand', fontsize = 20)
plt.ylabel('Integrand',fontsize = 14)
plt.xlabel('x',fontsize = 14)

#plotting derivative with respect to x.
xs = np.linspace(0,5,100)
plt.plot(xs,integrand(xs,2), label='a=2', color = 'b')
plt.plot(xs,integrand(xs,3), label='a=3', color = 'r')
plt.plot(xs,integrand(xs,4), label='a=4', color = 'm')
plt.legend()
plt.savefig('Gamma_Integrand')

def gamma(a):
    x,w = np.polynomial.legendre.leggauss(100) #generates x100 reference point locations and their weights for Legendre polynomials.
    c = np.ones(100)*(a-1) #analytically calculated constant in change of variables.
    z = x/(x+c) #change of variables to compute integral, changing upper bound from infinity to 1.
    dz = (c)/(x+c)**2 #differential segment dz.
    f = np.exp(c*np.log(z) - z) #gamma function integrand.
    gamfunc = 0.0
    for i in range(100):
        gamfunc += w[i]*f[i]*dz[i] #gaussian quadrature integration.
    return gamfunc
    
print(gamma(3/2))