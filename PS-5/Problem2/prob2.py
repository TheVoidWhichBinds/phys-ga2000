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
plt.savefig('GammaFunction_Integrand')

def gamma(a):
    x,w = np.polynomial.legendre.leggauss(100) #generates x100 reference point locations and their weights for Legendre polynomials.
    c = np.ones(100)*(a-1)
    z = x/(x+c)
    dz = (c)/(x+c)**2
    f = np.exp(c*np.log(z) - z)
    gamfunc = 0.0
    for i in range(100):
        gamfunc += w[i]*f[i]*dz[i]
    return gamfunc
    
print(gamma(3/2))