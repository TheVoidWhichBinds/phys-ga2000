import numpy as np
import matplotlib.pyplot as plt

V = 1e-3 #volume (m^3).
rho = np.float32(6.022e28) #density of solid aluminum (m^-3).
theta = 428 #Debye temperature (K).
k_B = np.float32(1.380649e-23) #Boltzmann constant (J/K).

N = 50 #number of points taken for gaussian quadrature.
T = np.arange(5,500,1) #array of temperature values.

#calculates heat capacity for a given temperature.
def heat_capacity(T):
    alpha = 9*V*rho*k_B*(T/theta)**3 #constant outside integral.
    a = 0 #lower bound of integral.
    b = theta/T #upper bound of integral.
    x,w = np.polynomial.legendre.leggauss(N) #generates reference point locations and their weights for Legendre polynomials.

    xp = 0.5*(b-a)*x + 0.5*(b+a) 
    wp = 0.5*(b-a)*w
    
    def heat_func(xp): #the contents of the integral.
        heat = (b**4)*(np.exp(xp))/(np.exp(xp)- 1)**2
        return heat
    
    Cv = 0.0
    for k in range(N):
        Cv += wp[k]*heat_func(xp[k]) #calculates the integral using gaussian quadrature.
    return alpha*Cv #total heat capacity.
    
Cv = np.array([heat_capacity(Ti) for Ti in T]) #makes an array of heat capacity values for each value of T in the array.


plt.figure()
plt.title('Heat Capacity of a Solid')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity Cv (J/K)')
plt.xscale('log')
plt.plot(T,Cv, color = 'r')
plt.savefig('Heat_Capacity')

