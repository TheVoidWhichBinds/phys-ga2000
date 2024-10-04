import numpy as np
import matplotlib.pyplot as plt

V = 1e-3 #volume (m^3).
rho = np.float32(6.022e28) #number density of solid aluminum (m^-3).
theta = 428 #Debye temperature (K).
k_B = np.float32(1.380649e-23) #Boltzmann constant (J/K).

T = np.arange(5,500,1) #array of temperature values.

#calculates heat capacity for a given temperature.
def heat_capacity(Ti,N):
    alpha = 9*V*rho*k_B*(Ti/theta)**3 #constant outside integral.
    a = 0 #lower bound of integral.
    b = theta/Ti #upper bound of integral.
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
    
Cv = np.array([heat_capacity(Ti,50) for Ti in T]) #makes an array of heat capacity values for each value of T in the array.


plt.figure()
plt.title('Heat Capacity of a Solid', fontsize = 16)
plt.xlabel('Temperature (K)', fontsize = 12)
plt.ylabel('Heat Capacity Cv (J/K)', fontsize = 12)
plt.xscale('log')
plt.plot(T,Cv, color = 'r')
plt.savefig('Heat_Capacity')

plt.figure()
plt.title('Heat Capacity Using Gaussian Quadrature', fontsize = 16)
plt.xlabel('Temperature (K)', fontsize = 12)
plt.ylabel('Heat Capacity Cv (J/K)', fontsize = 12)

plt.plot(T,[heat_capacity(Ti,10) for Ti in T], color = 'r', label = 'N = 10')
plt.plot(T,[heat_capacity(Ti,20) for Ti in T], color = 'y', label = 'N = 20')
plt.plot(T,[heat_capacity(Ti,30) for Ti in T], color = 'g', label = 'N = 30')
plt.plot(T,[heat_capacity(Ti,40) for Ti in T], color = 'c', label = 'N = 40')
plt.plot(T,[heat_capacity(Ti,50) for Ti in T], color = 'm', label = 'N = 50')
plt.legend()
plt.savefig('Quadrature_Divergence')
