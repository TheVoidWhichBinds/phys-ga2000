import numpy as np
import matplotlib.pyplot as plt

m = 1 #arbitrary mass unit.
C = np.sqrt(8*m) #constant in front of the integral.

N = 20

amax = 2
amp = np.arange(0,amax,amax/1000)


def anharmonic(a):
    x,w = np.polynomial.legendre.leggauss(N) #generates reference point locations and their weights for Legendre polynomials.
    xp = 0.5*a*x + 0.5*a
    wp = 0.5*a*w


    def V(xp): #allows input for a potential of any form.
        potential = xp**4 
        return potential
    
    def integral(a,xp):
        Vdiff = V(a)-V(xp) #potential difference between max and all other points.
        valid_Vdiff = np.where(Vdiff>0, Vdiff, np.nan) #ensures no divide by zero.
        return 1/np.sqrt(valid_Vdiff) #outputs denominator in integral.
    
    T = 0.0
    for i in range(N):
        T += wp[i]*integral(a,xp[i]) #integral via gaussian quadrature.
    return C*T #period calculated.

T = np.array([anharmonic(a) for a in amp])

plt.figure()
plt.title('Anharmonic Oscillator', fontsize = 17)
plt.xlabel('Initial Amplitude', fontsize = 12)
plt.ylabel('Period', fontsize = 12)
plt.xscale('log')
plt.plot(amp,T, color = 'g')
plt.savefig('Anharmonic_Oscillator')




        