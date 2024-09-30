import numpy as np
import matplotlib.pyplot as plt

m = 1 #arbitrary mass unit.
C = np.sqrt(8*m) #constant in front of the integral.

N = 20

amax = 2
amp = np.arange(0,amax,amax/100)


def anharmonic(a):
    x,w = np.polynomial.legendre.leggauss(N) #generates reference point locations and their weights for Legendre polynomials.
    xp = 0.5*a*x + 0.5*a
    wp = 0.5*a*w


    def V(xp):
        potential = xp**4
        return potential
    
    def integral(a,xp):
        Vdiff = V(a)-V(xp)
        valid_Vdiff = np.where(Vdiff>0, Vdiff, np.inf)
        return 1/np.sqrt(valid_Vdiff)
    
    T = 0.0
    for i in range(N):
        T += wp[i]*integral(a,xp[i])
    return C*T

T = np.array([anharmonic(a) for a in amp])

plt.figure()
plt.title('Anharmonic Oscillator')
plt.xlabel('Initial Amplitude')
plt.ylabel('Period')
plt.xscale('log')
plt.plot(amp,T, color = 'g')
plt.savefig('Anharmonic_Oscillator')




        