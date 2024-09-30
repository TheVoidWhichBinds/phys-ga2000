import numpy as np
import math
import matplotlib.pyplot as plt


def Hermite(n,x): #calculates Hermite polynomial for the nth energy level.
    H = np.zeros((n+1,len(x))) #initializes: nth energy level row and x position value columns.
    if n == 0:
        H[0,:] = 1 #0th energy level.
    if n == 1:
        H[1,:] = 2*x #1st energy level.
    if n > 1: #prevents bad indexing (row -1,-2).
        H[0,:] = 1
        H[1,:] = 2*x
        for i in range(2,n+1):
            H[i,:] = 2*x*H[i-1,:] - 2*n*H[i-2,:]
    return H
        

def Schrodinger(n,x): #wave(function^2)
    c = 1/np.sqrt((2**n)*math.factorial(n)*np.sqrt(math.pi)) #part of wavefunction independent of x.
    wavefunction = np.exp((-x**2)/2)*Hermite(n,x)[n,:] #part of wavefunction dependent on x.
    return c*wavefunction #returns normalized wavefunction.


def quantum_uncertainty(n,N):
    a = -np.inf #lower bound of integral.
    b = np.inf #upper bound of integral.
    x,w = np.polynomial.legendre.leggauss(N) #generates reference point locations and their weights for Legendre polynomials.
    xp = 0.5*(b-a)*x + 0.5*(b+a) 
    wp = 0.5*(b-a)*w

    rms = np.float32(0.0)
    for i in range(N):
        rms += wp[i]*Schrodinger(n,xp)[i]

    return np.sqrt(rms)

print(quantum_uncertainty(5,100))
    










plt.figure(figsize=(10, 6)) 
plt.title('1-Dimensional Harmonic Oscillator')
plt.ylabel(r'Probability Density $\Psi$')
plt.xlabel('Position x')

r1 = np.linspace(-4,4,100)
plt.plot(r1,Schrodinger(0,r1), label = 'n = 0', color = 'r')
plt.plot(r1,Schrodinger(1,r1), label = 'n = 1', color = 'c')
plt.plot(r1,Schrodinger(2,r1), label = 'n = 2', color = 'm')
plt.plot(r1,Schrodinger(3,r1), label = 'n = 3', color = 'k')
plt.legend()
plt.savefig('Wavefunctions')


plt.figure(figsize=(10, 6)) 
plt.title('1-Dimensional Harmonic Oscillator')
plt.ylabel(r'Probability Density $\Psi$')
plt.yticks(np.arange(-6e5, 6e5, 1e5))
plt.xlabel('Position x')

r2 = np.linspace(-10,10,500)
plt.plot(r2,Schrodinger(30,r2), label = 'n = 30', color = 'm')
plt.legend()
plt.savefig('n30')