import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar
from Banded import *
from vpython import rate

print(hbar)

#Parameters
m = np.float32(9.109E-31) #[kg].
L = np.float32(1E-8) #Box length [meters].
x_0 = L/2 #[meters].
sigma = 1E-10 #[meters].
kappa = 5E10 #[m^-1].
N = 1000 ##Spatial slices.
x_step = np.float32(L/N)#Spatial step size [meters].
t_step = np.float32(1E-18) #Time step size [seconds].
x = np.arange(0,L,x_step)


#Generates matrices A and B.
def matrices(N):
    #Coefficient matrix of Schrodinger equation for psi(t+t_step).
    A = np.zeros((N,N), dtype=np.complex128)
    a1 = np.complex32(1 + t_step*hbar/2/m/x_step**2*1j)
    a2 = np.complex32(-t_step*hbar*1j/4/m/x_step**2)
    np.fill_diagonal(A,a1) #Fills the main diagonal.             
    np.fill_diagonal(A[1:],a2) #Fills the diagonal above the main diagonal.     
    np.fill_diagonal(A[:, 1:],a2) #Fills the diagonal below the main diagonal.
    #Coefficient matrix of Schrodinger equation for psi(t).
    B = np.zeros((N,N), dtype=np.complex128)
    b1 = np.complex32(1 - t_step*hbar/2/m/x_step**2*1j)
    b2 = np.complex32(t_step*hbar*1j/4/m/x_step**2)
    np.fill_diagonal(B,b1)              
    np.fill_diagonal(B[1:],b2) 
    np.fill_diagonal(B[:, 1:],b2) 
    return A,B


#Solves psi_t at some time i*t_step.
psi_0 = np.exp(-(x-x_0)**2/(2*sigma**2))*np.exp(1j*kappa*x) #t=0 wavefunction.
_,B = matrices(N)
A,_ = matrices(N)
    
psi_t = psi_0 #Initialization.
while True: #In range(x), x indicates number of time steps of evolution (time = x*t_step).
    rate(10)
    psi_t = banded(A, B@psi_t, 1, 1)#Solves the matrix equation A @ psi_t = B @ psi_0.
    



