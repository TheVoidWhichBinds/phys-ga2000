import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar

#Parameters
m = np.float32(9.109E-31) #[kg].
L = np.float32(1E-8) #Box length [meters].
x_0 = L/2 #[meters].
sigma = 1E-10 #[meters].
kappa = 5E10 #[m^-1].
N = 1000 ##Spatial slices.
x_step = L/N #Spatial step size [meters].
t_step = 1E-18 #Time step size [seconds].
x = np.arange(0,L,x_step)


#Generates matrices A and B.
def matrices(N):
    #Coefficient matrix of Schrodinger equation for psi(t+t_step).
    A = np.zeros((N,N))
    a_1 = 1 + t_step*1j*hbar/(2*m*x_step**2)
    a_2 = - t_step*1j*hbar/(4*m*x_step**2)
    np.fill_diagonal(A,a_1) #Fills the main diagonal.             
    np.fill_diagonal(A[1:],a_2) #Fills the diagonal above the main diagonal.     
    np.fill_diagonal(A[:, 1:],a_2) #Fills the diagonal below the main diagonal.
    #Coefficient matrix of Schrodinger equation for psi(t).
    B = np.zeros((N,N))
    b_1 = 1 - t_step*1j*hbar/(2*m*x_step**2)
    b_2 = t_step*1j*hbar/(4*m*x_step**2)
    np.fill_diagonal(B,b_1)              
    np.fill_diagonal(B[1:],b_2) 
    np.fill_diagonal(B[:, 1:],b_2) 
    return A,B


#Generates the initial psi(x,0) wavefunction and the matrix v = B dot psi.
def wavefunction_matrix(x):
    psi_0 = np.exp(-(x-x_0)**2/(2*sigma**2))*np.exp(1j*kappa*x)
    _,B = matrices()
    v = np.dot(B,psi_0)
    return v

