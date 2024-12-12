


import numpy as np
from vpython import curve, vector, scene, canvas, rate
from scipy.constants import hbar
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

# Parameters
m = 9.109E-31  # [kg], mass of electron
L = 1E-8       # Box length [meters]
x_0 = L / 2    # [meters]
sigma = 1E-10  # [meters]
kappa = 5E10   # [m^-1]
N = 1000       # Spatial slices
delta_x = L / N # Spatial step size [meters]
delta_t = 1E-18 # Time step size [seconds]
x = np.linspace(0, L, N)  # Spatial grid

#Generate matrices A and B
def matrices(N):
    a1 = 1 + delta_t*hbar*1j/(2*m)/(delta_x**2)
    a2 = -delta_t*hbar*1j/(4*m)/(delta_x**2)
    offdiag = np.full(N-1, a2, dtype=np.complex64)  #Lower diagonal (below main diagonal)
    maindiag = np.full(N, a1, dtype=np.complex64)     #Main diagonal
    A_band = np.zeros((3, N), dtype=np.complex128)
    A_band[0, 1:] = offdiag  #Upper diagonal (shifted by 1)
    A_band[1, :] = maindiag  #Main diagonal
    A_band[2, :-1] = offdiag  #Lower diagonal (shifted by -1)

    B = np.zeros((N, N), dtype=np.complex64)
    b1 = 1 - delta_t*hbar*1j/(2*m)/(delta_x**2)
    b2 = delta_t*hbar*1j/(4*m)/(delta_x**2)
    np.fill_diagonal(B, b1)
    np.fill_diagonal(B[1:], b2)
    np.fill_diagonal(B[:, 1:], b2)
    return A_band, B

psi_0 = np.exp(-(x - x_0)**2 / (2 * sigma**2)) * np.exp(1j * kappa * x)  #Initial wavefunction
A_band, B = matrices(N)

#Time evolution of the wavefunction by solving the matrix equation
def wavefunction_evolution(psi_t):
    return solve_banded((1, 1), A_band, B @ psi_t)

#Simulation parameters
t_steps = 500  #Total number of time steps
snapshot_times = [0, 50, 100, 200, 300, 400, 499]  #Time steps to plot

#Store the wavefunction evolution
psi_evolution = np.zeros((t_steps, N), dtype=np.complex128)
psi_evolution[0, :] = psi_0

for i in range(t_steps - 1):
    psi_evolution[i + 1, :] = wavefunction_evolution(psi_evolution[i, :])

#Plotting the wavefunction amplitude at selected time steps
plt.figure(figsize=(10, 6))
for t in snapshot_times:
    amplitude = np.abs(psi_evolution[t, :])  #Taking the real part of the wavefunction
    plt.plot(x, amplitude, label=f'Time step {t}')
    
plt.title("Wavefunction Evolution")
plt.xlabel("Position [meters]")
plt.ylabel("Amplitude |ψ|")
plt.legend()
plt.grid()
plt.savefig('Crank_Wave')