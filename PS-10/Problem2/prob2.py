import numpy as np
from scipy.fftpack import dst, idst
from scipy.constants import hbar
import matplotlib.pyplot as plt


m = 9.109E-31  # [kg], mass of electron
L = 1E-8       # Box length [meters]
x_0 = L / 2    # [meters]
sigma = 1E-10  # [meters]
kappa = 5E10   # [m^-1]
N = 1000       # Spatial slices
delta_x = L / N # Spatial step size [meters]
delta_t = 1E-18 # Time step size [seconds]
x = np.linspace(0, L, N)  # Spatial grid

#Initial wavefunction
psi_0 = np.exp(-((x - x_0)**2) / (2 * sigma**2)) * np.exp(1j * kappa * x)

#Separates real and imaginary parts
psi_0_real = np.real(psi_0)
psi_0_imag = np.imag(psi_0)

#Performs discrete sine transform (DST)
a_k = dst(psi_0_real, type=2) / N  #Real part coefficients
eta_k = dst(psi_0_imag, type=2) / N  #Imaginary part coefficients

#Combines into complex coefficients
b_k = a_k + 1j * eta_k

k_values = np.arange(1, N)  # Mode indices (k = 1 to N-1)
E_k = (np.pi**2 * hbar**2 * k_values**2) / (2 * m * L**2)  # Energy values

#Function to update DST coefficients
def update_coefficients(psi_real, psi_imag):
    a_k_new = dst(psi_real, type=2) / N
    eta_k_new = dst(psi_imag, type=2) / N
    return a_k_new, eta_k_new

#Calculates the real part of the wavefunction at time t
def psi_real_t(x, t, a_k, eta_k):
    summation = np.zeros_like(x)
    for k in range(1, N):
        factor = np.sin(np.pi * k * x / L)
        omega_k = E_k[k - 1] / hbar  
        time_dep = a_k[k - 1] * np.cos(omega_k * t) - eta_k[k - 1] * np.sin(omega_k * t)
        summation += time_dep * factor
    return (2 / N) * summation

#Calculates and plots the wavefunction at t = delta_t
psi_real = psi_real_t(x, delta_t, a_k, eta_k)

#Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, psi_real, label=f"Re[ψ(x, t)] at t = {10**-16:.1e} s")
plt.title("Real Part of the Wavefunction at t = 10^-16 s")
plt.xlabel("Position (m)")
plt.ylabel("Re[ψ(x, t)]")
plt.grid()
plt.legend()
plt.savefig('Spectral_Wave')


#Time snapshots
num_steps = 5  # Number of snapshots
times = np.linspace(0, (num_steps - 1) * delta_t, num_steps)  # Time intervals

#sPlotting snapshots
plt.figure(figsize=(10, 6))
for t in times:
    #Updates coefficients for the current time step
    psi_real, psi_imag = np.real(psi_real_t(x, t, a_k, eta_k)), np.imag(psi_real_t(x, t, a_k, eta_k))
    a_k, eta_k = update_coefficients(psi_real, psi_imag)  # Update DST coefficients
    psi_real = psi_real_t(x, t, a_k, eta_k)  # Calculate the wavefunction at time t
    plt.plot(x, psi_real, label=f"t = {t:.1e} s")

plt.title("Snapshots of the Real Part of the Wavefunction")
plt.xlabel("Position (m)")
plt.ylabel("Re[ψ(x, t)]")
plt.grid()
plt.legend()
plt.savefig('Spectral_Wave_Snapshots')
plt.show()

