


import numpy as np
from vpython import curve, vector, scene, canvas, rate
from scipy.constants import hbar
from scipy.linalg import solve_banded
import matplotlib as plt

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

# Generate matrices A and B
def matrices(N):
    a1 = 1 + delta_t*hbar*1j/(2*m)/(delta_x**2)
    a2 = -delta_t*hbar*1j/(4*m)/(delta_x**2)
    offdiag = np.full(N-1, a2, dtype=np.complex64)  # Lower diagonal (below main diagonal)
    maindiag = np.full(N, a1, dtype=np.complex64)     # Main diagonal
    A_band = np.zeros((3, N), dtype=np.complex128)
    A_band[0, 1:] = offdiag  # Upper diagonal (shifted by 1)
    A_band[1, :] = maindiag  # Main diagonal
    A_band[2, :-1] = offdiag  # Lower diagonal (shifted by -1)

    B = np.zeros((N, N), dtype=np.complex64)
    b1 = 1 - delta_t*hbar*1j/(2*m)/(delta_x**2)
    b2 = delta_t*hbar*1j/(4*m)/(delta_x**2)
    np.fill_diagonal(B, b1)
    np.fill_diagonal(B[1:], b2)
    np.fill_diagonal(B[:, 1:], b2)
    return A_band, B

# Initial wavefunction at t=0
psi_0 = np.exp(-(x - x_0)**2 / (2 * sigma**2)) * np.exp(1j * kappa * x)  # Initial wavefunction
A_band, B = matrices(N)

# Initialize VPython canvas
scene = canvas(title="Wavefunction Evolution", width=800, height=600, visible=False)
wave = curve(color=vector(0, 0, 1))  # Blue line for wavefunction
scene.camera.pos = vector(5E-9, 0.1, 0)  # Adjust camera position
scene.camera.axis = vector(-1E-9, -0.1, 0)

# Initialize wavefunction for time evolution
psi_t = psi_0.copy()

def wavefunction_evolution(psi_t):
    psi_t_new = solve_banded((1, 1), A_band, B @ psi_t)
    return psi_t_new
    
    
rate(30)
t_steps = 500
frames = []
psi_evolution = np.zeros((t_steps, N))
psi_evolution[0,:] = psi_0

for i in range(t_steps - 1):
    psi_evolution[i + 1, :] = wavefunction_evolution(psi_evolution[i, :])
	
fig, ax = plt.subplots()
ax.set_xlim(450, 650)
ax.set_ylim(-0.03, 0.15)
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return (line,)

def frame(i):
    line.set_data(x, psi_evolution[i, :])
    return (line,)

anim = plt.animation.FuncAnimation(fig, frame, init_func=init, frames=t_steps, interval=40, blit=True)
anim.save("Wavefunction_ev.gif", writer="pillow", fps=30)
