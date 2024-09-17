import numpy as np
import matplotlib.pyplot as plt

iter_max = 100
N = 300

def seedconstant(N):
    x = np.linspace(-2,2,N, dtype=np.float64) 
    y = np.linspace(-2,2,N, dtype=np.float64)
    Re,Im = np.meshgrid(x,y)
    C = Re + 1j*Im
    return C

C = seedconstant(N)

def mandelbrot(C,iter_max):
    Z = C
    for i in range(iter_max):
            if np.abs(Z) > 2:
                return i
            Z = Z**2 + C
    return iter_max

itermap = np.zeros((N,N))#make an iteration map whose array dimensions match C's spacing.
for x in range(N): #real value x-axis.
     for y in range(N): #imaginary value y-axis.
        itermap[x,y] = mandelbrot(C[x,y],iter_max)#mapping mandelbrot iter count over entire Re,Im array.


plt.figure(figsize=(10, 10))
plt.imshow(itermap, cmap='inferno', extent=[-2,2,-2,2])
plt.colorbar(label="Iteration count")
plt.title("Mandelbrot Set")
plt.savefig('Mandelbrot')

