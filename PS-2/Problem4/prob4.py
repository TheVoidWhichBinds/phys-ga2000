import numpy as np
import matplotlib.pyplot as plt

iter_max = 100 #maximum iteration count.
N = 300 #resolution/gridsize.

def seedconstant(N): #defines the initial seed from which the Mandelbrot series propagates.
    x = np.linspace(-2,2,N, dtype=np.float64) #creates an x-axis array.
    y = np.linspace(-2,2,N, dtype=np.float64) #creates a y-axis array.
    Re,Im = np.meshgrid(x,y) #combines x and y axes into x2 2-D grids.
    C = Re + 1j*Im #combines the two grids into a single 2D grid with y values made imaginary.
    return C

C = seedconstant(N) 

def mandelbrot(C,iter_max): #calculates the Mandelbrot set.
    Z = C #initiates the complex plane.
    for i in range(iter_max): #iteration can mathematically go to infinity, but we clamp it.
            if np.abs(Z) > 2: #define locations in the complex plane outside the Mandelbrot set.
                return i #returns number of iterations undergone before leaving the set.
            Z = Z**2 + C #formula for the set.
    return iter_max #since iteration count can go to infinity, if a location stays wihtin the set, it will have undergone our variable max iterations.

itermap = np.zeros((N,N))#make an iteration map whose array dimensions match C's spacing.
for x in range(N): #real value x-axis.
     for y in range(N): #imaginary value y-axis.
        itermap[x,y] = mandelbrot(C[x,y],iter_max)#mapping mandelbrot iteration count over entire Re,Im array.


plt.figure(figsize=(9,9))
plt.imshow(itermap, cmap='gist_stern', extent=[-2,2,-2,2])
plt.colorbar(label="Iteration count")
plt.title("Mandelbrot Set", fontsize=19)
plt.xlabel('Real', fontsize=15)
plt.ylabel('Imaginary', fontsize=15)
plt.savefig('Mandelbrot')

