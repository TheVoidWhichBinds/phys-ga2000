import numpy as np
import timeit


n = 100 #half the number of atoms along one axis (total number of atoms is (2n)^3)

def Madelung_forloop(): #calculates the Madelung constant using a for-loop
    M_for = 0 #initiates
    for i in range(-n,n+1): #iterates along x-axis
        for j in range(-n,n+1): #iterates along y-axis
            for k in range(-n,n+1): #iterates along z-axis
                if (i == j == k == 0): #excludes the atom at the origin
                    element = 0
                else:
                    element = ((-1)**(i+j+k))/np.sqrt(i**2 + j**2 + k**2) #Madelung constant of individual atom.
                M_for = M_for + element #sums over contributions of each atom in lattice.
    return M_for


def Madelung_array(): #calculates the Madelung constant using an array method.
    d = np.arange(-n,n+1) #creates an array (of atoms along a Cartesian axis).
    i,j,k = np.meshgrid(d,d,d) #combines all indeces of the x,y, and z axes into a x3 3D grids of values (coordinates).

    origin = (i == 0) & (j == 0) & (k == 0) #relates origin to each 3D array (0 coordinates of each).
    M_ijk = ((-1)**np.abs(i+j+k)[~origin])/np.sqrt(i**2 + j**2 + k**2)[~origin] #3D array of size (2n)^3 with values equal to the contribution to the constant at each coordinate.
    M_array = M_ijk.sum() #sums over all values in the Madelung-coordinate array.
    return M_array

time_forloop = timeit.timeit(Madelung_forloop, number=2)  #times for-loop calculation.
time_array = timeit.timeit(Madelung_array, number=2)  #times array calculation.

print("The for-loop code takes (seconds):", time_forloop)
print("The array code takes (seconds):", time_array)
