import numpy as np
import timeit

n = 100

def Madelung_forloop():
    M_for = 0
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            for k in range(-n,n+1):
                if (i == j == k == 0):
                    element = 0
                else:
                    element = ((-1)**(i+j+k))/np.sqrt(i**2 + j**2 + k**2)
                M_for = M_for + element
    return M_for


def Madelung_array():
    d = np.arange(-n,n+1)
    i,j,k = np.meshgrid(d,d,d)

    origin = (i == 0) & (j == 0) & (k == 0)
    M_ijk = ((-1)**np.abs(i+j+k)[~origin])/np.sqrt(i**2 + j**2 + k**2)[~origin]
    M_array = M_ijk.sum()
    return M_array

time_forloop = timeit.timeit(Madelung_forloop, number=2)  # Adjust number for repetitions
time_array = timeit.timeit(Madelung_array, number=2)  # Adjust number for repetitions

print("The for-loop code takes (seconds):", time_forloop)
print("The array code takes (seconds):", time_array)
