import numpy as np
import matplotlib.pyplot as plt
import time


def matrixx(N,A,B): #for-loop calculation of a matrix product.
    F = np.zeros((N,N)) #initializes product matrix.
    startx = time.perf_counter() #start timer.
    for i in range(N):
        for j in range(N):
            for k in range(N):
                F[i,j] += A[i,k]*B[k,j]
    endx = time.perf_counter() #end timer after all loops have been executed.
    return (endx - startx) #time taken to run loop.

def matrixdot(N,A,B): #NumPy dot product calculation of a matrix product.
    startdot = time.perf_counter() #start timer.
    D = np.dot(A,B)
    enddot = time.perf_counter() #end timer after np.dot executed.
    return (enddot - startdot) #time taken to run calc.


N_max = 400 #maximum NxN matrix dimensions.
steps = int(N_max/10) #step size in matrix dimensions.
matrix_sizes = np.arange(10,N_max,steps) #array of matrix dimension(s) N.
timex_avg = np.zeros(len(matrix_sizes)) #initializes array to store time taken for size NxN matrices.
timedot_avg = np.zeros(len(matrix_sizes))
iavg = 5 #number of runs averaged over.

for i,N in enumerate(matrix_sizes): #i index matches entries number in matrix_sizes.
    #initializes random NxN matrices with values between -1 and 1:
    A = np.random.rand(N,N)-1
    B = np.random.rand(N,N)-1
    #initializes time taken:
    timex = 0
    timedot = 0
    #for a fixed N, runs and averages over iavg samples:
    for a in range(iavg): 
        timex += matrixx(N,A,B)
        timedot += matrixdot(N,A,B)
    timex_avg[i] = timex/iavg #stores time taken for NxN matrices multiplied at each time i.
    timedot_avg[i] = timedot/iavg

fig, calcx = plt.subplots()
calcx.scatter(matrix_sizes, timex_avg, color='b', marker='o', label='For-Loop Product')
calcx.set_xlabel('Square Matrix Dimension N', fontsize=14)
calcx.set_ylabel('Calculation Time (seconds)', fontsize=10)
calcx.set_ylim(0, max(timex_avg) * 1.1) 

# Create a second y-axis since the np.dot product takes significantly less time.
calcdot = calcx.twinx() 
calcdot.scatter(matrix_sizes, timedot_avg, color='g', marker='o', label='NumPy Dot Product')
calcdot.set_ylim(0, max(timedot_avg) * 1.1) 
calcdot.set_yscale('log')
calcdot.set_ylim(1e-5, max(timedot_avg) * 2)

calcx.legend(loc='upper left')
calcdot.legend(loc='upper right')

plt.title('Matrix Multiplication Calculation Load', fontsize=17)
plt.savefig('Matrices')
