import numpy as np
import matplotlib.pyplot as plt
import time
import time


def matrixx(N,A,B):
    F = np.zeros((N,N))
    startx = time.perf_counter()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                F[i,j] += A[i,k]*B[k,j]
    endx = time.perf_counter()
    return (endx - startx)

def matrixdot(N,A,B):
    startdot = time.perf_counter()
    D = np.dot(A,B)
    enddot = time.perf_counter()
    return (enddot - startdot)


N_max = 200
steps = int(N_max/10)
matrix_sizes = np.arange(10,N_max,steps)
timex_avg = np.zeros(len(matrix_sizes))
timedot_avg = np.zeros(len(matrix_sizes))
iavg = 5
for i,N in enumerate(matrix_sizes):
    A = np.random.rand(N,N)-1
    B = np.random.rand(N,N)-1
    C = np.random.rand(N,N)-1
    timex = 0
    timedot = 0
    for a in range(iavg):
        timex += matrixx(N,A,B)
        timedot += matrixdot(N,A,B)
    timex_avg[i] = timex/iavg
    timedot_avg[i] = timedot/iavg

fig, calcx = plt.subplots()
calcx.scatter(matrix_sizes, timex_avg, color='b', marker='o', label='For-Loop Product')
calcx.set_xlabel('Square Matrix Dimension N', fontsize=14)
calcx.set_ylabel('Calculation Time (seconds)', fontsize=10)
calcx.set_ylim(0, max(timex_avg) * 1.1) 

# Create a second y-axis
calcdot = calcx.twinx()  # Instantiate a second axes that shares the same x-axis
calcdot.scatter(matrix_sizes, timedot_avg, color='g', marker='o', label='NumPy Dot Product')
calcdot.set_ylim(0, max(timedot_avg) * 1.1) 
calcdot.set_yscale('log')
calcdot.set_ylim(1e-5, max(timedot_avg) * 2)
# Add legends

calcx.legend(loc='upper left')
calcdot.legend(loc='upper right')

plt.title('Matrix Multiplication Calculation Load', fontsize=17)
plt.savefig('Matrices')
