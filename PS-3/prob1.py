import numpy as np
import matplotlib.pyplot as plt
import time


def matrixx(N,A,B):
    F = np.zeros((N,N))
    startx = time.time()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                F[i,j] += A[i,k]*B[k,j]
    endx = time.time()
    return (endx - startx)

def matrixdot(N,A,B):
    startdot = time.time()
    D = np.dot(A,B)
    enddot = time.time()
    return (enddot - startdot)


N_max = 600
steps = int(N_max/10)
matrix_sizes = np.arange(10,N_max,steps)
timex = np.zeros(len(matrix_sizes))
timedot = np.zeros(len(matrix_sizes))

for i,N in enumerate(matrix_sizes):
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    C = np.random.rand(N,N)
    timex[i] = matrixx(N,A,B)
    timedot[i] = matrixdot(N,A,B)


plt.scatter(matrix_sizes,timex,color='b',marker='o')
#plt.scatter(matrix_sizes,timedot,color='g',marker='o')

n = np.linspace(10,N_max,steps)
y = n**3
plt.plot(n,y,color='r')


plt.title('Matrix Multiplication Calculation Load', fontsize=19)
plt.xlabel('Square Matrix Dimension N', fontsize=14)
plt.ylabel('Total Calculations', fontsize=14)
#plt.legend()
plt.show()
