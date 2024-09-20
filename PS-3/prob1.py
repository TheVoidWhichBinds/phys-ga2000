import numpy as np
import matplotlib.pyplot as plt

N_max = 100
steps = int(N_max/10)

def matrixx(N):
    opx = 0
    A = np.ones([N,N],float)
    B = np.ones([N,N],float)
    C = np.zeros([N,N],float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += A[i,k]*B[k,j]
                opx += 1
    return opx

def scalingx():
    i = 0
    opscalex = np.zeros((steps,2))
    for N in range(0,N_max,steps):
        opscalex[i] = [N,matrixx(N)]
        i += 1
    return opscalex
calcx = np.array(scalingx())


def matrixdot(N):
    opdot = 0
    A = np.ones([N, N], float)
    B = np.ones([N, N], float)
    C = np.zeros([N, N], float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
                opdot += 1
    return opdot

def scalingdot():
    i = 0
    opscaledot = np.zeros((steps, 2))
    for N in range(0, N_max, steps):
        opscaledot[i] = [N, matrixdot(N)]
        i += 1
    return opscaledot
calcdot = np.array(scalingdot())




plt.figure(figsize=(9,9))
plt.scatter(calcx[:,0],calcx[:,1],color='b',marker='o')
n = np.linspace(min(calcx[:, 0]), max(calcx[:, 0]),500)
y = n**3
plt.plot(n,y,color='r')
plt.scatter(calcdot[:,0],calcdot[:,1],color='g',marker='o')
plt.title('Matrix Multiplication Calculation Load', fontsize=19)
plt.xlabel('Square Matrix Dimension N', fontsize=14)
plt.ylabel('Total Calculations', fontsize=14)
#plt.legend()
plt.savefig('MMS')

