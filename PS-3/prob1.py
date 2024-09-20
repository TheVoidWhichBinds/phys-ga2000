import numpy as np
import matplotlib.pyplot as plt
from time import time
N = 100
steps = int(N/10)

A = np.random.rand([N,N],float)
B = np.random.rand([N,N],float)
C = np.random.rand([N,N],float)

def matrixx(N):
    F = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                F[i,j] += A[i,k]*B[k,j]
    return F
timex = time.time(matrixx(N))
print(timex)


#make matrices random

D = np.dot(A,B)

timedot = time.time(matrixdot(N))




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

