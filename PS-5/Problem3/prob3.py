import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('signal.dat', sep='\s+', header=0) 

plt.figure()
plt.figure(figsize=(10, 6)) 
plt.title('Signal', fontsize = 20)
plt.ylabel('Amplitude',fontsize = 14)
plt.xlabel('Time',fontsize = 14)
plt.scatter(data['time'], data['signal'])
plt.savefig('Signal')



t = data['time'].values
a = data['signal'].values

def polynomial_fit(t,n):
    ts = (t - np.mean(t))/np.std(t)
    T_n = np.column_stack([ts**i for i in range(n)])

    U, Sigma, V_T = np.linalg.svd(T_n, full_matrices=False)
    Sigma_inv = np.diag(1/Sigma)
    T_pseudoinv = V_T.T @ Sigma_inv @ U.T
    co = T_pseudoinv @ a
    poly_fit = np.dot(T_n,co)

    return poly_fit

    residuals = a - poly_fit


plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Best 3rd Order Polynomial Fit to Signal')

plt.plot(t, a, 'o', label='Signal Data')
plt.scatter(t, polynomial_fit(t,3), label='3rd Order Polynomial Fit', color='red')
plt.legend()


plt.subplot(2, 1, 2)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Best nth Order Polynomial Fit to Signal')

plt.plot(t, a, 'o', label='Signal Data')
plt.scatter(t, polynomial_fit(t,6), label='6th Order Polynomial Fit', color='blue')
plt.scatter(t, polynomial_fit(t,10), label='10th Order Polynomial Fit', color='m')
plt.legend()
plt.tight_layout()
plt.savefig('SVD')
