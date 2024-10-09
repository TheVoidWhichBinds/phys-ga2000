import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('signal.dat', sep='\s+', header=0) #importing data.

#plotting just the sample data.
plt.figure()
plt.figure(figsize=(10, 6)) 
plt.title('Signal', fontsize = 20)
plt.ylabel('Amplitude',fontsize = 14)
plt.xlabel('Time',fontsize = 14)
plt.plot(data['time'], data['signal'],'o', markersize = 5)
plt.savefig('Signal')


#asigning x and y to t(time) and a(amplitude).
t = data['time'].values
a = data['signal'].values

def polynomial_fit(t,n): #SVD to make a polynomial fit.
    ts = (t - np.mean(t))/np.std(t) #making the time data easier to work with.
    T_n = np.column_stack([ts**i for i in range(n+1)]) #creating design matrix.
    print(f'The condition number for the n = {n} polynomial fit is', np.linalg.cond(T_n)) #condition # of T_n

    #SVD decomposition:
    U, Sigma, V_T = np.linalg.svd(T_n, full_matrices=False)
    Sigma_inv = np.diag(1/Sigma)
    T_pseudoinv = V_T.T @ Sigma_inv @ U.T
    co = T_pseudoinv @ a
    poly_fit = T_n @ co #the polynomial fit.

    return poly_fit


t_fine = np.linspace(t.min(), t.max(), 1000) 
fig, axs = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)

#3rd Order Polynomial
axs[0].plot(t, a, 'o', markersize=3, label='Signal Data')
axs[0].plot(t_fine, polynomial_fit(t_fine, 3), label='3rd Order Polynomial Fit', color='r')
axs[0].set_title('3rd Order Polynomial')
axs[0].set_ylabel('Signal')
axs[0].legend()
print('The maximum residual for the n =3 polynomial is', np.max(a - polynomial_fit(t_fine,3)))

#15th Order Polynomial
axs[1].plot(t, a, 'o', markersize=3, label='Signal Data')
axs[1].plot(t_fine, polynomial_fit(t_fine, 15), label='15th Order Polynomial Fit', color='g')
axs[1].set_title('15th Order Polynomial')
axs[1].set_ylabel('Signal')
axs[1].legend()

#27th Order Polynomial
axs[2].plot(t, a, 'o', markersize=3, label='Signal Data')
axs[2].plot(t_fine, polynomial_fit(t_fine, 27), label='27th Order Polynomial Fit', color='m')
axs[2].set_title('27th Order Polynomial')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Signal')
axs[2].legend()

plt.tight_layout()
plt.savefig('SVD')

#Fourier SVD.
def Fourier_Series(harmonic,f0):
    F = np.zeros((np.size(t),2*harmonic+1))
    F[:,0] = 1 #first column is the constant a0.
    for h in range(1,harmonic+1):
        F[:, 2*h-1] = np.cos(2*np.pi*h*f0*t)  # Cosine terms
        F[:, 2*h] = np.sin(2*np.pi*h*f0*t)    # Sine terms
    
    co = np.linalg.pinv(F) @ a #coefficients found using pseudoinverse of F matrix.
    Fourier_fit = F @ co
    return Fourier_fit


plt.figure()
plt.figure(figsize=(10, 6)) 
plt.title('Signal', fontsize = 20)
plt.ylabel('Amplitude',fontsize = 14)
plt.xlabel('Time',fontsize = 14)

plt.plot(data['time'], data['signal'],'o', markersize = 5, label = 'Signal Data')
f_half = (t.max() - t.min())/2
plt.plot(t,Fourier_Series(3,f_half), color = 'm')
f_approx = 7.5/1e9
plt.plot(t_fine,np.convolve(Fourier_Series(7,f_approx), np.ones(10)/10, mode='same'), color = 'm', label = 'Fourier Series')
plt.legend()
plt.savefig('Fourier')

