import numpy as np
import matplotlib.pyplot as plt


dow = np.loadtxt('dow.txt')

#a )

plt.title('Dow Jones Industrial Average')
plt.xlabel('Day') #since 2006?
plt.ylabel('Daily Closing Value ($)')
plt.plot(dow, color = 'm')
plt.savefig('DJ')

#b )

c = np.fft.rfft(dow)

#c )

ten_c = int(0.1*len(c))
c_mod = np.zeros(len(c))
c_mod[0:ten_c] = np.abs(c[0:ten_c])

#d )

dow_ifft = np.fft.irfft(c_mod)
plt.title('Dow Jones Industrial Average with Fourier Reconstruction')
plt.xlabel('Day') #since 2006?
plt.ylabel('Daily Closing Value ($)')

plt.plot(dow, color = 'm', label = 'original data')
plt.plot(dow_ifft, color = 'r', label = 'Fourier-reconstructed data')
plt.legend()
plt.savefig('DJR')