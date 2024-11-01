import numpy as np
import matplotlib.pyplot as plt


dow = np.loadtxt('dow.txt')

#a )

plt.title('Dow Jones Industrial Average')
plt.xlabel('Day') #since 2006?
plt.ylabel('Daily Closing Value ($)')
plt.plot(dow, color = 'm')
plt.savefig('Dow Jones from late 2006 to end of 2010')

#b )

c = np.fft.rfft(dow)

#c )

ten_c = int(0.1*len(c))
c_mod = np.zeros(len(c))
c_mod[0:ten_c] = c[0:ten_c]

#d )

#dow_ifft = np.fft.irfft(c)
#plt.title('Dow Jones Industrial Average')
#plt.xlabel('Day') #since 2006?
#plt.ylabel('Daily Closing Value ($)')
#plt.plot(dow, color = 'm', label = 'original data')
#plt.plot(dow_ifft, color = 'r', label = 'Fourier-reconstructed data')
#plt.savefig('Dow Jones from late 2006 to end of 2010')
#plt.plot()
#plt.plot( ,dowf, color = 'm')