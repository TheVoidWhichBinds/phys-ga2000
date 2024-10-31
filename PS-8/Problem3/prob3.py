import numpy as np
import matplotlib.pyplot as plt


#import dow.txt
pts = len(dow)
#a )

plt.title('Dow Jones Industrial Average')
plt.xlabel('')
plt.ylabel('Daily Closing Value ($)')
plt.plot()

#b )

c = np.fft.rfft()

#c )

c[10:100,:] = 0

#d )

dowf = np.fft.irfft(c)
#copy part a plot details
plt.plot()
plt.plot( ,dowf, color = 'm')