import numpy as np
import matplotlib.pyplot as plt

#import data

#a ) 
def FFT(y):
    c = np.fft(y,instrument)
    plt.title('%s Signal',instrument)
    plt.xlabel('')
    plt.ylabel('')
    plt.plot(abs(c))

FFT()

#b ) 