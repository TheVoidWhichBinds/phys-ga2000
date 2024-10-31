import numpy as np
import matplotlib.pyplot as plt

piano = np.loadtxt('piano.txt')
trumpet = np.loadtxt('trumpet.txt')


#a ) 
def FFT(instrument, instrument_name):
 
    plt.figure(figsize=(18, 12))
    plt.title(f'{instrument_name} Note', fontsize=20)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    
    plt.plot(np.arange(0, len(instrument)), instrument, color='b')
    plt.savefig(f'{instrument_name}_Signal.png')


    c = np.fft.fft(instrument[0:10000])

    plt.figure(figsize=(18, 12))
    plt.title(f'{instrument_name} Note Fourier Coefficients', fontsize=20)
    plt.xlabel('Frequency', fontsize=16)
    plt.ylabel('Magnitude', fontsize=16)
    plt.plot(abs(c))
    plt.savefig(f'{instrument_name}_FFT.png')  

FFT(piano, 'Piano')
FFT(trumpet, 'Trumpet')



#b ) 