import numpy as np
import matplotlib.pyplot as plt

piano = np.loadtxt('piano.txt')
trumpet = np.loadtxt('trumpet.txt')


#a ) 
def FFT(instrument, instrument_name):
    #Plots data.
    plt.figure(figsize=(10, 6))
    plt.title(f'{instrument_name} Note', fontsize=18)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.plot(np.arange(0, len(instrument)), instrument, color='m')
    plt.savefig(f'{instrument_name}_Signal.png')

    #X-axis of Fourier plots must be altered by the sampling rate in order to read frequency.
    sampling_rate = 44100
    frequencies = np.fft.rfftfreq(len(instrument[0:10000]), 1/sampling_rate) #Frequency spectrum.
    c = np.fft.rfft(instrument[0:10000]) #Fast Fourier Transform to extract the coefficients.

    #Plotting Fourier coefficients on a frequency vs. magnitude graph.
    plt.figure(figsize=(10, 6))
    plt.title(f'{instrument_name} Note Fourier Coefficients', fontsize=18)
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.xlim(0,5000)
    plt.ylabel('Magnitude', fontsize=16)
    plt.plot(frequencies, abs(c))
    plt.savefig(f'{instrument_name}_FFT.png') 


    #b ) 
    #Finds the fundamental frequency of the instrument.
    threshold = 0.4 * 1e7
    for i, magnitude in enumerate(np.abs(c)): #Looks over all coefficients.
        if magnitude > threshold: #We only care about coefficients of significant (arbitrary) contribution.
            print(f'The note being played on the {instrument_name} has a fundamental frequency of', frequencies[i],'Hz')
            break #Only reads off the fundamental frequency, which is all that is needed to find the note.
    

FFT(piano, 'Piano')
FFT(trumpet, 'Trumpet')



