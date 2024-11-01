import numpy as np
import matplotlib.pyplot as plt


dow = np.loadtxt('dow.txt')

#a )
#Plots Dow Jones data.
plt.title('Dow Jones Industrial Average')
plt.xlabel('Day') #since 2006?
plt.ylabel('Daily Closing Value ($)')
plt.plot(dow, color = 'g')
plt.savefig('DJ')

#b )
#Extracting Fourier coefficients using Fast Fourier Transform, only the real component so we don't have redundancy.
c = np.fft.rfft(dow)

#c )

def reconstructed(percentage):
    
    nonzero = int(percentage*len(c)) #Calculates the number of indices of c we want to keep based on the input percentage.
    c_mod = np.zeros(len(c), dtype=complex) #Initializes new coefficient array.
    c_mod[0:nonzero] = np.abs(c[0:nonzero]) #Fills the first {percentage}% of the new coefficient array with the c.
    dow_ifft = np.fft.irfft(c_mod) #Reconstructs the data based on the new coeffficent array.

    #Plots the reconstructed data over the original data. Setting percentage to 100% verifies that the original data is recovered.
    plt.figure()
    plt.title('Dow Jones Industrial Average with Fourier Reconstruction')
    plt.xlabel('Day') #since 2006?
    plt.ylabel('Daily Closing Value ($)')
    plt.plot(dow, color = 'g', label = 'original data')
    plt.plot(dow_ifft, color='m', label=f'First {percentage*100:.1f}% of Fourier coefficients')
    plt.legend()
    plt.savefig(f'DJR_{int(percentage * 100)}%.png')  # Add file extension
    plt.close()  # Close the plot


#d )
reconstructed(0.1)

#e )
reconstructed(0.02)


