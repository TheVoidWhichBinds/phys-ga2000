from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from time import time

#a ) 

hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data[0:500,:]
galaxies = len(flux)
wavelengths = len(logwave)

plt.figure(figsize = (12,9))
plt.title('Central Optical Spectra', fontsize = 17)
plt.xlabel(r'Wavelength $\log_{10} \lambda$ (Å)', fontsize=14)
plt.ylabel(r'Flux $\frac{10^{-17} \text{ erg}}{\text{s} \, \text{cm}^2 \, \text{Å}}$', fontsize=14)
plt.plot(logwave, flux[0,:], color = 'm')
plt.savefig('COS')



#b ) 

flux_sum = np.sum(flux, axis=1)
flux_normed = flux/flux_sum[:,np.newaxis]



#c )

flux_mean = np.mean(flux_normed, axis=0)
flux_R = flux_normed - flux_mean



#d )


C = flux_R @ flux_R.T
Ceigval, Ceigvec = eig(C)
Ceigval_sorted = np.argsort(Ceigval)[::-1]
Ceigvec_sorted = Ceigvec[:, Ceigval_sorted]
eigenspectra = flux_R.T @ Ceigvec_sorted[:, :5]


plt.figure(figsize=(12, 8))
plt.title('First 5 Eigenspectra')
plt.xlabel(r'Wavelength $\log_{10} \lambda$ (Å)')
plt.ylabel('Flux')
plt.grid()
# Plot the first 5 eigenspectra (wavelength vs flux)
for i in range(5):
    plt.plot(logwave, eigenspectra[:, i], label=f'Eigenspectrum {i+1}')
plt.legend()
plt.savefig('PCA')



# e) Percentage of Variance Explained

# Calculate the total sum of the eigenvalues (total variance)
total_variance = np.sum(Ceigval)

# Compute the variance explained by each component (normalized by total variance)
explained_variance = (Ceigval / total_variance) * 100

# Sort the explained variance in the same order as eigenvalues
explained_variance_sorted = explained_variance[Ceigval_sorted]

# Plot the variance explained by the first few components
plt.figure(figsize=(12, 8))
plt.title('Variance Explained by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Percentage of Variance Explained')
plt.bar(range(1, 11), explained_variance_sorted[:10].real)  # Use real part in case of complex numbers
plt.savefig('explained_variance')




hdu_list.close()