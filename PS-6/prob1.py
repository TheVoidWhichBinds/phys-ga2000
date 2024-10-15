from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from time import time

hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data[0:500,:]
galaxies = len(flux)
wavelengths = len(logwave)

plt.figure(figsize=(12, 9))

#creates 4 subplots with different flux rows and colors
for i, color in enumerate(['m', 'b', 'g', 'r']):
    plt.subplot(2, 2, i+1)  #2x2 grid, selecting subplot i+1
    plt.title(f'Optical Spectra {i+1}', fontsize=15)
    plt.xlabel(r'Wavelength $\log_{10} \lambda$ (Å)', fontsize=12)
    plt.ylabel(r'Flux $\frac{10^{-17} \text{ erg}}{\text{s} \, \text{cm}^2 \, \text{Å}}$', fontsize=12)
    plt.plot(logwave, flux[i, :], color=color)

plt.tight_layout()
plt.savefig('COS')



#b ) Normalization

flux_sum = np.sum(flux, axis=1)
flux_normed = flux/flux_sum[:,np.newaxis] #flux at each wavelength divided by total over all wavelengths.



#c ) Offsetting Around Zero

flux_mean = np.mean(flux_normed, axis=0) #mean normalized flux over all galaxies.
flux_R = flux_normed - flux_mean #residuals of the flux.



#d ) Principal Component Analysis

C = flux_R @ flux_R.T #covariance matrix.
Ceigval, Ceigvec = eig(C) #eigenvalues and functions extracted.
Ceigval_sorted = np.argsort(Ceigval)[::-1] #order reversed to get values from largest to smallest.
Ceigvec_sorted = Ceigvec[:, Ceigval_sorted]
eigenspectra = flux_R.T @ Ceigvec_sorted[:, :5]

plt.figure(figsize=(12, 8))
plt.title('First 5 Eigenspectra', fontsize = 20)
plt.xlabel(r'Wavelength $\log_{10} \lambda$ (Å)', fontsize = 16)
plt.ylabel(r'Flux $\frac{10^{-17} \text{ erg}}{\text{s} \, \text{cm}^2 \, \text{Å}}$', fontsize=16)
plt.grid()
#plots the first 5 eigenspectra (wavelength vs flux).
for i in range(5):
    plt.plot(logwave, eigenspectra[:, i], label=f'Eigenspectrum {i+1}')
plt.legend()
plt.savefig('PCA')



# e ) Singular Value Decomposition (SVD)

U, S, VT = np.linalg.svd(flux_R, full_matrices=False) #SVD matrices
Reconstructed = (U[:, :5] @ np.diag(S[:5]) @ VT[:5, :]) #definition of the three matrices multiplied.

plt.figure(figsize=(12, 8))
plt.title('Original vs Reconstructed Spectra', fontsize = 20)
plt.xlabel(r'Wavelength $\log_{10} \lambda$ (Å)', fontsize = 16)
plt.ylabel(r'Flux $\frac{10^{-17} \text{ erg}}{\text{s} \, \text{cm}^2 \, \text{Å}}$', fontsize=16)
plt.plot(logwave, flux_R[0, :] + flux_mean, label='Original', color='m')
plt.plot(logwave, Reconstructed[0, :] + flux_mean, label='Reconstructed (5 components)', linestyle='--', color='c')
plt.legend()
plt.savefig('SVD')



#f ) 

hdu_list.close()
