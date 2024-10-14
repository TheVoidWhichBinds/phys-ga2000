from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig

#a

hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data
galaxies = len(flux)
wavelengths = len(logwave)

plt.figure(figsize = (12,9))
plt.title('Central Optical Spectra', fontsize = 17)
plt.xlabel(r'Wavelength $\log_{10} \lambda$ (Å)', fontsize=14)
plt.ylabel(r'Flux $\frac{10^{-17} \text{ erg}}{\text{s} \, \text{cm}^2 \, \text{Å}}$', fontsize=14)
plt.plot(logwave, flux[0,:4001], color = 'm')
plt.savefig('COS')


#b

def flux_normalization():
    flux_sum = np.sum(flux, axis=1)
    flux_normed = flux/flux_sum[:,np.newaxis]
    return flux_normed


#c

def flux_averaging():
    flux_mean = np.mean(flux_normalization(), axis=0)
    return flux_mean

flux_residual = flux_normalization() - flux_averaging()


#d

C = flux_residual @ flux_residual.T
Ceigval, Ceigvec = eig(C)


hdu_list.close()

