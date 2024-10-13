from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#a

hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data
galaxies = len(flux)
wavelengths = len(logwave)

print("Sliced logwave:", np.shape(logwave))
print("Sliced flux:", np.shape(flux))

plt.figure(figsize = (12,9))
plt.title('Central Optical Spectra', fontsize = 17)
plt.xlabel(r'Wavelength $\log_{10} \lambda$ (Å)', fontsize=14)
plt.ylabel(r'Flux $\frac{10^{-17} \text{ erg}}{\text{s} \, \text{cm}^2 \, \text{Å}}$', fontsize=14)
plt.plot(logwave, flux[0,:4001], color = 'm')
plt.savefig('COS')


#b

def flux_normalization():
    flux_sum = np.zeros(galaxies)
    for i in range(galaxies):
        flux_sum[i] = np.sum(flux[i,:])
    flux_normed = np.dot(flux,1/flux_sum)
    return flux_normed


#c

def flux_averaging():
    flux_mean = np.zeros(galaxies)
    for i in range(galaxies):
        flux_mean[i] = np.sum(flux[i,:])/wavelengths
    return flux_mean

flux_residual = flux_normalization() - flux_averaging()


hdu_list.close()

