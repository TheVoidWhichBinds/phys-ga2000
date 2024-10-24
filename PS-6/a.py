from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

print("Sliced logwave:", np.size(logwave[1:10]))
print("Sliced flux:", np.size(flux[1]))


plt.figure()
plt.title('Central Optical Spectra', fontsize = 17)
plt.xlabel('log lambda UNITS', fontsize = 14)
plt.ylabel('flux UNITS', fontsize = 14)
#plt.scatter(logwave[1:10], flux[1:10], marker = 'o', markersize = 4, color = 'm')
plt.savefig('COS')



hdu_list.close()

