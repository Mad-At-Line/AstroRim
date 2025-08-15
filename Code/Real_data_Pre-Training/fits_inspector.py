from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

file_path = r"C:\Users\mythi\.astropy\Code\Fits_work\4.5_fits_dataset1\simulation4.5_0000.fits"

with fits.open(file_path) as hdul:
    hdul.info()  # See the structure

    # Select HDU with image (try 1 for GT, 2 for LENSED)
    # image_data = hdul[1].data  # GT
    image_data = hdul[2].data  # LENSED

# Rescale for visibility
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(image_data)

# Display
plt.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.show()
