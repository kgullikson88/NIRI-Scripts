from astropy.modeling import models, fitting
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.geometry import circular_overlap_grid
from photutils.aperture_funcs import get_phot_extents
import sys


def fit_gaussian_psf(data, x_mean=512, y_mean=512, x_stddev=3, y_stddev=3):
    """
    Fit a gaussian PSF to the data

    data: a 2d numpy array giving the flux at each pixel
    returns: A 2d numpy array with the gaussian model of the data
    """
    YY, XX = np.indices(data.shape)
    p_init = models.Gaussian2D(amplitude=data[x_mean][y_mean], 
                               x_mean=x_mean, y_mean=y_mean, 
                               x_stddev=x_stddev, y_stddev=y_stddev)
    fitter = fitting.LevMarLSQFitter()
    gauss = fitter(p_init, XX, YY, data)

    return gauss, gauss(XX, YY)


def remove_concentric(data, x_mean=512, y_mean=512, x_stddev=3, y_stddev=3, Nsig=10, dp=0.1, gauss=None):
    """
    This function first fits a gaussian PSF to the data to find the centroid.
    Then, it subtracts concentric apertures from the data rather than just 
    subtracting the gaussian model.
    """
    data = data.copy() # Don't overwrite the data
    nx, ny = data.shape
    
    if gauss is None:
        gauss, _ = fit_gaussian_psf(data, x_mean=x_mean, y_mean=y_mean, 
                                    x_stddev=x_stddev, y_stddev=y_stddev)
    centroid = [gauss.x_mean.value, gauss.y_mean.value]
    std = (gauss.x_stddev.value + gauss.y_stddev.value)/2.0
    x0, y0 = centroid
    
    radii = np.arange(dp, Nsig*std, dp)[::-1]
    
    # Get extents using the photutils function
    pos = np.atleast_2d(centroid)
    extents = np.zeros((len(pos), 4), dtype=int)

    extents[:, 0] = pos[:, 0] - max(radii) + 0.5
    extents[:, 1] = pos[:, 0] + max(radii) + 1.5
    extents[:, 2] = pos[:, 1] - max(radii) + 0.5
    extents[:, 3] = pos[:, 1] + max(radii) + 1.5

    _, extent, phot_extent = get_phot_extents(data, pos, extents)
    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent
    args = [x_pmin[0], x_pmax[0], y_pmin[0], y_pmax[0], x_max[0]-x_min[0], y_max[0]-y_min[0]]

    # Fit apertures
    psf = np.zeros(data.shape)
    for radius in radii:
        Rin = radius - dp
        Rout = radius
        if Rin > dp/10.0:
            ap = CircularAnnulus(centroid, r_in=Rin, r_out=Rout)
        else:
            ap = CircularAperture(centroid, r=Rout)
        phot = aperture_photometry(data, ap)
        avg_flux = phot['aperture_sum'].item()/ap.area()
        
        outer = circular_overlap_grid(*args, r=Rout, use_exact=1, subpixels=5)
        if Rout-dp > dp/10.0:
            inner = circular_overlap_grid(*args, r=Rin, use_exact=1, subpixels=5)
        else:
            inner = 0.0
        annulus_overlap = outer - inner
        psf[y_min[0]:y_max[0], x_min[0]:x_max[0]] += annulus_overlap * avg_flux
    return data - psf, psf






if __name__ == '__main__':
    hdulist = fits.open(sys.argv[1])
    data = hdulist[1].data
    resid, psf = remove_concentric(data.copy())
    plt.imshow(resid)
    plt.colorbar()
    plt.show()