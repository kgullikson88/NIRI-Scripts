from astropy.modeling import models, fitting
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.geometry import circular_overlap_grid
from photutils.aperture_funcs import get_phot_extents
from matplotlib.colors import SymLogNorm
from matplotlib import cm
import sys


def fit_gaussian_psf(data, x_mean=512, y_mean=512, x_stddev=3, y_stddev=3):
    """
    Fit a gaussian PSF to the data

    data: a 2d numpy array giving the flux at each pixel
    returns: A 2d numpy array with the gaussian model of the data
    """
    YY, XX = np.indices(data.shape)
    XX, YY = np.indices(data.shape)
    p_init = models.Gaussian2D(amplitude=data[y_mean][x_mean], 
                               x_mean=x_mean, y_mean=y_mean, 
                               x_stddev=x_stddev, y_stddev=y_stddev)
    fitter = fitting.LevMarLSQFitter()
    gauss = fitter(p_init, XX, YY, data)

    return gauss, gauss(XX, YY)


@models.custom_model
def Moffat2D_Azimuthal(x, y, amplitude=1.0, x_0=512.0, y_0=512.0, gamma=6.0,
           alpha=1.0):
    """Two dimensional Moffat function that assumes the scale parameters are the same for both axes"""
    rr_gg = ((x - x_0) ** 2 + (y - y_0) ** 2) / gamma ** 2
    return amplitude * (1 + rr_gg) ** (-alpha)


@models.custom_model
def Moffat2D(x, y, amplitude=1.0, x_0=512.0, y_0=512.0, x_gamma=6.0, y_gamma=6.0,
             alpha=1.0):
    """Two dimensional Moffat function with different scale parameters for both axes"""
    rr_gg = (x - x_0) ** 2 / x_gamma**2 + (y - y_0) ** 2 / y_gamma ** 2
    return amplitude * (1 + rr_gg) ** (-alpha)


def fit_moffat_psf(data, x_mean=512, y_mean=512, x_fwhm=6, y_fwhm=6):
    """
    Fit a moffat function to the PSF. A moffat function is given by

    $I_r = I_0 [1+(r/\theta)^2]^{-\Beta}

    Where $I_0$ is the intensity at the center of the PSF, $\theta$ is the
    half-width at half maximum in the absense of atmospheric scattering, and
    $\Beta$ is the atmospheric scattering coefficient.
    """
    YY, XX = np.indices(data.shape)
    XX, YY = np.indices(data.shape)
    p_init = Moffat2D(amplitude=data[y_mean][x_mean], 
                      x_0=x_mean, y_0=y_mean, 
                      x_gamma=6.0, y_gamma=6.0, alpha=1.0)

    fitter = fitting.LevMarLSQFitter()
    moffat = fitter(p_init, XX, YY, data)

    return moffat, moffat(XX, YY)



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






def old_main():
    hdulist = fits.open(sys.argv[1])
    data = hdulist[1].data
    resid, psf = remove_concentric(data.copy())
    plt.imshow(resid)
    plt.colorbar()
    plt.show()


def make_companion(data, x0, y0, sep=5.0, sig=3.0, delta_mag=3):
    flux_ratio = 10**(-delta_mag/2.5)
    prim_flux = data[y0][x0]
    print(prim_flux, flux_ratio)
    center = (x0-sep, y0)

    YY, XX = np.indices(data.shape)
    companion = prim_flux*flux_ratio * np.exp(-0.5*((XX-center[0])**2 + (YY-center[1])**2) / sig**2)
    return companion



def main():
    # Compare the different fitting methods.
    hdulist = fits.open(sys.argv[1])
    data = hdulist[1].data

    # Add a synthetic companion to the data
    data += make_companion(data, 519, 509, sep=10, delta_mag=4)

    gauss, gauss_psf = fit_gaussian_psf(data)
    moffat, moffat_psf = fit_moffat_psf(data)
    concentric, concentric_psf = remove_concentric(data.copy())

    # Plot
    limits = (512-20, 512+20)
    fig1, ax1 = plt.subplots()
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(111)

    normalizer = SymLogNorm(linthresh=1e-3, linscale=0.2, clip=True)
    colormap = cm.hot
    im1 = ax1.imshow(data[limits[0]:limits[1], limits[0]:limits[1]], cmap=colormap, norm=normalizer)
    #ax1.set_xlim(limits)
    #ax1.set_ylim(limits)
    ax1.set_title('Data')
    plt.colorbar(im1, ax=ax1)

    print(gauss)
    im2 = ax2.imshow((data-gauss_psf)[limits[0]:limits[1], limits[0]:limits[1]], cmap=colormap, norm=normalizer)
    #ax2.set_xlim(limits)
    #ax2.set_ylim(limits)
    ax2.set_title('Gaussian PSF subtracted')
    plt.colorbar(im2, ax=ax2)

    print(moffat)
    im3 = ax3.imshow((data-moffat_psf)[limits[0]:limits[1], limits[0]:limits[1]], cmap=colormap, norm=normalizer)
    #ax3.set_xlim(limits)
    #ax3.set_ylim(limits)
    ax3.set_title('Moffat PSF subtracted')
    plt.colorbar(im3, ax=ax3)

    im4 = ax4.imshow((data-concentric_psf)[limits[0]:limits[1], limits[0]:limits[1]], cmap=colormap, norm=normalizer)
    #ax3.set_xlim(limits)
    #ax3.set_ylim(limits)
    ax4.set_title('Concentric Circles PSF subtracted')
    plt.colorbar(im4, ax=ax4)


    plt.show()


if __name__ == '__main__':
    main()