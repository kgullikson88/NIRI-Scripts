from astropy.modeling import models, fitting
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.geometry import circular_overlap_grid
from photutils.aperture_funcs import get_phot_extents
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib import cm
from scipy.interpolate import CloughTocher2DInterpolator
import sys
import os


def fit_gaussian_psf(data, x_mean=512, y_mean=512, x_stddev=3, y_stddev=3):
    """
    Fit a gaussian PSF to the data

    data: a 2d numpy array giving the flux at each pixel
    returns: A 2d numpy array with the gaussian model of the data
    """
    YY, XX = np.indices(data.shape)
    #XX, YY = np.indices(data.shape)
    p_init = models.Gaussian2D(amplitude=data[y_mean, x_mean], 
                               x_mean=x_mean, y_mean=y_mean, 
                               x_stddev=x_stddev, y_stddev=y_stddev) + models.Const2D(amplitude=1)
    fitter = fitting.LevMarLSQFitter()
    gauss = fitter(p_init, XX, YY, data)
    
    print(gauss)
    return gauss, gauss(XX, YY), np.array([gauss.x_mean_0.value, gauss.y_mean_0.value])


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
    #XX, YY = np.indices(data.shape)
    p_init = Moffat2D(amplitude=data[y_mean, x_mean], 
                      x_0=x_mean, y_0=y_mean, 
                      x_gamma=6.0, y_gamma=6.0, alpha=1.0) + models.Const2D(amplitude=1)

    fitter = fitting.LevMarLSQFitter()
    moffat = fitter(p_init, XX, YY, data)

    return moffat, moffat(XX, YY), np.array([moffat.x_0_0.value, moffat.y_0_0.value])



def remove_concentric(data, x_mean=512, y_mean=512, x_stddev=3, y_stddev=3, Nsig=10, dp=0.1):
    """
    This function first fits a gaussian PSF to the data to find the centroid.
    Then, it subtracts concentric apertures from the data rather than just 
    subtracting the gaussian model.
    """
    data = data.copy() # Don't overwrite the data
    nx, ny = data.shape
    
    gauss, psf, centroid = fit_gaussian_psf(data, x_mean=x_mean, y_mean=y_mean, 
                                          x_stddev=x_stddev, y_stddev=y_stddev)
    std = (gauss.x_stddev_0.value + gauss.y_stddev_0.value)/2.0
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
    return data - psf, psf, np.array(centroid)


def read_data(fname, rotate=True, data_hdu=1):
    header = fits.getheader(fname)
    sky_const = header['SKYCONST']

    if rotate:
        from skimage.transform import rotate
        angle = header['CRPA']
        #montage_wrapper.wrappers.reproject(fname, 'tmp.fits', north_aligned=True, hdu=1, exact_size=True)
        original_data = fits.getdata(fname, data_hdu)
        maxval, minval = original_data.max(), original_data.min()
        scaled_data = (original_data - minval) / (maxval - minval)
        rotated = rotate(scaled_data, -angle, order=3)
        data = rotated * (maxval - minval) + minval
    else:
        data = fits.getdata(fname, data_hdu)

    sky_const = np.median(data[250:750, 250:750])
    return data[250:750, 250:750] - sky_const


def recenter(data, centroid, Npix=200):
    """
    Given residual (or not) data, interpolate so that it peaks
    in the center of the image of size Npix/Npix
    """
    YY, XX = np.indices(data.shape)
    points = np.vstack((XX.flatten(), YY.flatten())).T
    values = data.flatten()
    fcn = CloughTocher2DInterpolator(points, values)

    new_XX, new_YY = np.meshgrid(np.arange(centroid[0]-Npix, centroid[0]+Npix),
                                 np.arange(centroid[1]-Npix, centroid[1]+Npix))
    new_points = np.vstack((new_XX.flatten(), new_YY.flatten())).T

    return fcn(new_points).reshape(new_XX.shape)


def get_average_residual_pattern(data_list, centroid_list, norm_list, method=np.median):
    """
    Find the average residual pattern of all the data in data_list.
    Recenter the data using centroids in centroid_list, and normalize
    using the peak flux values in norm_list. Method is a function that takes
    a 3D array and converts it into a 2D array with the average/median/whatever values
    at each pixel.
    """
    patterns = []
    centered_data = []
    for data, centroid, normalization in zip(data_list, centroid_list, norm_list):
        centered = recenter(data, centroid)
        patterns.append(centered / normalization)
        centered_data.append(centered)

    avg_pattern = method(patterns, axis=0)
    return avg_pattern, centered_data




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



def main(fname, rotate=True):
    """ Compare the different fitting methods. """
    data = read_data(fname, rotate=rotate)

    # Add a synthetic companion to the data
    #data += make_companion(data, 511, 511, sep=10, delta_mag=4)

    # Get the maximum value
    maxidx = np.argmax(data)
    YY, XX = np.indices(data.shape)
    #XX, YY = np.indices(data.shape)
    x0 = XX.flatten()[maxidx]
    y0 = YY.flatten()[maxidx]

    gauss, gauss_psf, gauss_centroid = fit_gaussian_psf(data, x0, y0)
    moffat, moffat_psf, moffat_centroid = fit_moffat_psf(data, *gauss_centroid)
    concentric, concentric_psf, concentric_centroid = remove_concentric(data.copy(), *gauss_centroid)

    # Plot
    plotsize = 30
    xlimits = (x0-plotsize, x0+plotsize)
    ylimits = (y0-plotsize, y0+plotsize)
    fig1, ax1 = plt.subplots()
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(111)

    normalizer = SymLogNorm(linthresh=1e-3, linscale=0.01)#, clip=True, vmin=-50000, vmax=50000)
    #normalizer = None
    #normalizer = Normalize(vmin=-10000, vmax=10000, clip=True)
    colormap = cm.hot
    im1 = ax1.imshow(data[ylimits[0]:ylimits[1], xlimits[0]:xlimits[1]], cmap=colormap, norm=normalizer)
    #ax1.set_xlim(limits)
    #ax1.set_ylim(limits)
    ax1.set_title('Data')
    plt.colorbar(im1, ax=ax1)

    print(gauss)
    im2 = ax2.imshow((data-gauss_psf)[ylimits[0]:ylimits[1], xlimits[0]:xlimits[1]], cmap=colormap, norm=normalizer)
    ax2.scatter(*(gauss_centroid-np.array((x0-plotsize, y0-plotsize))), color='green', marker='+')
    #ax2.set_xlim(limits)
    #ax2.set_ylim(limits)
    ax2.set_title('Gaussian PSF subtracted')
    plt.colorbar(im2, ax=ax2)

    print(moffat)
    im3 = ax3.imshow((data-moffat_psf)[ylimits[0]:ylimits[1], xlimits[0]:xlimits[1]], cmap=colormap, norm=normalizer)
    ax3.scatter(*(moffat_centroid-np.array((x0-plotsize, y0-plotsize))), color='green', marker='+')
    #ax3.set_xlim(limits)
    #ax3.set_ylim(limits)
    ax3.set_title('Moffat PSF subtracted')
    plt.colorbar(im3, ax=ax3)

    im4 = ax4.imshow((data-concentric_psf)[ylimits[0]:ylimits[1], xlimits[0]:xlimits[1]], cmap=colormap, norm=normalizer)
    ax4.scatter(*(concentric_centroid-np.array((x0-plotsize, y0-plotsize))), color='green', marker='+')
    #ax3.set_xlim(limits)
    #ax3.set_ylim(limits)
    ax4.set_title('Concentric Circles PSF subtracted')
    plt.colorbar(im4, ax=ax4)

    # Save figure
    basename = os.path.split(fname)[-1][:-5]
    fig1.savefig('Figures/{}_Original.pdf'.format(basename))
    fig2.savefig('Figures/{}_GaussianPSF.pdf'.format(basename))
    fig3.savefig('Figures/{}_MoffatPSF.pdf'.format(basename))
    fig4.savefig('Figures/{}_ConcentricPSF.pdf'.format(basename))
    #plt.show()
    plt.close('all')

    # Output fits files with the residuals
    header = fits.getheader(fname)
    fits.writeto('{}_gaussian_psf_norotation.fits'.format(fname[:-5]), data-gauss_psf, header, clobber=True)
    fits.writeto('{}_moffat_psf_norotation.fits'.format(fname[:-5]), data-moffat_psf, header, clobber=True)
    fits.writeto('{}_concentric_psf_norotation.fits'.format(fname[:-5]), data-concentric_psf, header, clobber=True)

    return data-gauss_psf, data-moffat_psf, data-concentric_psf, gauss_centroid, moffat_centroid, concentric_centroid, np.max(data)
    

if __name__ == '__main__':
    gauss_residuals = []
    moffat_residuals = []
    concentric_residuals = []
    gauss_centroids = []
    moffat_centroids = []
    concentric_centroids = []
    peak_fluxes = []
    for fname in sys.argv[1:]:
        print(fname)
        gauss_resid, moffat_resid, conc_resid, gauss_cent, moffat_cent, conc_cent, peak = main(fname, rotate=False)
        print('\n\n')

        gauss_residuals.append(gauss_resid)
        moffat_residuals.append(moffat_resid)
        concentric_residuals.append(conc_resid)
        gauss_centroids.append(gauss_cent)
        moffat_centroids.append(moffat_cent)
        concentric_centroids.append(moffat_cent)
        peak_fluxes.append(peak)

    print('Getting Average residuals for gaussian subtraction')
    avg_pattern, centered_data = get_average_residual_pattern(gauss_residuals, gauss_centroids, peak_fluxes)
    for fname, data, peak in zip(sys.argv[1:], centered_data, peak_fluxes):
        header = fits.getheader(fname)
        fits.writeto('{}_gaussian_psf_despeckled.fits'.format(fname[:-5]), data - avg_pattern*peak, header, clobber=True)


    print('Getting Average residuals for moffat subtraction')
    avg_pattern, centered_data = get_average_residual_pattern(moffat_residuals, moffat_centroids, peak_fluxes)
    for fname, data, peak in zip(sys.argv[1:], centered_data, peak_fluxes):
        fits.writeto('{}_moffat_psf_despeckled.fits'.format(fname[:-5]), data- avg_pattern*peak, header, clobber=True)
    
    print('Getting Average residuals for concentric subtraction')
    avg_pattern, centered_data = get_average_residual_pattern(concentric_residuals, concentric_centroids, peak_fluxes)
    for fname, data, peak in zip(sys.argv[1:], centered_data, peak_fluxes):
        fits.writeto('{}_concentric_psf_despeckled.fits'.format(fname[:-5]), data- avg_pattern*peak, header, clobber=True)

    




