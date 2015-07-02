from photutils import aperture_photometry, CircularAperture
from photutils.geometry import circular_overlap_grid
from photutils.aperture_funcs import get_phot_extents
from astropy.io import fits
import numpy as np
import HelperFunctions
import logging



def get_aperture_fluxes(data, ap_radius):
    """
    Make an aperture for every pixel, and do aperture photometry at all of them.

    :param data: A 2D numpy array with the data to do photometry on
    :ap_radius: The radius of the aperture to use.
    """

    # Get all coordinates
    nx, ny = data.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    coords = np.vstack((xx.flatten(), yy.flatten())).T

    # Do photometry
    aps = CircularAperture(coords, ap_radius)
    phot = aperture_photometry(data, aps)

    return phot['aperture_sum'].data.reshape(data.shape)


def get_significance(coord, ap_fluxes, ap_radius, center=None):
    """
    Get the significance of a given point by comparing its aperture flux to that
    of other pixels at the same radius from the image center.

    :param coord: The coordinate to get the significance of
    :param ap_fluxes: A 2D array giving the aperture fluxes (such as returned by get_aperture_fluxes)
    :param ap_radius: The aperture radius used to get aperture fluxes.
    :param center: The coordinates of the primary star. If not given, 
                   it is assumed to be in the middle of the image
    """

    if center is None:
        nx, ny = ap_fluxes.shape
        center = (nx/2., ny/2.)
    
    R = np.sqrt((center[0] - coord[0])**2 + (center[1] - coord[1])**2)

    # Find the overlap of each pixel with the an annulus of radius R-ap_radius/2 to R+ap_radius/2
    pos = np.atleast_2d(center)
    extents = np.zeros((len(pos), 4), dtype=int)

    extents[:, 0] = pos[:, 0] - (R+ap_radius/2.0) + 0.5
    extents[:, 1] = pos[:, 0] + (R+ap_radius/2.0) + 1.5
    extents[:, 2] = pos[:, 1] - (R+ap_radius/2.0) + 0.5
    extents[:, 3] = pos[:, 1] + (R+ap_radius/2.0) + 1.5

    _, extent, phot_extent = get_phot_extents(ap_fluxes, pos, extents)
    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent
    args = [x_pmin[0], x_pmax[0], y_pmin[0], y_pmax[0], x_max[0]-x_min[0], y_max[0]-y_min[0]]

    # Make an array giving the fraction each pixel gives to this radius
    outer = circular_overlap_grid(*args, r=R+ap_radius, use_exact=1, subpixels=5)
    if R - ap_radius < 1e-5:
        inner = 0.0
    else:
        inner = circular_overlap_grid(*args, r=R-ap_radius, use_exact=1, subpixels=5)
    annulus = np.zeros(ap_fluxes.shape)
    annulus[y_min[0]:y_max[0], x_min[0]:x_max[0]] = outer - inner

    #avg_flux = np.mean((ap_fluxes*annulus)[annulus > 0])
    #flux_std = np.std((ap_fluxes*annulus)[annulus > 0])
    
    avg_flux = np.median((ap_fluxes*annulus)[annulus > 0])
    flux_std = HelperFunctions.mad((ap_fluxes*annulus)[annulus > 0]) * 1.4826

    return (ap_fluxes[coord[0], coord[1]] - avg_flux) / flux_std


def map_significance(filename, ap_radius=2, Npix=100, data_hdu=0):
    """
    Make a full significance map of the data in 'filename'. 

    :param filename: A string giving the path to a fits file to analyze
    :param ap_radius: The aperture radius to use. Default = 2 pixels
    :param Npix: The number of pixels to analyze. It can take quite a while to get the
                 significance for every pixel. The default is a 100x100 square around
                 the center, which takes about a minute on my computer.
    :param data_hdu: The header data unit that holds the data.
    """
    data = fits.getdata(filename, data_hdu)

    logging.info('Getting aperture fluxes for all pixels in the data')
    fluxes = get_aperture_fluxes(data, ap_radius=ap_radius)

    #Make a coordinate grid
    nx, ny = data.shape
    x = np.arange(nx/2 - Npix/2, nx/2 + Npix/2)
    y = np.arange(ny/2 - Npix/2, ny/2 + Npix/2)
    xx, yy = np.meshgrid(x, y)
    coords = np.vstack((xx.flatten(), yy.flatten())).T

    # Get the significance at all coordinates. This is the slow part!
    significance = np.zeros(coords.shape[0])
    for i, c in enumerate(coords):
        sig = get_significance(c, fluxes, ap_radius)
        significance[i] = sig
        if i % 100 == 0:
            logging.info('Done with pixel {}/{}'.format(i+1, significance.size))

    return significance.reshape(xx.shape)







