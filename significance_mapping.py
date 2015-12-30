import logging
import sys

from photutils import aperture_photometry, CircularAperture
from photutils.geometry import circular_overlap_grid
from photutils.aperture_funcs import get_phot_extents
from astropy.io import fits
import numpy as np
from progressbar import ProgressBar
import pandas as pd


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

grid_cache = {}
def get_circular_overlap(*args, **kwargs):
    global grid_cache

    key = list(args)
    for arg in ['r', 'use_exact', 'subpixels']:
        key.append(kwargs[arg])
    key = tuple(key)
    try:
        retval = grid_cache[key]
    except KeyError:
        retval = circular_overlap_grid(*args, **kwargs)
        #grid_cache[key] = retval
    return retval


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
    kwargs = dict(r=R+ap_radius, use_exact=1, subpixels=5)
    outer = get_circular_overlap(*args, **kwargs)
    if R - ap_radius < 1e-5:
        inner = 0.0
    else:
        kwargs = dict(r=R-ap_radius, use_exact=1, subpixels=5)
        inner = get_circular_overlap(*args, **kwargs)
    annulus = np.zeros(ap_fluxes.shape)
    annulus[y_min[0]:y_max[0], x_min[0]:x_max[0]] = outer - inner
    
    flx = ap_fluxes[annulus > 0] * annulus[annulus > 0]
    avg_flux = np.median(flx)
    flux_std = np.median(np.abs(flx - avg_flux)) * 1.4826

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
    # Make a progress bar
    with ProgressBar(maxval=coords.shape[0]/10.0, redirect_stdout=True) as p:
        for i, c in enumerate(coords):
            sig = get_significance(c, fluxes, ap_radius)
            significance[i] = sig
            if i%10 == 0:
                p.update(i/10)

    return significance.reshape(xx.shape)


def get_annulus_stats(coord, ap_fluxes, ap_radius, center=None, stat_funcs=(np.mean, np.std)):
    """
    Get the significance of a given point by comparing its aperture flux to that
    of other pixels at the same radius from the image center.

    :param coord: The coordinate to get the significance of
    :param ap_fluxes: A 2D array giving the aperture fluxes (such as returned by get_aperture_fluxes)
    :param ap_radius: The aperture radius used to get aperture fluxes.
    :param center: The coordinates of the primary star. If not given,
                   it is assumed to be in the middle of the image
    :param stat_funcs: A tuple of callables that take an array and return some descriptive statistic of the array.
                       The default functions are mean and standard deviation. The functions should all have a unique
                       __name__ attribute.
    """

    if center is None:
        nx, ny = ap_fluxes.shape
        center = (nx / 2., ny / 2.)

    R = np.sqrt((center[0] - coord[0]) ** 2 + (center[1] - coord[1]) ** 2)

    # Find the overlap of each pixel with the an annulus of radius R-ap_radius/2 to R+ap_radius/2
    pos = np.atleast_2d(center)
    extents = np.zeros((len(pos), 4), dtype=int)

    extents[:, 0] = pos[:, 0] - (R + ap_radius / 2.0) + 0.5
    extents[:, 1] = pos[:, 0] + (R + ap_radius / 2.0) + 1.5
    extents[:, 2] = pos[:, 1] - (R + ap_radius / 2.0) + 0.5
    extents[:, 3] = pos[:, 1] + (R + ap_radius / 2.0) + 1.5

    _, extent, phot_extent = get_phot_extents(ap_fluxes, pos, extents)
    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent
    args = [x_pmin[0], x_pmax[0], y_pmin[0], y_pmax[0], x_max[0] - x_min[0], y_max[0] - y_min[0]]

    # Make an array giving the fraction each pixel gives to this radius
    kwargs = dict(r=R + ap_radius, use_exact=1, subpixels=5)
    outer = get_circular_overlap(*args, **kwargs)
    if R - ap_radius < 1e-5:
        inner = 0.0
    else:
        kwargs = dict(r=R - ap_radius, use_exact=1, subpixels=5)
        inner = get_circular_overlap(*args, **kwargs)
    annulus = np.zeros(ap_fluxes.shape)
    annulus[y_min[0]:y_max[0], x_min[0]:x_max[0]] = outer - inner

    flx = ap_fluxes[annulus > 0] * annulus[annulus > 0]
    output = {f.__name__: f(flx) for f in stat_funcs}
    output['radius'] = R

    return pd.Series(data=output)


def get_detection_curve(filename, ap_radius=2, Npix=100, data_hdu=0, center=None):
    """
    Make a pandas DataFrame with the average and standard deviation
    of the aperture fluxes as a function of separation.

    :param filename: A string giving the path to a fits file to analyze
    :param ap_radius: The aperture radius to use. Default = 2 pixels
    :param Npix: The number of pixels to analyze. It can take quite a while to get the
                 significance for every pixel. The default is a 100x100 square around
                 the center, which takes about a minute on my computer.
    :param data_hdu: The header data unit that holds the data.
    :param center: The pixel location of the central star (given as a tuple of size 2).
                   If not given, assumed to be in the middle of the given data.

    :return: pandas DataFrame with the results.
    """
    data = fits.getdata(filename, data_hdu)

    logging.info('Getting aperture fluxes for all pixels in the data')
    fluxes = get_aperture_fluxes(data, ap_radius=ap_radius)

    # Make a coordinate grid
    nx, ny = data.shape
    if center is None:
        x0 = nx / 2
        y0 = ny / 2
    else:
        assert len(center) == 2
        x0, y0 = center
    x = np.arange(x0 - Npix / 2, x0 + Npix / 2)
    y = np.arange(y0 - Npix / 2, y0 + Npix / 2)
    xx, yy = np.meshgrid(x, y)
    coords = np.vstack((xx.flatten(), yy.flatten())).T

    # Get the significance at all coordinates. This is the slow part!
    df_list = []
    # Make a progress bar
    with ProgressBar(maxval=coords.shape[0] / 10.0, redirect_stdout=True) as p:
        for i, c in enumerate(coords):
            df = get_annulus_stats(c, fluxes, ap_radius, center=center)
            df_list.append(df)
            if i % 10 == 0:
                p.update(i / 10)

    df = pd.concat(df_list, axis=1).T.drop_duplicates().sort_values(by='radius')
    return df


def old_main():
    file_list = sys.argv[1:]
    for fname in file_list:
        header = fits.getheader(fname)
        significance = map_significance(fname, Npix=400)
        outfilename = '{}_significance_map.fits'.format(fname[:-5])
        fits.writeto(outfilename, significance, header, clobber=True)
        print('Done with file {}\n\n'.format(fname))


if __name__ == '__main__':
    file_list = sys.argv[1:]
    for fname in file_list:
        print('Starting file {}'.format(fname))
        detection_fluxes = get_detection_curve(fname, Npix=400)
        outfilename = '{}_apfluxes.csv'.format(fname[:-5])
        detection_fluxes.to_csv(outfilename)
        print('Done with file {}\n\n'.format(fname))





