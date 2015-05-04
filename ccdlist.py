from astropy.io import fits
import sys

if __name__ == '__main__':
    file_list = sys.argv[1:]

    for fname in file_list:
        hdr = fits.getheader(fname)
        if 'gcalshut' in hdr:
            print(fname, hdr['OBJECT'], hdr['OBSTYPE'], hdr['gcalshut'])
        else:
            print(fname, hdr['OBJECT'], hdr['OBSTYPE'], hdr['EXPTIME'])
