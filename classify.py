from astropy.io import fits
import sys
from collections import defaultdict


def classify_file(fname):
    hdr = fits.getheader(fname)
    obstype = hdr['OBSTYPE']
    if obstype.lower() == 'object':
        return obstype, hdr['OBJECT'].replace(' ', '_')
    elif obstype.lower() == 'flat':
        return obstype, hdr['gcalshut']
    elif obstype.lower() == 'dark':
        return obstype, obstype


def make_lists(file_list, skip=False):
    

    # classify each file
    file_dict = defaultdict(list)
    for fname in file_list:
        maintype, subtype = classify_file(fname)
        if maintype == subtype:
            file_dict['{}.list'.format(maintype)].append(fname)
        else:
            file_dict['{}-{}.list'.format(maintype, subtype)].append(fname)

    # make lists for each type. Cut the first one from each sequence
    for key in file_dict:
        print('Making file for {}'.format(key))
        if skip:
            print('Skipping file {}'.format(file_dict[key][0]))
            file_dict[key] = sorted(file_dict[key][1:])
        outfile = open(key, 'w')
        for fname in file_dict[key]:
            outfile.write('{}\n'.format(fname))
        outfile.close()
    return file_dict


if __name__ == '__main__':
    file_list = []
    skip = False
    #file_list = sys.argv[1:]
    for arg in sys.argv[1:]:
        if 'skip-first' in arg:
            skip=True
        else:
            file_list.append(arg)

    make_lists(file_list, skip=skip)
