#!/usr/bin/env python

# 2007 Aug 27 - Andrew W Stephens - alpha version
# 2007 Aug 28 - AWS - beta version
# 2007 Sep 20 - AWS - new coefficients based on spectroscopic flats
# 2008 Feb 12 - AWS - handle coadds & nprepared data
# 2008 Feb 12 - AWS - default output naming to match Gemini IRAF convention
# 2008 Feb 12 - AWS - add .fits extension if not given
# 2008 Feb 14 - AWS - new coefficients based on average count rate
# 2008 Apr 15 - AWS - new coefficients based on new model
# 2008 Apr 25 - AWS - include y-position coefficients
# 2008 Oct 21 - AWS - update coefficients derived with low-flux data
# 2008 Oct 23 - AWS - include high-flux (phot) and full-range flux (spec) coefficients
# 2008 Oct 24 - AWS - handle NaN, Inf, and negative corrections
# 2008 Oct 29 - AWS - settle on three best sets of shallow-well coefficients
# 2008 Nov 03 - AWS - set minimum counts for correction = 10 ADU
# 2008 Dec 07 - AWS - add HRN Deep-well coefficients and max count limits
# 2009 Jan 05 - AWS - set uncorrectable pixels to BADVAL
# 2009 Jan 09 - AWS - add option to force correction outside recommended range
# 2009 Jan 13 - AWS - check that this script has not already been run
# 2009 May 22 - AWS - verify FITS header to catch old images with unquoted release date string
# 2010 Jun 18 - Nicholas T Lange - new coefficients for subarrays based on average count rate
# 2010 Jul 19 - NTL - modified coefficients for MRN-1024-Shallow, now correct down to 1 count
# 2010 Aug 24 - NTL - coefficients from SVD fit are now included
# 2010 Sep 26 - AWS - include med-RN 256 shallow-well
# 2010 Nov 22 - AWS - include high-RN 1024 deep and shallow-well; multiply by coadds at end
# 2013 Apr 11 - AWS - multiply exposure time of nprepared images by the number of coadds
# 2013 Jun 24 - AWS - convert history list to a string before searching it

#-----------------------------------------------------------------------

import datetime
import getopt
import glob
import numpy
import os
import pyfits
import sys

version = '2013 Jun 24'

#-----------------------------------------------------------------------

def usage():
    print ''
    print 'NAME'
    print '       nirlin.py - NIR linearization\n'
    print 'SYNOPSIS'
    print '       nirlin.py [options] infile\n'
    print 'DESCRIPTION'
    print '       Run on raw or nprepared Gemini NIRI data, this'
    print '       script calculates and applies a per-pixel linearity'
    print '       correction based on the counts in the pixel, the'
    print '       exposure time, the read mode, the bias level and the'
    print '       ROI.  Pixels over the maximum correctable value are'
    print '       set to BADVAL unless given the force flag.'
    print '       Note that you may use glob expansion in infile,'
    print '       however, any pattern matching characters (*,?)'
    print '       must be either quoted or escaped with a backslash.'
    print ' '
    print 'OPTIONS'
    print '       -b <badval> : value to assign to uncorrectable pixels [0]'
    print '       -f : force correction on all pixels'
    print '       -o <file> : write output to <file> [l<inputfile>]'
    print '            If no .fits is included this is assumed to be a directory'
    print '       -v : verbose debugging output\n'
    print 'VERSION'
    print '       ', version
    print ''
    raise SystemExit

#-----------------------------------------------------------------------

def main():

    try:
        opts,_ = getopt.getopt(sys.argv[1:], 'b:fo:v')
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    
    nargs = len(sys.argv[1:])
    nopts = len(opts)
    
    badval = 0
    force = False
    outputfile = 'default'
    verbose = False
    
    for o, a in opts:
        if o in ('-b'):      # value for bad pixels (over correction limit)
            badval = a
            nopts += 1
        if o in ('-f'):      # force correction on all pixels, even if over limit
            force = True
        if o in ('-o'):      # linearized output file
            outputfile = a
            nopts += 1
        if o in ('-v'):      # verbose debugging output
            verbose = True
    
    if (verbose):
        print "...nargs = ", nargs
        print "...nopts = ", nopts
    
    if (nargs - nopts) != 1:
        usage()
    
    inputfile = sys.argv[nopts+1]

    files = glob.glob(inputfile)
    if (verbose):
        print '...files = ', files

    for f in files:
        nirlin(f,badval=badval,force=force,outputfile=outputfile,verbose=verbose)

#-----------------------------------------------------------------------

def getCoefficients(naxis2, welldepth, readmode, force=False):

    if   readmode == 'low-noise' and naxis2 == 1024 and welldepth == 'shallow':
        maxcounts = 12000
        dt = 1.2662732
        g = 7.3877618e-06
        e = 1.940645271e-10
        params = (dt,g,e)

    elif readmode == 'medium-noise' and naxis2 == 1024 and welldepth == 'shallow':
        maxcounts = 12000
        dt = 0.09442515154
        g = 3.428783846e-06
        e = 4.808353308e-10
        params = (dt,g,e)

    elif readmode == 'medium-noise' and naxis2 == 256 and welldepth == 'shallow':
        maxcounts = 12000
        dt = 0.01029262589
        g = 6.815415667e-06
        e = 2.125210479e-10
        params = (dt,g,e)

    elif readmode == 'high-noise' and naxis2 == 1024 and welldepth == 'shallow':
        maxcounts = 12000
        dt = 0.009697324059
        g = 3.040036696e-06
        e = 4.640788333e-10
        params = (dt,g,e)

    elif readmode == 'high-noise' and naxis2 == 1024 and welldepth == 'deep':
        maxcounts = 21000
        dt = 0.007680816203
        g = 3.581914163e-06
        e = 1.820403678e-10
        params = (dt,g,e)

    else:
        print 'ERROR: coefficients do not exist for this mode.'
        print 'Please contact Gemini Observatory for more information.'
        sys.exit(2)
 
    if force:
        maxcounts = 65000
        print '...forcing linearity correction on all pixels'
    else:
        print '...upper limit for linearization =', maxcounts, 'ADU/coadd'

    return maxcounts,params

#-----------------------------------------------------------------------

def getSVDCorrection(counts, gamma, eta):
    return counts + gamma*counts**2 + eta*counts**3

#-----------------------------------------------------------------------

def nirlin(inputfile, badval=0, force=False, outputfile='default', verbose=False):

    print 'NIRLIN v.', version

    # Check file names
    if not inputfile.endswith('.fits'):
        inputfile = inputfile + '.fits'

    if outputfile == 'default':
        #outputfile = os.path.join(os.path.dirname(inputfile), 'l' + os.path.basename(inputfile))
        outputfile = 'l' + os.path.basename(inputfile)
    else:
        if ( not outputfile.endswith('.fits') ):
            # outputfile = outputfile + '.fits'
            if not os.path.isdir(outputfile):
                os.mkdir(outputfile)
            outputfile = outputfile + '/l' + os.path.basename(inputfile)

    if verbose:
        print '...output = ', outputfile

    # Error checking:
    if not os.path.exists(inputfile):      # check whether input file exists
        print inputfile, 'does not exist'
        sys.exit(2)
    if os.path.exists(outputfile):        # check whether output file exists
        print '...removing old', outputfile
        os.remove(outputfile)

    print '...reading', inputfile
    hdulist = pyfits.open(inputfile)

    print '...verifying...'
    if verbose:
        hdulist.verify('fix')
    else:
        hdulist.verify('silentfix')

    # Check if this image has already been linearized:
    try:
        history = hdulist[0].header['HISTORY']
    except:
        history = ''
    if verbose:
        print '...history = ', history
    if str(history).count("Linearized by nirlin.py") > 0:
        print "ERROR: ", history
        sys.exit(2)

    # Get the number of extensions in the image:
    next = len(hdulist)
    if verbose:
        print '...number of extensions =', next
    if next == 1:
        sci = 0
    else:
        sci = 1
    if verbose:
        print '...assuming science data are in extension', sci

    # Get the image dimensions:
    try:
        naxis1,naxis2 = hdulist[sci].header['NAXIS1'],hdulist[sci].header['NAXIS2']
        print '...image dimensions =', naxis1, 'x', naxis2
    except:
        print 'ERROR: cannot get the dimensions of extension ', sci
        pyfits.info(inputfile)
        sys.exit(2)
    
    exptime = hdulist[0].header['EXPTIME']
    print '...input exposure time =', exptime, 's'

    # Check that exposure time is in range:
    if exptime > 600:
        print 'WARNING: exposure time is outside the range used to derive correction.'

    # Read science data:
    counts = hdulist[sci].data
    if verbose:
        print 'INPUT DATA:'
        print counts
    print '...mean of input image =', numpy.mean(counts)

    # Convert to counts / coadd:
    coadds = hdulist[0].header['COADDS']
    print '...number of coadds =', coadds
    if coadds > 1:
        print '...converting to counts / coadd...'
        counts = counts / coadds

    # Nprepare modifies the exposure time keyword value to be exptime * ncoadds
    # so if nprepared, undo this operation to get the original exposure time:
    nprepared = False
    try:
        hdulist[0].header['PREPARE']
        print '...image has been nprepared'
        nprepared = True
    except:
        if verbose:
            print '...image has not been nprepared (which is okay)'

    if nprepared and coadds > 1:
        print '...converting to exptime / coadd...'
        exptime = exptime / coadds
        print '...exptime = ', exptime

    # Read mode:
    lnrs = hdulist[0].header['LNRS']
    print '...number of low noise reads =', lnrs 
    ndavgs = hdulist[0].header['NDAVGS']
    print '...number of digital averages =', ndavgs
    if   ( lnrs == 1  and ndavgs == 1  ):
        readmode = 'high-noise'
    elif ( lnrs == 1  and ndavgs == 16 ):
        readmode = 'medium-noise'
    elif ( lnrs == 16 and ndavgs == 16 ):
        readmode = 'low-noise'
    else:
        print 'ERROR: Unknown read mode'
        sys.exit(2)
    print '...read mode =', readmode

    # Bias level / Well-depth:
    vdduc = hdulist[0].header['A_VDDUC']
    vdet  = hdulist[0].header['A_VDET']
    biasvoltage = vdduc - vdet
    print '...bias voltage =', biasvoltage, 'V'
    shallowbias = -0.60   # shallow-well detector bias (V)
    deepbias    = -0.87   # deep-well detector bias (V)
    if abs(biasvoltage - shallowbias) < 0.05:
        welldepth = 'shallow'
    elif abs(biasvoltage - deepbias) < 0.05:
        welldepth = 'deep'
    else:
        print '...ERROR: can not determine well depth.'
        sys.exit(2)
    print '...well depth =', welldepth

    maxcounts, coefficients = getCoefficients(naxis2, welldepth, readmode, force=force)
    if verbose:
        print 'COEFFICIENTS:', coefficients

    # Create array of pixel y-positions:
    y1 = numpy.arange(naxis2/2,0,-1).repeat(naxis2).reshape(naxis2/2,naxis2)
    y2 = numpy.arange(1,naxis2/2+1).repeat(naxis2).reshape(naxis2/2,naxis2)
    ypos = numpy.concatenate((y1,y2))
    if verbose:
        print 'YPOS:'
        print ypos

    # Calculate correction:
    newcounts = getSVDCorrection(counts, coefficients[1], coefficients[2])

    # Set out-of-range pixels to BADVAL:
    if not force:
        print '...setting out-of-range pixels to', badval
        newcounts[counts>maxcounts] = badval

    # Return total counts (instead of counts / coadd):
    if coadds > 1:
        print '...converting back to total counts...'
        newcounts = newcounts * coadds

    # Write FITS data:
    hdulist[sci].data = newcounts

    if verbose:
        print 'OUTPUT DATA:'
        print hdulist[sci].data

    print '...mean of output image =', hdulist[sci].data.mean()

    print '...updating header...'
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    if (verbose):
        print '...time stamp =', timestamp
    hdulist[0].header.add_history('Linearized by nirlin.py ' + timestamp)

    if verbose:
        print '...correcting exposure time by adding %.3fs / coadd' % coefficients[0]
    exptime += coefficients[0]

    if nprepared and coadds > 1:
        print '...converting back to total exptime...'
        exptime = exptime * coadds

    print '...new exposure time = %.3f s' % exptime
    hdulist[0].header.update('EXPTIME',exptime)

    print '...writing', outputfile
    hdulist.writeto(outputfile)
    hdulist.close()
    
#-----------------------------------------------------------------------
    
if __name__ == "__main__":
    main()
