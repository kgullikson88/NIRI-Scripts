# IRAF cl script

#
# Initial processing with the Gemini-provided python scripts
#
!sh run_python_scripts.sh

#
# Reduce the flats, making a flat frame and a bad pixel map
#
!cat FLAT-OPEN.list FLAT-CLOSED.list DARK.list > CAL.list
nprepare @CAL.list

!python classify.py n*.fits
niflat @FLAT-OPEN.list flatf=FLATFILE lampsoff=@FLAT-CLOSED.list darks=@DARK.list

#
# Prepare the object frames, using the new bad pixel map
#
!cat OBJECT*.list > SCI.list
nprepare @SCI.list bpm=FLATFILE_bpm.pl

#
# Propagate saturated pixels to the next frames
#
!cat OBJECT*.list | xargs -I {} echo 'n'{} > SCI.list
nresidual @SCI.list proptime=2

#
# Reduce each star separately
#
!python classify.py b*.fits

iraf.objmasks.convolve = ""
iraf.nisky.ngrow = 20
iraf.nisky.agrow = 5
iraf.nisky.minpix = 10
!echo "circle 512 512 150" > CN.dat
!echo "circle 695 330 150" > LR.dat
!echo "circle 325 330 150" > LL.dat
!echo "circle 325 700 150" > UL.dat
!echo "circle 690 700 150" > UR.dat


#
# Do the rest by hand for each star:
#
nisky  @OBJECT-{star}.list  outim=sky1  fl_keepmasks+

# Residuals still visible, so tweak the masks by hand (loop through the blnN files)

#mskregions CN.dat blnN20150322S0093msk.pl "" append+
#mskregions CN.dat blnN20150322S0094msk.pl "" append+
#mskregions CN.dat blnN20150322S0095msk.pl "" append+
#mskregions CN.dat blnN20150322S0096msk.pl "" append+
#mskregions CN.dat blnN20150322S0097msk.pl "" append+
#mskregions LR.dat blnN20150322S0098msk.pl "" append+
#mskregions LR.dat blnN20150322S0099msk.pl "" append+
#mskregions LR.dat blnN20150322S0100msk.pl "" append+
#mskregions LR.dat blnN20150322S0101msk.pl "" append+
#mskregions LR.dat blnN20150322S0102msk.pl "" append+
#mskregions LL.dat blnN20150322S0103msk.pl "" append+
#mskregions LL.dat blnN20150322S0104msk.pl "" append+
#mskregions LL.dat blnN20150322S0105msk.pl "" append+
#mskregions LL.dat blnN20150322S0106msk.pl "" append+
#mskregions LL.dat blnN20150322S0107msk.pl "" append+
#mskregions UL.dat blnN20150322S0108msk.pl "" append+
#mskregions UL.dat blnN20150322S0109msk.pl "" append+
#mskregions UL.dat blnN20150322S0110msk.pl "" append+
#mskregions UL.dat blnN20150322S0111msk.pl "" append+
#mskregions UL.dat blnN20150322S0112msk.pl "" append+
#mskregions UR.dat blnN20150322S0113msk.pl "" append+
#mskregions UR.dat blnN20150322S0114msk.pl "" append+
#mskregions UR.dat blnN20150322S0115msk.pl "" append+
#mskregions UR.dat blnN20150322S0116msk.pl "" append+
#mskregions UR.dat blnN20150322S0117msk.pl "" append+

# Re-generate a new sky and compare to the first:
#nisky  @OBJECT-{star}.list  outim=sky2  fl_keepmasks+

#nireduce  @OBJECT-{star}.list  skyim=sky2  flatim=flat

!python classify.py r*.fits

#imcoadd @OBJECT-{star}.list outimage = {star}.fits geofitgeom=shift \ 
  rotate- fl_over+ fl_scale+ fl_fixpix+ fl_find+ fl_map+ fl_trn+ fl_med+ \
  fl_add+ fl_avg+ badpix=FLATFILE_bpm.fits niter=1 datamax=100000 fwhm=4
