AO Image Reduction and Analysis Scripts
-----------------------------------------

Use these scripts to reduce data. Use the following steps:

  1. python classify.py N*.fits --skip-first
  2. cat FLAT*.list DARK.list > CALS.list
  3. cat OBJECT*.list > SCI.list
  4. Process calibration frames:

    ```
    nprepare @CALS.list
    niflat flatfile=FLATFILE lampson=n//@FLAT-OPEN.list lampsoff=n//@FLAT-CLOSED.list darks=n//@DARKS.list thresh_dlo=20 thresh_dup=5000
    ```
  5. nprepare @SCI.list bpm=FLATFILE_bpm.pl
  6. Do the non-linearity correction:

     ```bash
     for f in `cat SCI.list`
     do
       python nirlin.py n$f
     done
     ```
  7. Process each object separately from here on out
     - nresidual ln//@OBJECT_STARNAME.list proptime=5
     - nisky bln//@OBJECT_STARNAME.list outim=sky1 fl_keepmasks+
     - Check for residuals of the star dither pattern in the sky frame (sky1.fits). If they exist (they probably will), manually add to the mask with something like:

       ```
       echo "circle 512 512 150" > CN.dat
       mskregions CN.dat blnN20150322S0093msk.pl "" append+
       ```
     - nisky bln//@OBJECT_STARNAME.list outim=sky2 fl_keepmasks+
     - nireduce bln//@OBJECT_STARNAME.list skyim=sky2 flatim=FLATFILE
     - imcoadd rbln//@OBJECT_STARNAME.list outim=STARNAME fwhm=4 geofit=shift datamax=100000
