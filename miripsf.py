"""There may still be something subtly wrong with the shifting (?)
"""

from my_utils import cdine

import os
import shutil
import glob

import numpy as np
from scipy.ndimage import shift, zoom

from skimage.transform import downscale_local_mean
from photutils.centroids import centroid_2dg, centroid_sources

from astropy.wcs import WCS
from astropy.io import fits

from .pipeline_steven_edit_lab import run_Image3Pipeline

from jwst import datamodels


def get_psf_model(which='jens-v6', y=None, verbose=True):

    if which == 'jens-v5':
        with fits.open (f'{DATA_PATH}/PSF-hudf/Jens-Stacks/v5-29mar23/MIRI_LMC_55mas_emp_psf.fits') as hdu:
            psf = hdu[0].data

    elif which == 'jens-v6':
        with fits.open (f'{DATA_PATH}/PSF_hudf/Jens-Stacks/v6-12jun23/MIRI_LMC_55mas_emp_final_psf.fits') as hdu:
            # y coordinate dependent psf
            if 0 <= y <= 341:
                ext = 1
            elif 341 < y <= 682:
                ext = 2
            elif 682 < y < 1024:
                ext = 3
            else:
                print(f'y={y} out of bounds, should be skipped')
                return 'bad'

            if verbose:
                print(f"loaded ext={ext} for y={y}")
            psf = hdu[ext].data

    return psf


class MIRIPSF():

    def __init__(self,
                 ra_list,
                 dec_list,
                 img2dir,
                 img3dir,
                 procdir,
                 psf_which='jens-v6',
                 run_clean=True,
                 master_filename='HUDF_F560W_i2d.fits',
                 pixscale=0.06, pixfrac=1):

        # we don't want astropy coordinates here,
        # and we allow for a list of coords
        try:
            self.ra_list = np.atleast_1d(ra_list.value)
        except:
            self.ra_list = np.atleast_1d(ra_list)
        try:
            self.dec_list = np.atleast_1d(dec_list.value)
        except:
            self.dec_list = np.atleast_1d(dec_list)

        self.img2dir = img2dir
        self.img3dir = img3dir
        self.procdir = procdir

        self.psf_which = psf_which

        self.run_clean = run_clean
        self.master_filename = master_filename

        self.pixscale = pixscale
        self.pixfrac = pixfrac

        print(self.ra_list, self.dec_list, self.img2dir, self.img3dir, self.procdir)

        # hardcoded setup
        INPUT_DIRECTORY = f'{DATA_PATH}/PSF-hudf/Copenhagen/cal_files/'
        self.INPUT_FILES = []
        for i in range(96):
            self.INPUT_FILES.append(os.path.join(INPUT_DIRECTORY, f'HUDF_F560W_exp{i:02d}_cal.fits'))

        # we created the clean file once by running only self._prep_img2dir() and self.run_pipeline()
        if self.run_clean == True:
            self.img2dir_clean = self.img2dir.rstrip('/')+'-clean/'
            self.img3dir_clean = self.img3dir.rstrip('/')+'-clean/'
            self.CLEAN_FILE = os.path.join(self.procdir, self.master_filename.replace('.fits', '-clean.fits'))
            print(self.img2dir_clean, self.img3dir_clean)
        else:
            self.run_clean = False
            self.CLEAN_FILE = run_clean #f'{DATA_PATH}/PSF-hudf/Insertion/1.8.4_automatic_runs/v1/IMG3-clean/HUDF_F560W_i2d.fits'

        # individual exposure filenames to process - these only exist after running self._prep_img2dir
        self.files = []
        for i in range(96):
            self.files.append(os.path.join(img2dir, f'HUDF_F560W_exp{i:02d}_cal.fits'))

        # filenames for inserted and recovered files
        self.psf_inserted = os.path.join(self.procdir, self.master_filename)
        self.psf_recovered = os.path.join(self.procdir, self.master_filename).replace('.fits', '-psf_inserted-original.fits')


    def run(self, cleanup=True):
        if self.run_clean:
            # insertion
            self._prep_img2dir(clean=True)
            # run pipeline without psf inserted
            self._run_pipeline(clean=True)
            # run clean file to procdir
            self._copy_pipeline_output(clean=True)

        # insert at pixel center coordinates
        self._update_coords_to_pixel_centers_in_clean()

        # insertion
        self._prep_img2dir()

        # insert psf in images
        self._insert_psf_in_images()

        # run pipeline with psf inserted
        self._run_pipeline()

        # copy inserted output to procdir
        self._copy_pipeline_output()

        # subtract clean from psf-inserted
        self._subtract_original()

        # cutout and normalise inserted psf
        self._cutout_psf(pixel_scale_ratio=0.055/self.pixscale)  # scale of the input psf image

        if cleanup:
            # cleanup img2dir and img3dir
            self._cleanup()


    ### INSERTION
    def _prep_img2dir(self, clean=False, drop_unnecessary_exts=False, slice_only=slice(None)):
        """First copy the files to the img2dir where we run the pipeline."""
        print('prep_img2dir')
        if clean:
            img2dir = self.img2dir_clean
        else:
            img2dir = self.img2dir

        cdine(img2dir)

        for f in self.INPUT_FILES[slice_only]:
            shutil.copy2(f, img2dir)

        if drop_unnecessary_exts:
            self._drop_unnecessary_exts(clean=clean)


    def _insert_psf_in_images(self, slice_only=slice(None), **kwargs):
        print('insert_psf_in_images')
        for f in self.files[slice_only]:
            self._insert_psf_in_image(f, self.psf_which, **kwargs)


    def _insert_psf_in_image(self,
                             fname,
                             psf_which,
                             ra_list=None,
                             dec_list=None,
                             name='',
                             write_psfonly=False,
                             oversample_factor=2,    #1,  is two for Jens' new 55mas PSF
                             boost_factor=1.,
                             debug=False):
        """Insert PSF into file (inplace).
        """
        if ra_list is None:
            ra_list = self.ra_list
        if dec_list is None:
            dec_list = self.dec_list

        assert isinstance(oversample_factor, int), "oversample_factor must be integer"

        # load cal file to insert
        cal_gwcs = datamodels.open(fname)
        wcs = cal_gwcs.meta.wcs
        with fits.open(fname) as hdu:
            cal_h = hdu[1].header
            cal = hdu[1].data

            if write_psfonly:
                hdu[1].data = np.zeros_like(hdu[1].data)

            for ra, dec in zip(ra_list, dec_list):

                # find pixel coord of ra, dec and determine integer and fractional part
                # wcs = WCS(cal_h)
                #x, y = wcs.all_world2pix(ra, dec, 0, tolerance=1e-6)
                x, y = wcs.backward_transform(ra, dec)
                print(ra, dec, x, y)

                # load input psf
                psf = get_psf_model(psf_which, y=y)
                if isinstance(psf, str):
                    print('coord not in psf, skipping')
                    continue

                xf, xi = np.modf(x)
                xi = int(xi)
                yf, yi = np.modf(y)
                yi = int(yi)

                print('pix pos (int, frac)-part of ra, dec', (xi, xf), (yi, yf))

                if oversample_factor > 1:
                    # the psf is oversampled by a factor, the way we treat this is by
                    # oversampling the image to input in, adding the psf, and then downsampling again
                    # this means that we need to increase the (x,y) pixel coords by the same factor
                    x *= oversample_factor
                    y *= oversample_factor
                    # for even oversampling the pixel center then shifts by 0.5 (i.e. center is now corner, etc.)
                    # I don't understand why this is a + and not - (?) but that gives correct results.
                    # it's because of the 0 indexing
                    x += 0.5 * ((oversample_factor + 1)%2)
                    y += 0.5 * ((oversample_factor + 1)%2)

                xf, xi = np.modf(x)
                xi = int(xi)
                yf, yi = np.modf(y)
                yi = int(yi)

                print('pix pos (int, frac)-part of ra, dec', (xi, xf), (yi, yf), 'after oversampling')

                if (yi < psf.shape[0]//2 or
                    xi < psf.shape[1]//2 or
                    yi > cal.shape[0]*oversample_factor - psf.shape[0]//2 - 1 or
                    xi > cal.shape[1]*oversample_factor - psf.shape[1]//2 - 1):
                    print('psf too close to edge, skipping')
                    continue

                # insert the psf into an empty array like the image
                # first at the integer coordinates (exact)
                # then shift whole array by remaining fraction
                tmpl = np.zeros(np.array(cal.shape) * oversample_factor)

                # odd array, so do range-up + 1
                tmpl[yi - psf.shape[0]//2: yi + psf.shape[0]//2 + 1,
                     xi - psf.shape[1]//2: xi + psf.shape[1]//2 + 1] = psf

                if debug:
                    tmp = fits.PrimaryHDU(tmpl)
                    tmp.writeto(fname.replace('.fits', f'{name}-debug-insert.fits'), overwrite=True)

                # this is now correct
                tmpl_shift = shift(tmpl, (yf, xf), order=3, mode='constant', cval=0, prefilter=True)

                if debug:
                    tmp = fits.PrimaryHDU(tmpl_shift)
                    tmp.writeto(fname.replace('.fits', f'{name}-debug-insert-shift.fits'), overwrite=True)

                if oversample_factor > 1:
                    # if we oversample, we now downsample
                    ### NOTE zoom may actually be somewhat inaccurate - may need to replace with alternative?
                    #tmpl_shift = zoom(tmpl_shift, 1./oversample_factor, mode='constant', cval=0, order=3)  # order 1 is fine here, for a factor 2 downsample
                    # the right thing to do is of course just to average 2x2, which we can accomplish with downscale_local_mean
                    tmpl_shift = downscale_local_mean(tmpl_shift, oversample_factor, cval=0)

                # make PSF brighter
                tmpl_shift *= boost_factor

                if debug:
                    tmp = fits.PrimaryHDU(tmpl_shift)
                    tmp.writeto(fname.replace('.fits', f'{name}-debug-insert-shift-downsample.fits'), overwrite=True)

                # add tmpl to data
                hdu[1].data += tmpl_shift

            if write_psfonly:
                hdu.writeto(fname.replace('.fits', f'{name}-psfonly.fits'), overwrite=True)
            else:
                hdu.writeto(fname.replace('.fits', f'{name}.fits'), overwrite=True)


    ### Run pipeline
    def _run_pipeline(self, clean=False):
        print('run_pipeline')
        cwd = os.getcwd()
        print(os.getcwd())

        if clean:
            img2dir = self.img2dir_clean
            img3dir = self.img3dir_clean
        else:
            img2dir = self.img2dir
            img3dir = self.img3dir

        cdine(img3dir)
        os.chdir(img3dir)
        print(os.getcwd())

        try:
            # run pipeline in cwd
            run_Image3Pipeline(img2dir, '', self.pixscale, self.pixfrac)
        except Exception as e:
            print(e)
            print("There was an error running the jwst pipeline")

        os.chdir(cwd)


    ### POST-PROCESSING
    def _copy_pipeline_output(self, clean=False) :#filename=None):
        print('copy_pipeline_output')
        cdine(self.procdir)

        if clean:
            filename = os.path.join(self.img3dir_clean, self.master_filename)
        else:
            filename = os.path.join(self.img3dir, self.master_filename)

        # if filename is None:
        #     filename = os.path.join(self.img3dir, master_filename)

        shutil.copy2(filename , self.procdir)

        if clean:
            shutil.move(filename, os.path.join(self.img3dir_clean, self.CLEAN_FILE))


    def _subtract_original(self, psf_inserted=None):
        print('subtract_original')
        if psf_inserted is None:
            psf_inserted = self.psf_inserted

        with fits.open(self.CLEAN_FILE) as hdu:
            orig = hdu[1].data

        # load cal file to insert
        with fits.open(psf_inserted) as hdu:
            hdu[1].data -= orig

            hdu.writeto(self.psf_recovered, overwrite=True)

    def _cutout_psf(self,
                    psf_recovered=None,
                    ra_list=None,
                    dec_list=None,
                    cutout_size=None,
                    original_cutout_size=(299,299),     # seems to work well for Jens' psf at 60 mas
                    pixel_scale_ratio=1.
                    ):
        print('cutout_psf')
        if psf_recovered is None:
            psf_recovered = self.psf_recovered

        if ra_list is None:
            ra_list = self.ra_list
        if dec_list is None:
            dec_list = self.dec_list

        cal_gwcs = datamodels.open(psf_recovered)
        wcs = cal_gwcs.meta.wcs
        with fits.open(psf_recovered) as hdu:
            cal_h = hdu[1].header
            cal = hdu[1].data

        # extract for each coordinate and save numbered starting at 1
        for coordindex, (ra, dec) in enumerate(zip(ra_list, dec_list)):
            coordindex += 1

            # find pixel coord of ra, dec and determine integer and fractional part
            #wcs = WCS(cal_h)
            # x, y = wcs.all_world2pix(ra, dec, 0, tolerance=1e-6)
            x, y = wcs.backward_transform(ra, dec)
            # the positions should be integers here by construction,
            # but we round because the wcs can give (x-1).99999 values
            xi = int(np.round(x,0))
            yi = int(np.round(y,0))
            assert abs(xi - x) < 0.01 and (yi - y) < 0.01, "psfs are not at pixel centers to 0.01 level {} {}".format(x,y)

            round_odd = lambda x: np.array(np.ceil(x) * 2 + 1, dtype=int)  # round to odd integer

            if cutout_size is None:
                cutout_size = round_odd(np.array(original_cutout_size) * pixel_scale_ratio / 2)

            # cutout psf - the size of original
            out = cal[yi - cutout_size[0]//2: yi + cutout_size[0]//2 + 1,
                      xi - cutout_size[1]//2: xi + cutout_size[1]//2 + 1]

            hdu_out = fits.PrimaryHDU(out)

            print(xi, yi, cutout_size[0]//2 + 1, cutout_size[1]//2 + 1, ra, dec)

            hdu_out.header['PSF_X'] = cutout_size[0]//2 + 1
            hdu_out.header['PSF_Y'] = cutout_size[0]//2 + 1

            hdu_out.header['PSF_RA'] = ra
            hdu_out.header['PSF_DEC'] = dec

            hdu_out.header['PIXAR_A2'] = cal_h['PIXAR_A2']

            hdu_out.writeto(psf_recovered.replace('.fits', f'{coordindex}-cutout.fits'), overwrite=True)

            # normalise - bg
            # background subtract from corners
            ii = 50
            bg = np.mean(np.concatenate([out[:ii, :ii], out[:ii, -ii:], out[-ii:, :ii], out[-ii:, -ii:]]))
            out -= bg
            hdu_out.data = out
            out_bg = np.mean(np.concatenate([out[:ii, :ii], out[:ii, -ii:], out[-ii:, :ii], out[-ii:, -ii:]]))
            print('background', bg, 'after subtr', out_bg)

            hdu_out.writeto(psf_recovered.replace('.fits', f'{coordindex}-cutout-bg.fits'), overwrite=True)

            sumout = np.sum(out)
            out /= sumout
            hdu_out.data = out
            print('norm', sumout, 'after norm', np.sum(out))
            hdu_out.writeto(psf_recovered.replace('.fits', f'{coordindex}-cutout-bg-norm.fits'), overwrite=True)

            # add extra step to refine the centroid
            xcenter, ycenter = cutout_size[0]//2, cutout_size[1]//2
            xcentroid, ycentroid = centroid_sources(out, ycenter, xcenter,
                                                     box_size=round_odd(cutout_size / 5), centroid_func=centroid_2dg)
            hdu_out.data = shift(out, (ycenter-ycentroid, xcenter-xcentroid), order=3, mode='constant', cval=0, prefilter=True)
            print('center', xcenter, ycenter, 'centroid', xcentroid, ycentroid)
            hdu_out.writeto(psf_recovered.replace('.fits', f'{coordindex}-cutout-bg-norm-centered.fits'), overwrite=True)


    def _cleanup(self):
        if self.run_clean:
            shutil.rmtree(self.img2dir_clean)
            shutil.rmtree(self.img3dir_clean)
        shutil.rmtree(self.img2dir)
        shutil.rmtree(self.img3dir)


    def _update_coords_to_pixel_centers_in_clean(self):

        print('updating coordinates to pixel centers in clean image')

        cal_gwcs = datamodels.open(self.CLEAN_FILE)
        wcs_clean = cal_gwcs.meta.wcs
        # with fits.open(self.CLEAN_FILE) as hdu:
        #     wcs_clean = WCS(hdu['SCI'].header)

        ra_center, dec_center = [], []
        for ra, dec in zip(self.ra_list, self.dec_list):

            #x, y = wcs_clean.all_world2pix(ra, dec, 0, tolerance=1e-6)
            x, y = wcs_clean.backward_transform(ra, dec)
            xi, yi = int(np.round(x,0)), int(np.round(y,0))

            #ra_cent, dec_cent = wcs_clean.all_pix2world(xi, yi, 0)
            ra_cent, dec_cent = wcs_clean(xi, yi)

            print('original', ra, dec, x, y)
            print('updated', ra_cent, dec_cent, xi, yi)

            ra_center.append(ra_cent)
            dec_center.append(dec_cent)

        # each value is an array, convert to list
        self.ra_list = list(np.array(ra_center))
        self.dec_list = list(np.array(dec_center))


    def _drop_unnecessary_exts(self, clean=False):

        # this doesn't work, the pipeline won't run w/o them


        # ALTERNATIVE FROM JAMES (NOT TESTED YET):
        # from jwst import datamodels


        # model = datamodels.open("jw001234_blah_blah_cal.fits")
        # del model.var_poisson
        # del model.var_flat
        # del model.var_rnoise
        # model.save("test.fits")

        print("drop_unnecessary_exts")

        EXTS = ['ERR', 'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT']

        for f in glob.glob(os.path.join((self.img2dir_clean if clean else self.img2dir), '*.fits')):
            with fits.open(f) as hdu:
                for EXT in EXTS:
                    del hdu[EXT]

                hdu.writeto(f, overwrite=True)
