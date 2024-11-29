"""Script to run the PSF insertion and recovery.
"""

import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as U

from importlib import reload
import miripsf
reload(miripsf)
import miripsf.miripsf as M
reload(M)

# overall relative path - check code below for some hard-coded links inside this
DATA_PATH="../PSF_hudf/Insertion/"


########## clean run (to subtract off inserted run) - only need to run this once and then refer to it below
img2dir = f'{DATA_PATH}/1.8.4_automatic_runs/v1/IMG2-clean/'
img3dir = f'{DATA_PATH}/1.8.4_automatic_runs/v1/IMG3-clean/'
procdir = f'{DATA_PATH}/1.8.4_automatic_runs/v1/clean/'
# note ra, dec are irrelevant here and we skip the insertion and post-processing steps
m_clean = M.MIRIPSF([None], [None], img2dir, img3dir, procdir)
m_clean._prep_img2dir()
m_clean._run_pipeline()


########## insertion run

pixscale = 60  # mas
img2dir = f'{DATA_PATH}/1.8.4_automatic_runs/v6/IMG2-9psfs-{pixscale}mas/'
img3dir = f'{DATA_PATH}/1.8.4_automatic_runs/v6/IMG3-9psfs-{pixscale}mas/'
procdir = f'{DATA_PATH}/1.8.4_automatic_runs/v6/9psfs-{pixscale}mas/'

# positions to insert; rough centers in final image, but it will be recomputed to be exactly the center of the pixel
coordinate_list = np.array([[53.1751187,-27.7665497],
                            [53.1511092,-27.7924772],
                            [53.1675452,-27.7942692],
                            [53.1550086,-27.7790141],
                            [53.1596492,-27.7841003],
                            [53.1729267,-27.7852995],
                            [53.1618306,-27.7707003],
                            [53.1667962,-27.7756657],
                            [53.1864842,-27.7801942]
                            ])

ra = coordinate_list[:,0]
dec = coordinate_list[:,1]

m_9psfs = M.MIRIPSF(ra, dec, img2dir, img3dir, procdir, pixscale=pixscale/1000.,  # arcsec
                    run_clean = f'{DATA_PATH}/1.8.4_automatic_runs/v6/9psfs-{pixscale}mas/HUDF_F560W_i2d.fits')  # file from the clean run (i.e. nothing inserted)

m_9psfs._update_coords_to_pixel_centers_in_clean()
m_9psfs._prep_img2dir()
m_9psfs._insert_psf_in_images(oversample_factor=2, boost_factor=1., write_psfonly=True)
m_9psfs._run_pipeline()
m_9psfs._copy_pipeline_output()
m_9psfs._subtract_original()
m_9psfs._cutout_psf(pixel_scale_ratio=0.055/m_9psfs.pixscale)
