"""
This file is a copy of JWST_1.8.4_HUDF_stage3.py
by Leindert Boogaard on 17.03.23 for modifications
updated by Leindert Boogaard on 18.06.23 for integration in psf pipeline

Created on Thu Jan 13 15:01:08 2022
Purpose: stage3 pipeline reduction for HUDF
@author: srigi
"""

###########################
#standard modules
import numpy as np
import sys,sep,os,glob,datetime
from astropy.io import fits
import matplotlib.pyplot as plt;plt.ioff()
from photutils import make_source_mask
from photutils.segmentation import SegmentationImage
from reproject import reproject_interp
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from photutils.background import Background2D, MedianBackground,BiweightLocationBackground
from astropy.stats import SigmaClip
#from astroscrappy import detect_cosmics
from astropy.nddata.bitmask import bitfield_to_boolean_mask, interpret_bit_flags
from astropy.modeling import models as apy_models
from astropy.modeling import fitting as apy_fitting
from astropy.convolution import convolve as apy_convolve
from astropy.convolution import Gaussian2DKernel
from scipy.ndimage import median_filter as scipy_median_filter
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable

#jwst modules
import jwst,crds,stdatamodels
from jwst import datamodels
from jwst.datamodels import ImageModel, FlatModel, dqflags
from jwst.flatfield.flat_field import do_correction, do_flat_field, apply_flat_field
from stdatamodels.util import create_history_entry
from jwst.pipeline import Detector1Pipeline,Image2Pipeline,Image3Pipeline
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase
from jwst.associations.asn_from_list import asn_from_list
from jwst.associations.lib.member import Member
from jwst.datamodels.dqflags import pixel as dqflags_pixel
from jwst.skymatch import SkyMatchStep
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.associations.lib.rules_level3 import Asn_Lv3Image


#  Pipeline Verion
VER=jwst.__version__
print (VER,'set to 1.8.4')
VER='1.8.4'

#Context
os.environ["CRDS_CONTEXT"] = "jwst_1019.pmap"
#os.environ["CRDS_PATH"]="/home/projects/dtu_00026/people/stegil/crds_cache/jwst_ops"
print('CRDS CONTEXT: ',os.environ["CRDS_CONTEXT"])
print('CRDS URL: ',os.environ["CRDS_SERVER_URL"])
print('CRDS PATH: ',os.environ["CRDS_PATH"])

## abs_refcat from clean run
abs_refcat_clean = '/Users/boogaard/Sterrenkunde/Instruments/JWST/DATA/1283/MIRI/PSF-hudf/Insertion/1.8.4/asns-clean/fit_gaiadr2_ref_F560W.ecsv'

###########################

def run_Image3Pipeline(raw_direc,
                       output_DIR,
                       pix_scale=0.06,
                       pixfrac=1.0):
    ###########
    #CALIMG3 Default
    ######switches#######

    #define input and output folders
#    raw_direc=VER+'/IMG2/aligned'
#    output_DIR=VER+'/IMG3/wo_stripes/skyflat'
    # raw_direc=VER+'/IMG2-southonly/'
    # output_DIR=VER+'/IMG3-southonly/'
   # raw_direc=VER+'/IMG2-psfnorth-v5/'
    # output_DIR=VER+'/IMG3-psfnorth-v5/'

    # info
    # clean = fix abs_refcat, this fixed alignment
    # clean-v2 = disable skymatch and outlier detection
    # raw_direc=VER+'/IMG2-clean/'
    # output_DIR=VER+'/IMG3-clean-v2/'

    # from this point we used the clean catalog for tweakreg
    # raw_direc=VER+'/IMG2-psfv5/'
    # output_DIR=VER+'/IMG3-psfv5-v2/'

    #define filters
    Filters=['F560W']

    try:
        os.makedirs(output_DIR)
    except:
        pass

    for FF in range(len(Filters)):

        #filter
        Filter=Filters[FF]

        #get files
        infits=sorted(glob.glob(os.path.join(raw_direc,'*'+Filter+'*_cal.fits')))
        asn_name = 'HUDF_'+Filter  #set the association name

        # create an association
        asn = asn_from_list(infits, product_name=asn_name)

        # set some metadata
        asn['asn_pool'] = asn_name + '_pool'
        asn['asn_type'] = 'image3'

        # print the association and save to file
        name, ser = asn.dump()
        asn_file = asn_name + '_lvl3_asn.json'
        with open(asn_file, 'w') as f:
            f.write(ser)

        #get box size
        header=fits.open(infits[0])['SCI'].header
        box_size = min(header['NAXIS1'], header['NAXIS2']) // 30
        print('box size',box_size)

        #set pixel scale
        # pix_scale=0.06   # 0.04
        # pixfrac=1.0
        pix_scale_ratio=0.111/pix_scale

        #run pipeline
        Image3Pipeline.call(asn_file, save_results=True, output_dir=output_DIR,
        steps={'tweakreg':{#'abs_refcat':'GAIADR2',#'tables/hlf_ra_dec_cut.fits',#~900 sources selected by F160W, ''
                           #'abs_refcat':'1.8.4/asns-clean/fit_gaiadr2_ref_F560W.ecsv',  # LAB: use the one from the clean run
                           'abs_refcat': abs_refcat_clean,
                           'save_abs_catalog':True,
                           'minobj':'7', #default is 15
                           'searchrad':'0.6', #default is 2.0
                           'separation':'1.0', #default is 0.1
                           'tolerance':'0.3', #default is 0.7 arcsec
                           'abs_minobj':'7', #default is 15
                           'abs_searchrad':'0.6', #default is 2.0
                           'abs_separation':'1.0', #default is 0.1
                           'abs_tolerance':'0.3', #default is 0.7 arcsec
                           #'expand_refcat':True,
                           'expand_refcat':False,    #LAB: keep refcat fixed to that from the clean run
                           'snr_threshold':'2.0',
                           'enforce_user_order':True,
                           'searchrad':'2.0',
                           'sigma':'1.5',
                           'abs_sigma':1.5,
                           'abs_nclip':'2',
                           'nclip':2,  #'brightest':'5',
                           'search_output_file':False,
                           'output_use_model':True,
            'skip': True},
                'outlier_detection':{'pixfrac':pixfrac, 'skip':True},     # LAB can we modify this? don't skip.
                'skymatch':{'subtract':True,                 # LAB can turn this off
                            'skymethod':"global+match",      # or we do other differences
                            'save_results':True, 'skip':True},
                'resample':{'kernel':'square',# see -- https://jwst-pipeline.readthedocs.io/en/latest/jwst/resample/arguments.html
                            'pixfrac':pixfrac,
                            'pixel_scale_ratio':pix_scale_ratio,
                            'pixel_scale':pix_scale},
                'source_catalog':{'bkg_boxsize':box_size,
                                'kernel_fwhm':3.0,
                                'snr_threshold':1.0,
                                  'npixels':3,
                                  'skip':True}})

        #mv other outputs to folders
        # os.system('mv '+asn_file+' '+VER+'/asns/.')
        # os.system('mv fit_gaiadr2_ref.ecsv '+VER+'/asns/fit_gaiadr2_ref_'+Filter+'.ecsv')
        # os.system('mv *_outlier_i2d.fits '+output_DIR+'/.')

        # cleanup everything except the following
        try:
            all_files = glob.glob(os.path.join(output_DIR,'*.fits'))
            all_files.remove('HUDF_F560W_i2d.fits')
            #all_files.remove('HUDF_F560W_segm.fits')
            for f in all_files:
                print("os.system('rm '+ f)")
        except:
            print('failed cleanup')
            print('glob was ', all_files)
