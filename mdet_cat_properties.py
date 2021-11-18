

from re import T
import fitsio as fio
import numpy as np
import galsim
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import time
from numpy.core.shape_base import stack
from numpy.lib.function_base import flip
from numpy.lib.twodim_base import tril_indices
import seaborn as sns
from scipy import stats
import meds
from esutil import stat
import esutil as eu
from esutil.pbar import PBar
from scipy.optimize import curve_fit
from tqdm import tqdm
import json
from joblib import Parallel, delayed
import time
import pickle
import math
from skimage.measure import block_reduce
import drawDECam.drawDECam as dDECam
import matplotlib

sns.set()
mdet_pars = ['noshear', '1p', '1m', '2p' ,'2m']

PATH = "/data/des70.a/data/masaya/"
# PATH  = "/Users/masayayamamoto/Desktop/DarkEnergySurvey/des-y6-analysis"

def simple_properties():
    ## check how many objects didn't make each cut. 
    mdet_pars = ['noshear', '1p', '1m', '2p' ,'2m']
    for shear in mdet_pars:
        print(shear)
        nobj = len(d[d['mdet_step']==shear])
        print('total object number: ', nobj)
        mask = ((d['mdet_step'] == shear) & (d['flags'] != 0))
        print('non-zero flag object number: ', np.round((len(d[mask])/nobj)*100, 2), '%')
        mask = ((d['mdet_step'] == shear) & (d['flags'] != 0) & (d['mdet_s2n'] < 10))
        print('s2n < 10: ', np.round((len(d[mask])/nobj)*100, 2), '%')
        mask = ((d['mdet_step'] == shear) & (d['flags'] != 0) & (d['mdet_T_ratio'] < 1.2))
        print('mdet_T_ratio < 1.2: ', np.round((len(d[mask])/nobj)*100, 2), '%')
        mask = ((d['mdet_step'] == shear) & (d['flags'] != 0) & (d['mfrac'] > 1.0))
        print('mfrac > 1.0: ', np.round((len(d[mask])/nobj)*100, 2), '%')
        print('-------------------------------------')
    
    mdet_pars = ['noshear', '1p', '1m', '2p' ,'2m']
    fig, axs = plt.subplots(2,4,figsize=(20,12))

    quantities1 = ['mdet_s2n', 'mdet_T_ratio', 'psfrec_T', 'psfrec_g', 'mfrac', 'mdet_band_flux', 'mdet_band_flux', 'mdet_band_flux']
    flux_band = ['r', 'i', 'z']
    x = 0
    for q,ax in enumerate(axs.ravel()):
        #if q == 7:
        #    break
        #for shear in mdet_pars:
        shear = 'noshear' 
        if quantities1[q] == 'mdet_band_flux':
            msk = (
            (d['mdet_step'] == shear)
            & (d['flags'] == 0)
            & (d[quantities1[q]][:,x] > 0))
            vals = np.random.choice(d['mdet_band_flux'][msk][:,x], size=10000)
            ax.hist(np.log10(vals), bins=13, label=shear)
            ax.set_xlabel('log10('+flux_band[x]+'_band_flux)')
            #ax.legend()
        elif quantities1[q] == 'psfrec_g':
            msk1 = (
            (d['mdet_step'] == shear)
            & (d['flags'] == 0)
            )
            vals1 = np.random.choice(d['psfrec_g'][msk][:,0], size=10000)
            vals2 = np.random.choice(d['psfrec_g'][msk][:,1], size=10000)
            ax.hist(vals1, bins=13, label=shear)
            ax.hist(vals2, bins=13, label=shear)
            ax.set_xlabel('psfrec_g')
        else:
            msk = (
            (d['mdet_step'] == shear)
            & (d['flags'] == 0)
            & (d[quantities1[q]] > 0))
            vals = np.random.choice(d[quantities1[q]][msk], size=10000)
            if quantities1[q] in ['mdet_T_ratio', 'psfrec_T']:
                ax.hist(vals, bins=13, label=shear)
                ax.set_xlabel(quantities1[q])
                #ax.legend()
            else:
                ax.hist(np.log10(vals), bins=13, label=shear)
                ax.set_xlabel('log10('+quantities1[q]+')')
                #ax.legend()

        if quantities1[q] == 'mdet_band_flux':
            x+=1
    plt.savefig('mdet_test_noshear_properties.png')


def _make_cuts(d, shear, additional_cuts=None):

    if additional_cuts is None:
        # msk = (
        #     (d['flags'] == 0)
        #     & (d['mdet_s2n'] > 10)
        #     & (d['mdet_T_ratio'] > 1.2)
        #     & (d['mfrac'] < 0.1)
        #     & (d['mdet_step'] == shear)
        # )
        msk = (d['MDET_STEP'] == shear)
    else:
        # if additional_cuts['quantity'] == 'PSFREC_G':
        #     msk = (
        #         (d[additional_cuts['quantity']+'_'+additional_cuts['e']] > additional_cuts['cuts'][0])
        #         & (d[additional_cuts['quantity']+'_'+additional_cuts['e']] < additional_cuts['cuts'][1])
        #         & (d['MDET_STEP'] == shear)
        #     )
        # else:
        msk = (
            (d[additional_cuts['quantity']] > additional_cuts['cuts'][0])
            & (d[additional_cuts['quantity']] < additional_cuts['cuts'][1])
            & (d['MDET_STEP'] == shear)
        )
    mean_shear = [np.mean(d['MDET_G_1'][msk], axis=0), np.mean(d['MDET_G_2'][msk], axis=0)]
    mean_shear_err = bootstrap_sampling_error(200, d['MDET_G_1'][msk], d['MDET_G_2'][msk])
    len_cuts = len(d['MDET_G_1'][msk])
    return mean_shear, mean_shear_err, len_cuts

def calculate_response(d, additional_cuts=None):

    g_noshear, gerr_noshear, len_noshear = _make_cuts(d, 'noshear', additional_cuts=additional_cuts)
    g_1p, gerr_1p, len_1p = _make_cuts(d, '1p', additional_cuts=additional_cuts)
    g_1m, gerr_1m, len_1m = _make_cuts(d, '1m', additional_cuts=additional_cuts)
    g_2p, gerr_2p, len_2p = _make_cuts(d, '2p', additional_cuts=additional_cuts)
    g_2m, gerr_2m, len_2m = _make_cuts(d, '2m', additional_cuts=additional_cuts)

    R11 = (g_1p[0] - g_1m[0])/0.02
    R22 = (g_2p[1] - g_2m[1])/0.02
    R = [R11, R22]

    return R, g_noshear, gerr_noshear, len_noshear

def bootstrap_sampling_error(N, data1, data2):

    # data1 = data_set[:, 0]
    # data2 = data_set[:, 1]

    fi = []
    for n in range(N):
        sample1 = np.random.choice(np.arange(len(data1)),len(data1),replace=True)
        fi.append(np.mean(data1[sample1]))
    f_mean = np.sum(fi)/N 
    fi = np.array(fi)
    cov1 = np.sqrt(np.sum((fi-f_mean)**2)/(N-1))

    fi = []
    for n in range(N):
        sample2 = np.random.choice(np.arange(len(data2)),len(data2),replace=True)
        fi.append(np.mean(data2[sample2]))
    f_mean = np.sum(fi)/N 
    fi = np.array(fi)
    cov2 = np.sqrt(np.sum((fi-f_mean)**2)/(N-1))
    
    return [cov1, cov2]

def exclude_gold_mask_objects(d):

    import healpy as hp

    gold_mask = fio.read('/data/des70.a/data/masaya/gold/y6a2_foreground_mask_v1.1.fits')
    exclude_pix = np.unique(gold_mask['PIXEL'])
    hpix = hp.ang2pix(4096, d['RA'], d['DEC'], lonlat=True, nest=True)
    mask = np.in1d(hpix, exclude_pix, invert=True)
    
    return d[mask]


def mdet_shear_pairs_plotting_percentile(d, nperbin, cut_quantity):

    ## for linear fit
    def func(x,m,n):
        return m*x+n
    
    ## psf shape/area vs mean shear. 
    fig,axs = plt.subplots(5,2,figsize=(36,20))
    perc = [99,99,98,98,95,95,90,90,85,85]
    # def_mask = ((d['flags'] == 0) & (d['mdet_s2n'] > 10) & (d['mdet_T_ratio'] > 1.2) & (d['mfrac'] < 0.1))
    for q,ax in enumerate(axs.ravel()):
        T_max = np.percentile(d[cut_quantity], perc[q], axis=0)
        T_mask = (d[cut_quantity] < T_max)
        Tr = d['MDET_T_RATIO'][T_mask]
        hist = stat.histogram(Tr, nperbin=3000000, more=True)
        bin_num = len(hist['hist'])
        g_obs = np.zeros(bin_num)
        gerr_obs = np.zeros(bin_num)
        for i in tqdm(range(bin_num)):
            additional_cuts = {'quantity': 'MDET_T_RATIO', 'cuts': [hist['low'][i], hist['high'][i]]}
            R, g_mean, gerr_mean, bs = calculate_response(d, additional_cuts=additional_cuts)
            g_obs[i] = g_mean[q%2]/R[q%2]
            gerr_obs[i] = (gerr_mean[q%2]/R[q%2])
        
        params = curve_fit(func,hist['mean'],g_obs,p0=(0.,0.))
        m1,n1=params[0]
        x = np.linspace(hist['mean'][0], hist['mean'][bin_num-1], 100)

        ax.plot(x, func(x,m1,n1), label='linear fit')
        ax.errorbar(hist['mean'], g_obs, yerr=gerr_obs, fmt='o', fillstyle='none', label=str(100-perc[q])+'percent cut, Tmax='+str("{:2.2f}".format(T_max)))
        # if q%2 == 0:
        #     ax.set_ylim(-3e-3, 5e-3)
        # elif q%2 == 1:
        #     ax.set_ylim(-1e-2, 2e-2)
        ax.set_xlabel(r'$T_{ratio}$')
        ax.set_ylabel('<e'+str(q%2 + 1)+'>')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.legend(loc='upper left')
    plt.savefig('mdet_shear_Tcuts_v2.png')

def mdet_shear_pairs_plotting(d, nperbin):

    ## for linear fit
    def func(x,m,n):
        return m*x+n
    
    ## psf shape/area vs mean shear. 
    fig,axs = plt.subplots(3,2,figsize=(22,12))
    # exclude objects in healpix which is the same as the gold. 
    d = exclude_gold_mask_objects(d)
    for q,ax in enumerate(axs.ravel()):
        if q==0 or q==1:
            psf_ = d['PSFREC_G_'+str(q+1)]
            hist = stat.histogram(psf_, nperbin=nperbin, more=True)
            bin_num = len(hist['hist'])
        elif q==2 or q==3:
            Tpsf = d['PSFREC_T']
            hist = stat.histogram(Tpsf, nperbin=nperbin, more=True)
            bin_num = len(hist['hist'])
        elif q==4 or q==5:
            Tr = d['MDET_T_RATIO'] # d['mdet_T'][T_mask]
            hist = stat.histogram(Tr, nperbin=2000000, more=True)
            bin_num = len(hist['hist'])
        g_obs = np.zeros(bin_num)
        gerr_obs = np.zeros(bin_num)
        print(len(hist['low']), len(hist['mean']))
        for i in tqdm(range(bin_num)):
            if q==0 or q==1:
                additional_cuts = {'quantity': 'PSFREC_G_'+str(q+1), 'cuts': [hist['low'][i], hist['high'][i]]}
            elif q==2 or q==3:
                additional_cuts = {'quantity': 'PSFREC_T', 'cuts': [hist['low'][i], hist['high'][i]]}
            elif q==4 or q==5:
                additional_cuts = {'quantity': 'MDET_T_RATIO', 'cuts': [hist['low'][i], hist['high'][i]]}
                print(i, hist['low'][i], hist['high'][i])

            R, g_mean, gerr_mean, bs = calculate_response(d, additional_cuts=additional_cuts)
            g_obs[i] = g_mean[q%2]/R[q%2]
            gerr_obs[i] = (gerr_mean[q%2]/R[q%2])
        
        params = curve_fit(func,hist['mean'],g_obs,p0=(0.,0.))
        m1,n1=params[0]
        x = np.linspace(hist['mean'][0], hist['mean'][bin_num-1], 100)

        ax.plot(x, func(x,m1,n1), label='linear fit')
        ax.errorbar(hist['mean'], g_obs, yerr=gerr_obs, fmt='o', fillstyle='none', label='Y6 metadetect test')
        if q==0 or q==1:
            ax.set_xlabel('e'+str(q+1)+',PSF')
        elif q==2 or q==3:
            ax.set_xlabel(r'$T_{PSF}$')
        elif q==4 or q==5:
            ax.set_xlabel(r'$T_{ratio}$')
        ax.set_ylabel('<e'+str(q%2 + 1)+'>')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[0,0].legend(loc='upper right')
    plt.savefig('mdet_psf_vs_shear_fit_v2_goldmaskpix.png')


def categorize_obj_in_ccd(piece_side, nx, ny, ccd_x_min, ccd_y_min, x, y, msk_obj):

    xind = np.floor((x-ccd_x_min + 0.5)/piece_side).astype(int)
    yind = np.floor((y-ccd_y_min + 0.5)/piece_side).astype(int)

    msk_cut = np.where(
        (xind >= 0)
        & (xind < nx)
        & (yind >= 0)
        & (yind < ny)
    )[0]
    if len(msk_cut) == 0:
        return None

    msk_obj = msk_obj[msk_cut]
    xind = xind[msk_cut]
    yind = yind[msk_cut]

    return xind, yind, msk_obj


def spatial_variations(ccdres, mdet_obj, coadd_files, ccd_x_min, ccd_y_min, x_side, y_side, piece_side, t, bands):

    ## collect info (id, ra, dec, CCD coord, mean property values), save it, and plot later. 
    # mdet_cat = fio.read(os.path.join(PATH, 'metadetect/'+t+'_metadetect-v3_mdetcat_part0000.fits'))
    def _get_ccd_num(image_path):
        return int(image_path.split('/')[1].split('_')[2][1:])

    def _accum_shear(ccdres, ccdnum, cname, shear, mdet_step, xind, yind, g, x_side, y_side):
        msk_s = (mdet_step == shear)
        if cname not in list(ccdres[ccdnum]):
            ccdres[ccdnum][cname] = np.zeros((y_side, x_side))
            ccdres[ccdnum]["num_" + cname] = np.zeros((y_side, x_side))
        
        if np.any(msk_s):
            # see https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html#numpy.ufunc.at
            np.add.at(
                ccdres[ccdnum][cname], 
                (yind[msk_s], xind[msk_s]), 
                g[msk_s],
            )
            np.add.at(
                ccdres[ccdnum]["num_" + cname], 
                (yind[msk_s], xind[msk_s]), 
                np.ones_like(g[msk_s]),
            )

        return ccdres
    
    for pizza_f,band in zip(coadd_files, bands):
        coadd = fio.FITS(os.path.join('/data/des70.a/data/masaya/pizza-slice/v2/'+band+'_band/', pizza_f))
        epochs = coadd['epochs_info'].read()
        image_info = coadd['image_info'].read()
        
        #################################################
        ## ONLY A FEW OBJECTS FOR PARALLELIZATION TEST. #
        #################################################
        # mdet_cat = mdet_cat[:10]

        # For each metadetect object, find the slice and single epochs that it is in, 
        # and get the wcs the object is in, and convert the object's ra/dec into CCD coordinates.
        # After that, accumulate objects on each CCD, cut the CCD into smaller pieces, 
        # and compute the response in those pieces. 
        image_id = np.unique(epochs[(epochs['flags']==0)]['image_id'])
        image_id = image_id[image_id != 0]
        for iid in image_id:
            msk_im = np.where(image_info['image_id'] == iid)
            gs_wcs = galsim.FitsWCS(header=json.loads(image_info['wcs'][msk_im][0]))
            position_offset = image_info['position_offset'][msk_im][0]

            msk = ((epochs['flags'] == 0) & (epochs['image_id']==iid) & (epochs['weight'] > 0))
            if not np.any(msk):
                continue
            unique_slices = np.unique(epochs['id'][msk])

            msk_obj = np.where(np.in1d(mdet_obj['SLICE_ID'], unique_slices))[0]
            if len(msk_obj) == 0:
                continue
        
            n = len(msk_obj)
            ra_obj = mdet_obj['RA'][msk_obj]
            dec_obj = mdet_obj['DEC'][msk_obj]

            # pos_x, pos_y = wcs.sky2image(ra_obj, dec_obj)
            pos_x, pos_y = gs_wcs.radecToxy(ra_obj, dec_obj, units="degrees")
            pos_x = pos_x - position_offset
            pos_y = pos_y - position_offset
            ccdnum = _get_ccd_num(image_info['image_path'][msk_im][0])
            xind, yind, msk_obj = categorize_obj_in_ccd(piece_side, x_side, y_side, ccd_x_min, ccd_y_min, pos_x, pos_y, msk_obj)
            if (np.any(pos_x<=98) or np.any(pos_x>1950)):
                print('No objects in the buffer of total 98 pixels.')
                print(pos_x[((pos_x<=98) | (pos_x>1950))])
            if (np.any(pos_y<=98) or np.any(pos_y>3998)):
                print('No objects in the buffer of total 98 pixels.')
                print(pos_y[((pos_y<=98) | (pos_y>3998))])
            if ccdnum not in list(ccdres):
                ccdres[ccdnum] = {}
            mdet_step = mdet_obj["MDET_STEP"][msk_obj]
            ccdres = _accum_shear(ccdres, ccdnum, "g1", "noshear", mdet_step, xind, yind, mdet_obj["MDET_G_1"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g2", "noshear", mdet_step, xind, yind, mdet_obj["MDET_G_2"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g1p", "1p", mdet_step, xind, yind, mdet_obj["MDET_G_1"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g1m", "1m", mdet_step, xind, yind, mdet_obj["MDET_G_1"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g2p", "2p", mdet_step, xind, yind, mdet_obj["MDET_G_2"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g2m", "2m", mdet_step, xind, yind, mdet_obj["MDET_G_2"][msk_obj], x_side, y_side)

    return ccdres

def _compute_g1_g2(ccdres, ccdnum):
    g1 = ccdres[ccdnum]["g1"] / ccdres[ccdnum]["num_g1"]
    g1p = ccdres[ccdnum]["g1p"] / ccdres[ccdnum]["num_g1p"]
    g1m = ccdres[ccdnum]["g1m"] / ccdres[ccdnum]["num_g1m"]
    R11 = (g1p - g1m) / 2 / 0.01

    g2 = ccdres[ccdnum]["g2"] / ccdres[ccdnum]["num_g2"]
    g2p = ccdres[ccdnum]["g2p"] / ccdres[ccdnum]["num_g2p"]
    g2m = ccdres[ccdnum]["g2m"] / ccdres[ccdnum]["num_g2m"]
    R22 = (g2p - g2m) / 2 / 0.01
    
    return g1/R11, g2/R22

def calculate_tile_response(ccd_list, piece_ccd_tile, ver, batch, save=True):
    mean_shear_divisions = {l:[] for l in ccd_list}
    for div in ccd_list:
        div_data = piece_ccd_tile[div]
        if len(div_data) == 0:
            continue
            #mean_shear_divisions[div].append([None, None, None, None])
        else:
            ## make sure div_data to pass to calculate_response has the same data structure as d. 
            div_data = np.stack(piece_ccd_tile[div], axis=0)
            #print(div, len(div_data))
            R, g_mean, gerr_mean, length = calculate_response(div_data, additional_cuts=None)
            mean_shear_divisions[div].append(g_mean[0]/R[0])
            mean_shear_divisions[div].append(g_mean[1]/R[1])
            mean_shear_divisions[div].append(gerr_mean[0]/R[0])
            mean_shear_divisions[div].append(gerr_mean[1]/R[1])
    if save:
        with open('/data/des70.a/data/masaya/metadetect/'+ver+'/mdet_shear_variations_focal_plane_'+str(batch)+'.pickle', 'wb') as handle:
            pickle.dump(mean_shear_divisions, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    return mean_shear_divisions

def jackknife_cov(jk_x_mean_g1, jk_y_mean_g1, jk_x_mean_g2, jk_y_mean_g2, N):

    x_cov_g1 = np.sqrt(np.sum(jk_x_mean_g1**2, axis=0)/N - (np.sum(jk_x_mean_g1, axis=0)/N)**2) * np.sqrt(N-1)
    y_cov_g1 = np.sqrt(np.sum(jk_y_mean_g1**2, axis=0)/N - (np.sum(jk_y_mean_g1, axis=0)/N)**2) * np.sqrt(N-1)
    x_cov_g2 = np.sqrt(np.sum(jk_x_mean_g2**2, axis=0)/N - (np.sum(jk_x_mean_g2, axis=0)/N)**2) * np.sqrt(N-1)
    y_cov_g2 = np.sqrt(np.sum(jk_y_mean_g2**2, axis=0)/N - (np.sum(jk_y_mean_g2, axis=0)/N)**2) * np.sqrt(N-1)

    return x_cov_g1, y_cov_g1, x_cov_g2, y_cov_g2


def plot_shear_vaiations_ccd(x_side, y_side, ccdres, num_ccd, jk=False, jc=None):

    x0 = dDECam.CCDSECTION_X0
    y0 = dDECam.CCDSECTION_Y0

    # Draw DECam CCDs using Plot function (Unrotated)
    def rotate_xy(x, y, theta, x0=0, y0=0, units='degrees'):
        """
        Rotates (x,y) by angle theta and (x0,y0) translation
        """
        if units == 'degrees':
            d2r = math.pi / 180.  # degrees to radians shorthand
            theta = d2r * theta
        x_new = (x - x0) * math.cos(theta) - (y - y0) * math.sin(theta)
        y_new = (x - x0) * math.sin(theta) + (y - y0) * math.cos(theta)
        return x_new, y_new

    def stack_CCDs(ccdres, name, x_side, y_side):
        
        ## stack all the CCDs to find any non-trivial trend in focal plane. ##
        ## Need to flip CCD 32-62 about x-axis. ##

        SECTIONS = dDECam.CCDSECTIONS
        stack_g1 = np.zeros((y_side, x_side))
        num_g1 = np.zeros((y_side, x_side))
        stack_g2 = np.zeros((y_side, x_side))
        num_g2 = np.zeros((y_side, x_side))
        for k, v in list(SECTIONS.items()):
            if k in list(ccdres):
                g1, g2 = _compute_g1_g2(ccdres, k)
            else:
                continue

            if k < 32:
                stack_g1 = np.nansum(np.dstack((stack_g1, g1)), 2)
                rows,cols = np.where(~np.isnan(g1))
                np.add.at(num_g1, (rows, cols), 1)

                stack_g2 = np.nansum(np.dstack((stack_g2, g2)), 2)
                rows,cols = np.where(~np.isnan(g2))
                np.add.at(num_g2, (rows, cols), 1)
            else:
                g1 = np.flip(g1, 0)
                stack_g1 = np.nansum(np.dstack((stack_g1, g1)), 2)
                rows,cols = np.where(~np.isnan(g1))
                np.add.at(num_g1, (rows, cols), 1)

                g2 = np.flip(g2, 0)
                stack_g2 = np.nansum(np.dstack((stack_g2, g2)), 2)
                rows,cols = np.where(~np.isnan(g2))
                np.add.at(num_g2, (rows, cols), 1)
        mean_g1 = np.rot90(stack_g1/num_g1)
        mean_g2 = np.rot90(stack_g2/num_g2)

        if jk:
            x_data = []
            y_data = []
            for g in [mean_g1, mean_g2]:
                xbin_num = 25
                ybin_num = 15
                x_reduced = block_reduce(g, block_size=(1, y_side//xbin_num), func=np.nanmean)
                y_reduced = block_reduce(g, block_size=(x_side//ybin_num, 1), func=np.nanmean)
                x_stacked = np.nanmean(x_reduced, axis=0)
                y_stacked = np.nanmean(y_reduced, axis=1)
                x_data.append(x_stacked)
                y_data.append(y_stacked)
            return x_data, y_data

        # ax = plt.gca()
        fig, ax1 = plt.subplots(2,2,figsize=(35,18))
        # plt.style.use('default')
        matplotlib.rcParams.update({'font.size': 28})
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='k', alpha=1.)
        piece_side = 32
        # X, Y = np.meshgrid(np.linspace(1, 2049, (2048//piece_side)+1), np.linspace(1, 4097, (4096//piece_side)+1))
        # mesh = ax1[0].pcolormesh(X,Y,stacked_mean_shear, vmin=-0.01, vmax=0.01)
        X, Y = np.meshgrid(np.linspace(1, 4001, (4000//piece_side)+1), np.linspace(1, 1953, (1952//piece_side)+1))
        
        mesh = ax1[0,0].pcolormesh(X,Y,mean_g1, vmin=-0.05, vmax=0.05, cmap=cmap)
        ax1[0,0].set_aspect(1)
        ax1[0,0].set_title('<e1>', fontsize=20)
        ax1[0,0].set_xticks([])
        ax1[0,0].set_yticks([])
        plt.colorbar(mesh, orientation='horizontal', ax=ax1[0], pad=0.03)

        mesh = ax1[0,1].pcolormesh(X,Y,mean_g2, vmin=-0.05, vmax=0.05, cmap=cmap)
        ax1[0,1].set_aspect(1)
        ax1[0,1].set_title('<e2>', fontsize=20)
        ax1[0,1].set_xticks([])
        ax1[0,1].set_yticks([])
        # plt.colorbar(mesh, orientation='horizontal', ax=ax1[0], pad=0.03)

        ## stack this 61x125 CCD in 10 bins along the x or y directions.
        xbin_num = 25
        ybin_num = 15
        x_reduced = block_reduce(mean_g1, block_size=(1, y_side//xbin_num), func=np.nanmean)
        y_reduced = block_reduce(mean_g1, block_size=(x_side//ybin_num, 1), func=np.nanmean)
        x_stacked = np.nanmean(x_reduced, axis=0)
        y_stacked = np.nanmean(y_reduced, axis=1)
        
        x_ = np.linspace(0,1,len(x_stacked))
        y_ = np.linspace(0,1,len(y_stacked))
        ax1[1,0].plot(x_, x_stacked, c='b', label='x-stacked')
        ax1[1,0].errorbar(x_, x_stacked, yerr=jc[0], c='b')
        ax1[1,0].plot(y_, y_stacked, c='r', label='y-stacked')
        ax1[1,0].errorbar(y_, y_stacked, yerr=jc[1], c='r')
        ax1[1,0].set_ylim(-0.2,0.2)
        ax1[1,0].set_xlabel('CCD coordinates')
        ax1[1,0].set_ylabel('<e1>')
        ax1[1,0].set_xticks([])

        x_reduced = block_reduce(mean_g2, block_size=(1, y_side//xbin_num), func=np.nanmean)
        y_reduced = block_reduce(mean_g2, block_size=(x_side//ybin_num, 1), func=np.nanmean)
        x_stacked = np.nanmean(x_reduced, axis=0)
        y_stacked = np.nanmean(y_reduced, axis=1)

        ax1[1,1].plot(x_, x_stacked, c='b', label='x-stacked')
        ax1[1,1].errorbar(x_, x_stacked, yerr=jc[2], c='b')
        ax1[1,1].plot(y_, y_stacked, c='r', label='y-stacked')
        ax1[1,1].errorbar(y_, y_stacked, yerr=jc[3], c='r')
        ax1[1,1].set_ylim(-0.2,0.2)
        ax1[1,1].set_xlabel('CCD coordinates')
        ax1[1,1].set_ylabel('<e2>')
        ax1[1,1].set_xticks([])

        plt.legend(fontsize='large')
        # plt.savefig('mdet_shear_variations_focal_plane_stacked_'+name+'.pdf')
        return


    def drawDECamCCDs_Plot(x0, y0, ccdres, name, trim=False, rotate=True, label=False, **kwargs):
        """
        Draws DECam CCDs shapes using matplotlib Plot function on the current plot
        """
        plt.figure(1,figsize=(20,20))
        matplotlib.rcParams.update({'font.size': 28})
        ax = plt.gca()
        if trim:
            TRIM_CCDSECTIONS = dDECam.CCDSECTIONS.copy()
            borderpix = 104  # 208/2. as 208 is the space between chips in pixels
            for _k, _v in list(TRIM_CCDSECTIONS.items()):
                (_x1, _x2, _y1, _y2) = _v
                _x1 = _x1 + borderpix
                _x2 = _x2 - borderpix
                _y1 = _y1 + borderpix
                _y2 = _y2 - borderpix
                TRIM_CCDSECTIONS[_k] = [_x1, _x2, _y1, _y2]
            SECTIONS = TRIM_CCDSECTIONS
        else:
            SECTIONS = dDECam.CCDSECTIONS

        for k, v in list(SECTIONS.items()):
            (x1, x2, y1, y2) = v
            # if rotate:
            #     x1, y1 = rotate_xy(x1, y1, theta=-90, x0=x0, y0=y0)
            #     x2, y2 = rotate_xy(x2, y2, theta=-90, x0=x0, y0=y0)
            # else:
            #     x1, y1 = rotate_xy(x1, y1, theta=0, x0=x0, y0=y0)
            #     x2, y2 = rotate_xy(x2, y2, theta=0, x0=x0, y0=y0)
            if k in list(ccdres):
                g1, g2 = _compute_g1_g2(ccdres, k)
            else:
                continue
            # Into numpy arrays
            x = np.array([x1, x2, x2, x1, x1])
            y = np.array([y1, y1, y2, y2, y1])
            # ax.plot(x, y, **kwargs)

            cmap = plt.get_cmap('viridis')
            # cmap.set_bad(color='black', alpha=1.)
            X, Y = np.meshgrid(np.linspace(x1+48, x2-48, x_side+1), np.linspace(y1+48, y2-48, y_side+1))
            if name == 'e1':
                mesh = ax.pcolormesh(X, Y, g1, vmin=-0.05, vmax=0.05, snap=True, cmap=cmap)
            elif name == 'e2':
                mesh = ax.pcolormesh(X, Y, g2, vmin=-0.05, vmax=0.05, snap=True, cmap=cmap)
            #ax.imshow(shear_e1, origin='lower', extent=[x1,x2,y1,y2])
            if label:
                ax.text(0.5 * (x2 + x1), 0.5 * (y2 + y1), "CCD%s" %
                        k, ha='center', va='center')
        
        ax.set_xlim(-2000,32000)
        ax.set_ylim(-2000,32000)
        ax.set_aspect(1)
        ax.set_title(name, fontsize=20)
        plt.tight_layout()
        plt.colorbar(mesh, ax=ax)
        plt.savefig('mdet_shear_variations_focal_plane_'+name+'.pdf')
        plt.clf()
        return

    # drawDECamCCDs_Plot(x0,y0,ccdres,'e1',rotate=False,label=False,color='k',lw=0.5,ls='-')
    # drawDECamCCDs_Plot(x0,y0,ccdres,'e2',rotate=False,label=False,color='k',lw=0.5,ls='-')
    stack_CCDs(ccdres, 'all', x_side, y_side)
    # print('saved figure...')


def main(argv):

    ver = 'v2'
    if sys.argv[1] == 'shear_pair':

        if not os.path.exists(os.path.join(PATH, 'metadetect/'+ver+'/mdet_test_all_'+ver+'.fits')):
            f = open('/home/s1/masaya/des-y6-analysis/tiles.txt', 'r')
            tilenames = f.read().split('\n')[:-1]
            start = 0
            for f in tilenames:
                print('Reading in '+f+'...')
                if start == 0:
                    d = fio.read(os.path.join(PATH, f+'_metadetect-v3_mdetcat_part0000.fits'))
                    start += 1
                else:
                    try: 
                        d2 = fio.read(os.path.join(PATH, f+'_metadetect-v3_mdetcat_part0000.fits'))
                        d = np.concatenate((d,d2))
                    except OSError:
                        print(f+' tile does not exist. Please check the catalog.')
                        continue
            fio.write(os.path.join(PATH, 'metadetect/mdet_test_all.fits'), d)
        else:
            d = fio.read(os.path.join(PATH, 'metadetect/'+ver+'/mdet_test_all_'+ver+'.fits'))

        # simple_properties()
        # mdet_shear_pairs(40, 1000)
        mdet_shear_pairs_plotting(d, 4000000)
        # mdet_shear_pairs_plotting_percentile(d, 4000000, 'MDET_T')
    elif sys.argv[1] == 'shear_spatial':
        just_plot = True
        plotting = False
        save_raw = True
        work = '/data/des70.a/data/masaya'
        ccd_x_min = 48
        ccd_x_max = 2000
        ccd_y_min = 48
        ccd_y_max = 4048
        piece_side = 32
        ver = 'v2'
        x_side = int(np.ceil((ccd_x_max - ccd_x_min)/piece_side))
        y_side = int(np.ceil((ccd_y_max - ccd_y_min)/piece_side))
        pieces = x_side * y_side
        num_ccd = 62
        div_tiles = np.resize(np.array([k for k in range(1,pieces+1)]), (y_side, x_side))
        ccdres = {}
        t0 = time.time()

        f = fio.read(os.path.join(work, 'metadetect/'+ver+'/mdet_test_all_v2.fits'))
        tilenames = np.unique(f['TILENAME'])
        if not just_plot:
            
            coadd_f = fio.read(os.path.join(work, 'pizza-slice/'+ver+'/pizza_slices_coadd_v2.fits')) # Made from make_download_files_v2.py
            coadd_files = {t: [] for t in tilenames}
            bands = {t: [] for t in tilenames}
            for coa in coadd_f:
                tname = coa['FILENAME'][:12]
                coadd_files[tname].append(coa['FILENAME'])
                bands[tname].append(coa['BAND'])

            # array_split = 5
            # ii = int(sys.argv[2])
            # split_tilenames = np.array_split(tilenames, array_split)[ii]
            # print('Processing the '+str(ii)+' batch...')

            t0 = time.time()
            for t in tqdm(tilenames):
                ccdres = {}
                ccdres = spatial_variations(ccdres, f[f['TILENAME']==t], coadd_files[t], ccd_x_min, ccd_y_min, x_side, y_side, piece_side, t, bands[t])
                if save_raw:
                    # with open('/data/des70.a/data/masaya/metadetect/'+ver+'/mdet_shear_focal_plane_'+str(ii)+'.pickle', 'wb') as raw:
                    with open('/data/des70.a/data/masaya/metadetect/'+ver+'/mdet_shear_focal_plane_'+t+'.pickle', 'wb') as raw:
                        pickle.dump(ccdres, raw, protocol=pickle.HIGHEST_PROTOCOL)
                        print('saving dict to a file...', t)
            sys.exit()

            ## starting from the middle. ##
            if plotting:
                print('Plotting...')
                #if just_plot:
                with open('/data/des70.a/data/masaya/metadetect/'+ver+'/mdet_shear_variations_focal_plane.pickle', 'rb') as handle:
                    ccdres = pickle.load(handle)
                plot_shear_vaiations_ccd(x_side, y_side, ccdres, num_ccd)
            
            print('time it took, ', time.time()-t0)
        
        else:
            print('Plotting...')
            # with open('./binshear_test/mdet_shear_focal_plane_0.pickle', 'rb') as handle:
            jk_x_g1_mean = []
            jk_y_g1_mean = []
            jk_x_g2_mean = []
            jk_y_g2_mean = []
            for i in tqdm(range(len(tilenames))):
                ccdres = {}
                for j,t in enumerate(tilenames):
                    if i == j:
                        continue
                    if len(ccdres) == 0:
                        with open('/data/des70.a/data/masaya/metadetect/'+ver+'/mdet_shear_focal_plane_'+t+'.pickle', 'rb') as handle:
                            ccdres = pickle.load(handle)
                        continue
                    else:
                        with open('/data/des70.a/data/masaya/metadetect/'+ver+'/mdet_shear_focal_plane_'+t+'.pickle', 'rb') as handle:
                            ccdres_ = pickle.load(handle)
                        for k in range(1,63):
                            if (k in list(ccdres)) and (k in list(ccdres_)):
                                ccdres[k]['g1'] = ccdres[k]['g1'] + ccdres_[k]['g1']
                                ccdres[k]['num_g1'] = ccdres[k]['num_g1'] + ccdres_[k]['num_g1']
                                ccdres[k]['g1p'] = ccdres[k]['g1p'] + ccdres_[k]['g1p']
                                ccdres[k]['num_g1p'] = ccdres[k]['num_g1p'] + ccdres_[k]['num_g1p']
                                ccdres[k]['g1m'] = ccdres[k]['g1m'] + ccdres_[k]['g1m']
                                ccdres[k]['num_g1m'] = ccdres[k]['num_g1m'] + ccdres_[k]['num_g1m']
                                ccdres[k]['g2'] = ccdres[k]['g2'] + ccdres_[k]['g2']
                                ccdres[k]['num_g2'] = ccdres[k]['num_g2'] + ccdres_[k]['num_g2']
                                ccdres[k]['g2p'] = ccdres[k]['g2p'] + ccdres_[k]['g2p']
                                ccdres[k]['num_g2p'] = ccdres[k]['num_g2p'] + ccdres_[k]['num_g2p']
                                ccdres[k]['g2m'] = ccdres[k]['g2m'] + ccdres_[k]['g2m']
                                ccdres[k]['num_g2m'] = ccdres[k]['num_g2m'] + ccdres_[k]['num_g2m']
    
                data = plot_shear_vaiations_ccd(x_side, y_side, ccdres, num_ccd, jk=True)
                print(len(data))
                sys.exit()
                jk_x_g1_mean.append(x_data[0])
                jk_y_g1_mean.append(y_data[0])
                jk_x_g2_mean.append(x_data[1])
                jk_y_g2_mean.append(y_data[1])
            jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2 = jackknife_cov(jk_x_g1_mean, jk_y_g1_mean, jk_x_g2_mean, jk_y_g2_mean, len(tilenames))
            with open('/data/des70.a/data/masaya/metadetect/'+ver+'/mdet_shear_focal_plane_all.pickle', 'rb') as handle:
                ccdres = pickle.load(handle)
                plot_shear_vaiations_ccd(x_side, y_side, ccdres, num_ccd, jk=False, jc=[jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2])

if __name__ == "__main__":
    main(sys.argv)











