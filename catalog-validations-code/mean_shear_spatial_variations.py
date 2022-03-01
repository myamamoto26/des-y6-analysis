
from statistics import mean
import fitsio as fio
import numpy as np
from matplotlib import pyplot as plt
import os, sys
# from scipy import stats
# import meds
from esutil import stat
import esutil as eu
# from scipy.optimize import curve_fit
from tqdm import tqdm
import json
# from joblib import Parallel, delayed
import time
import pickle
from skimage.measure import block_reduce
import drawDECam.drawDECam as dDECam
import matplotlib
import math

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
def _accum_shear_per_ccd(ccdres_all, ccdres, tilename):

    # Create ccd number key. 
    for ccdnum in list(ccdres):
        if ccdnum not in list(ccdres_all):
            ccdres_all[ccdnum] = {}
        if tilename not in list(ccdres_all[ccdnum]):
            ccdres_all[ccdnum][tilename] = {}

    cnames = ['g1', 'g2', 'g1p', 'g1m', 'g2p', 'g2m']
    for ccdnum in list(ccdres):
        for cname in cnames:
            ccdres_all[ccdnum][tilename][cname] = ccdres[ccdnum][cname]
            ccdres_all[ccdnum][tilename]["num_"+cname] = ccdres[ccdnum]["num_"+cname]

    return ccdres_all

def _accum_shear_from_file(ccdres_all, ccdres, x_side, y_side, per_ccd=False):

    if not per_ccd:
        # Create ccd number key. 
        for ccdnum in list(ccdres):
            if ccdnum not in list(ccdres_all):
                ccdres_all[ccdnum] = {}

        cnames = ['g1', 'g2', 'g1p', 'g1m', 'g2p', 'g2m']
        for ccdnum in list(ccdres):
            for cname in cnames:
                if cname not in list(ccdres_all[ccdnum]):
                    ccdres_all[ccdnum][cname] = np.zeros((y_side, x_side))
                    ccdres_all[ccdnum]["num_" + cname] = np.zeros((y_side, x_side))
                if cname in list(ccdres_all[ccdnum]):
                    rows,cols = np.where(~np.isnan(ccdres[ccdnum][cname]))
                    np.add.at(ccdres_all[cname], (rows, cols), ccdres[ccdnum][cname][rows, cols])
                    np.add.at(ccdres_all["num_"+cname], (rows, cols), ccdres[ccdnum]["num_"+cname][rows, cols])
                    # ccdres_all[ccdnum][cname] += ccdres[ccdnum][cname]
                    # ccdres_all[ccdnum]["num_"+cname] += ccdres[ccdnum]["num_"+cname]
    else:
        cnames = ['g1', 'g2', 'g1p', 'g1m', 'g2p', 'g2m']
        for tname in list(ccdres):
            for cname in cnames:
                if cname not in list(ccdres_all):
                    ccdres_all[cname] = np.zeros((y_side, x_side))
                    ccdres_all["num_" + cname] = np.zeros((y_side, x_side))
                if cname in list(ccdres_all):
                    rows,cols = np.where(~np.isnan(ccdres[tname][cname]))
                    np.add.at(ccdres_all[cname], (rows, cols), ccdres[tname][cname][rows, cols])
                    np.add.at(ccdres_all["num_"+cname], (rows, cols), ccdres[tname]["num_"+cname][rows, cols])
                    # ccdres_all[cname] += ccdres[tname][cname]
                    # ccdres_all["num_"+cname] += ccdres[tname]["num_"+cname]

    return ccdres_all

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

def _compute_g1_g2_per_ccd(ccdres):
    g1 = ccdres["g1"] / ccdres["num_g1"]
    g1p = ccdres["g1p"] / ccdres["num_g1p"]
    g1m = ccdres["g1m"] / ccdres["num_g1m"]
    R11 = (g1p - g1m) / 2 / 0.01

    g2 = ccdres["g2"] / ccdres["num_g2"]
    g2p = ccdres["g2p"] / ccdres["num_g2p"]
    g2m = ccdres["g2m"] / ccdres["num_g2m"]
    R22 = (g2p - g2m) / 2 / 0.01
    
    return g1/R11, g2/R22

def _compute_jackknife_cov(jk_x_g1, jk_y_g1, jk_x_g2, jk_y_g2, N):

    # compute jackknife average. 
    jk_x_g1_ave = np.sum(jk_x_g1, axis=0)/N
    jk_y_g1_ave = np.sum(jk_y_g1, axis=0)/N
    jk_x_g2_ave = np.sum(jk_x_g2, axis=0)/N
    jk_y_g2_ave = np.sum(jk_y_g2, axis=0)/N

    x_cov_g1 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_x_g1 - jk_x_g1_ave)**2, axis=0))
    y_cov_g1 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_y_g1 - jk_y_g1_ave)**2, axis=0))
    x_cov_g2 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_x_g2 - jk_x_g2_ave)**2, axis=0))
    y_cov_g2 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_y_g2 - jk_y_g2_ave)**2, axis=0))

    return x_cov_g1, y_cov_g1, x_cov_g2, y_cov_g2

def _categorize_obj_in_ccd(piece_side, nx, ny, ccd_x_min, ccd_y_min, x, y, msk_obj):

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

    # How this function works: Collect info (id, ra, dec, CCD coord, mean property values), save it, and plot later. 
    for pizza_f,band in zip(coadd_files, bands):
        coadd = fio.FITS(os.path.join('/global/cscratch1/sd/myamamot/pizza-slice/griz', pizza_f))
        try:
            epochs = coadd['epochs_info'].read()
            image_info = coadd['image_info'].read()
        except OSError:
            print('Corrupt file.?', pizza_f)
            raise OSError

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

            msk_obj = np.where(np.in1d(mdet_obj['slice_id'], unique_slices))[0]
            if len(msk_obj) == 0:
                continue
        
            n = len(msk_obj)
            ra_obj = mdet_obj['ra'][msk_obj]
            dec_obj = mdet_obj['dec'][msk_obj]

            # pos_x, pos_y = wcs.sky2image(ra_obj, dec_obj)
            pos_x, pos_y = gs_wcs.radecToxy(ra_obj, dec_obj, units="degrees")
            pos_x = pos_x - position_offset
            pos_y = pos_y - position_offset
            ccdnum = _get_ccd_num(image_info['image_path'][msk_im][0])
            xind, yind, msk_obj = _categorize_obj_in_ccd(piece_side, x_side, y_side, ccd_x_min, ccd_y_min, pos_x, pos_y, msk_obj)
            if (np.any(pos_x<=98) or np.any(pos_x>1950)):
                print('No objects in the buffer of total 98 pixels.')
                # print(pos_x[((pos_x<=98) | (pos_x>1950))])
            if (np.any(pos_y<=98) or np.any(pos_y>3998)):
                print('No objects in the buffer of total 98 pixels.')
                # print(pos_y[((pos_y<=98) | (pos_y>3998))])
            if ccdnum not in list(ccdres):
                ccdres[ccdnum] = {}
            mdet_step = mdet_obj["mdet_step"][msk_obj]
        
            ccdres = _accum_shear(ccdres, ccdnum, "g1", "noshear", mdet_step, xind, yind, mdet_obj["mdet_g_1"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g2", "noshear", mdet_step, xind, yind, mdet_obj["mdet_g_2"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g1p", "1p", mdet_step, xind, yind, mdet_obj["mdet_g_1"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g1m", "1m", mdet_step, xind, yind, mdet_obj["mdet_g_1"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g2p", "2p", mdet_step, xind, yind, mdet_obj["mdet_g_2"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g2m", "2m", mdet_step, xind, yind, mdet_obj["mdet_g_2"][msk_obj], x_side, y_side)

    return ccdres

def compute_shear_stack_CCDs(ccdres, x_side, y_side, stack_north_south=False, per_ccd=False):
        
    if per_ccd:
        g1, g2 = _compute_g1_g2_per_ccd(ccdres)
        mean_g1 = np.rot90(g1, 3)
        mean_g2 = np.rot90(g2, 3)
        return mean_g1, mean_g2

    ## stack all the CCDs to find any non-trivial trend in focal plane. ##
    ## Need to flip CCD 32-62 about x-axis. ##

    SECTIONS = dDECam.CCDSECTIONS
    # CCD 1-31
    stack_north_g1 = np.zeros((y_side, x_side))
    num_north_g1 = np.zeros((y_side, x_side))
    stack_south_g1 = np.zeros((y_side, x_side))
    num_south_g1 = np.zeros((y_side, x_side))

    # CCD 32-62
    stack_north_g2 = np.zeros((y_side, x_side))
    num_north_g2 = np.zeros((y_side, x_side))
    stack_south_g2 = np.zeros((y_side, x_side))
    num_south_g2 = np.zeros((y_side, x_side))

    # Stacking CCDs on top of each other. 
    for k, v in list(SECTIONS.items()):
        if k in list(ccdres):
            g1, g2 = _compute_g1_g2(ccdres, k)
        else:
            continue

        if k < 32:
            # stack_north_g1 = np.nansum(np.dstack((stack_north_g1, g1)), 2)
            rows,cols = np.where(~np.isnan(g1))
            np.add.at(stack_north_g1, (rows, cols), g1[rows, cols])
            np.add.at(num_north_g1, (rows, cols), 1)

            # stack_north_g2 = np.nansum(np.dstack((stack_north_g2, g2)), 2)
            rows,cols = np.where(~np.isnan(g2))
            np.add.at(stack_north_g2, (rows, cols), g2[rows, cols])
            np.add.at(num_north_g2, (rows, cols), 1)
        else:
            # g1 = np.flip(g1, 0)
            # stack_south_g1 = np.nansum(np.dstack((stack_south_g1, g1)), 2)
            rows,cols = np.where(~np.isnan(g1))
            np.add.at(stack_south_g1, (rows, cols), g1[rows, cols])
            np.add.at(num_south_g1, (rows, cols), 1)

            # g2 = np.flip(g2, 0)
            # stack_south_g2 = np.nansum(np.dstack((stack_south_g2, g2)), 2)
            rows,cols = np.where(~np.isnan(g2))
            np.add.at(stack_south_g2, (rows, cols), g2[rows, cols])
            np.add.at(num_south_g2, (rows, cols), 1)
    # print(np.sum(num_north_g1))

    if not stack_north_south: 
        mean_north_g1 = np.rot90(stack_north_g1/num_north_g1, 3)
        mean_north_g2 = np.rot90(stack_north_g2/num_north_g2, 3)
        mean_south_g1 = np.rot90(stack_south_g1/num_south_g1, 3)
        mean_south_g2 = np.rot90(stack_south_g2/num_south_g2, 3)
        mean_g1 = [mean_north_g1, mean_south_g1]
        mean_g2 = [mean_north_g2, mean_south_g2]
        return mean_g1, mean_g2
    else:
        # Stack north and south but be careful of the directions of stacking.
        g1 = (stack_north_g1+np.flip(stack_south_g1,0))/(num_north_g1+np.flip(num_south_g1,0))
        g2 = (stack_north_g2+np.flip(stack_south_g2,0))/(num_north_g2+np.flip(num_south_g2,0))
        mean_g1 = np.rot90(g1, 3)
        mean_g2 = np.rot90(g2, 3)
        return mean_g1, mean_g2

def plot_stacked_xy(x_side, y_side, ccdres, xbin, ybin, plot=False, jc=None, per_ccd=False):

    if per_ccd:
        mean_g1, mean_g2 = compute_shear_stack_CCDs(ccdres, x_side, y_side, per_ccd=True)
    else:
        mean_g1, mean_g2 = compute_shear_stack_CCDs(ccdres, x_side, y_side, stack_north_south=True)
    # Trimming around the edges. 
    mean_g1 = mean_g1[1:-1,1:-1]
    mean_g2 = mean_g2[1:-1,1:-1]
 
    x_data = []
    y_data = []
    for g in [mean_g1, mean_g2]:
        xbin_num = xbin
        ybin_num = ybin
        x_reduced = block_reduce(g, block_size=(1, y_side//xbin_num), func=np.nanmean)
        y_reduced = block_reduce(g, block_size=(x_side//ybin_num, 1), func=np.nanmean)
        
        x_stacked = np.nanmean(x_reduced, axis=0)
        y_stacked = np.nanmean(y_reduced, axis=1)
        x_data.append(x_stacked)
        y_data.append(y_stacked)
    if not plot:
        return x_data, y_data
    else:
        g1_flat = mean_g1.flatten()
        g2_flat = mean_g2.flatten()
        print(g1_flat[g1_flat < -10], g2_flat[g2_flat < -10])
        # plt.hist(mean_g1.flatten(), bins=200, histtype='step', range=(-0.1, 0.1), label=r'$e_{1}$')
        # plt.hist(mean_g2.flatten(), bins=200, histtype='step', range=(-0.1, 0.1), label=r'$e_{2}$')
        # plt.xlabel(r'$<e_{1,2}>$')
        # plt.legend()
        # plt.savefig('pixel_values_hist_xy.pdf')
        # plt.clf()
        fig, ax1 = plt.subplots(2,2,figsize=(35,18))
        # plt.style.use('default')
        matplotlib.rcParams.update({'font.size': 28})
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='k', alpha=1.)
        piece_side = 32
        X, Y = np.meshgrid(np.linspace(1, 3873, (3872//piece_side)+2), np.linspace(1, 1825, (1824//piece_side)+2))

        mean = np.nanmean(mean_g1[mean_g1 > -10])
        sig = np.nanstd(mean_g1[mean_g1 > -10]) / np.sqrt(mean_g1[mean_g1 > -10].size)
        mesh = ax1[0,0].pcolormesh(X,Y,mean_g1, vmin=mean-2*sig, vmax=mean+2*sig, cmap=cmap)
        ax1[0,0].set_aspect(1)
        ax1[0,0].set_title(r'$\langle e_{1} \rangle$', fontsize=22)
        ax1[0,0].set_xticks([])
        ax1[0,0].set_yticks([])
        plt.colorbar(mesh, ax=ax1[0,0], pad=0.01)

        mean = np.nanmean(mean_g2[mean_g2 > -10])
        sig = np.nanstd(mean_g2[mean_g2 > -10]) / np.sqrt(mean_g2[mean_g2 > -10].size)
        mesh = ax1[0,1].pcolormesh(X,Y,mean_g2, vmin=mean-2*sig, vmax=mean+2*sig, cmap=cmap)
        ax1[0,1].set_aspect(1)
        ax1[0,1].set_title(r'$\langle e_{2} \rangle$', fontsize=22)
        ax1[0,1].set_xticks([])
        ax1[0,1].set_yticks([])
        plt.colorbar(mesh, ax=ax1[0,1], pad=0.01)
        
        x_ = [1+i*(3872//len(x_stacked)) for i in range(len(x_stacked))] # np.linspace(1, 3872, len(x_stacked))
        y_ = [1+i*(1824//len(y_stacked)) for i in range(len(y_stacked))] # np.linspace(1, 1824, len(y_stacked))

        ax1[1,0].plot(x_, x_data[0], c='b', label='x-stacked')
        ax1[1,0].errorbar(x_, x_data[0], yerr=jc[0], c='b')
        ax1[1,0].plot(y_, y_data[0], c='r', label='y-stacked')
        ax1[1,0].errorbar(y_, y_data[0], yerr=jc[1], c='r')
        # ax1[1,0].set_ylim(-0.2,0.2)
        ax1[1,0].set_xlabel('CCD coordinates', fontsize=22)
        ax1[1,0].set_ylabel(r'$\langle e_{1} \rangle$', fontsize=22)
        # ax1[1,0].set_xticks([])

        ax1[1,1].plot(x_, x_data[1], c='b', label='x-stacked')
        ax1[1,1].errorbar(x_, x_data[1], yerr=jc[2], c='b')
        ax1[1,1].plot(y_, y_data[1], c='r', label='y-stacked')
        ax1[1,1].errorbar(y_, y_data[1], yerr=jc[3], c='r')
        # ax1[1,1].set_ylim(-0.2,0.2)
        ax1[1,1].set_xlabel('CCD coordinates', fontsize=22)
        ax1[1,1].set_ylabel(r'$\langle e_{2} \rangle$', fontsize=22)
        # ax1[1,1].set_xticks([])

        plt.legend(fontsize='large')
        # plt.tight_layout()
        plt.savefig('mdet_shear_variations_focal_plane_stacked_xy.pdf', bbox_inches='tight')
        return None


def plot_stacked_ccd_north_south(x_side, y_side, ccdres):
    
    x0 = dDECam.CCDSECTION_X0
    y0 = dDECam.CCDSECTION_Y0

    mean_g1, mean_g2 = compute_shear_stack_CCDs(ccdres, x_side, y_side, stack_north_south=False)
    for i in range(2):
        mean_g1[i] = mean_g1[i][1:-1, 1:-1]
        mean_g2[i] = mean_g2[i][1:-1, 1:-1]
    # plt.hist(mean_g2[1].flatten(), bins=200)
    # plt.xlabel(r'$<e_{2}>$')
    # plt.savefig('pixel_values_hist_southe2.pdf')
    # plt.clf()
    fig, ax1 = plt.subplots(2,2,figsize=(35,18))
    # plt.style.use('default')
    matplotlib.rcParams.update({'font.size': 28})
    cmap = plt.get_cmap('viridis')
    # cmap.set_bad(color='k', alpha=1.)
    piece_side = 32
    X, Y = np.meshgrid(np.linspace(1, 3873, (3872//piece_side)+1), np.linspace(1, 1825, (1824//piece_side)+1))
    
    mean = np.nanmean(mean_g1[0])
    sig = np.nanstd(mean_g1[0]) / np.sqrt(mean_g1[0].size)
    mesh = ax1[0,0].pcolormesh(X,Y,mean_g1[0], vmin=mean-2*sig, vmax=mean+2*sig, cmap=cmap)
    ax1[0,0].set_aspect(1)
    ax1[0,0].set_title(r'$\langle e_{1} \rangle$', fontsize=22)
    ax1[0,0].set_xticks([])
    ax1[0,0].set_yticks([])
    ax1[0,0].set_ylabel('North (CCD1-31)', fontsize=25)
    plt.colorbar(mesh, ax=ax1[0,0], pad=0.01)

    mean = np.nanmean(mean_g2[0])
    sig = np.nanstd(mean_g2[0]) / np.sqrt(mean_g2[0].size)
    mesh = ax1[0,1].pcolormesh(X,Y,mean_g2[0], vmin=mean-2*sig, vmax=mean+2*sig, cmap=cmap)
    ax1[0,1].set_aspect(1)
    ax1[0,1].set_title(r'$\langle e_{2} \rangle$', fontsize=22)
    ax1[0,1].set_xticks([])
    ax1[0,1].set_yticks([])
    plt.colorbar(mesh, ax=ax1[0,1], pad=0.01)

    mean = np.nanmean(mean_g1[1])
    sig = np.nanstd(mean_g1[1]) / np.sqrt(mean_g1[1].size)
    mesh = ax1[1,0].pcolormesh(X,Y,mean_g1[1], vmin=mean-2*sig, vmax=mean+2*sig, cmap=cmap)
    ax1[1,0].set_aspect(1)
    ax1[1,0].set_title(r'$\langle e_{1} \rangle$', fontsize=22)
    ax1[1,0].set_xticks([])
    ax1[1,0].set_yticks([])
    ax1[1,0].set_ylabel('South (CCD32-62)', fontsize=25)
    plt.colorbar(mesh, ax=ax1[1,0], pad=0.01)


    mean = np.nanmean(mean_g2[1][mean_g2[1] > -10])
    sig = np.nanstd(mean_g2[1][mean_g2[1] > -10]) / np.sqrt(mean_g2[1][mean_g2[1] > -10].size)
    mesh = ax1[1,1].pcolormesh(X,Y,mean_g2[1], vmin=mean-2*sig, vmax=mean+2*sig, cmap=cmap)
    ax1[1,1].set_aspect(1)
    ax1[1,1].set_title(r'$\langle e_{2} \rangle$', fontsize=22)
    ax1[1,1].set_xticks([])
    ax1[1,1].set_yticks([])
    plt.colorbar(mesh, ax=ax1[1,1], pad=0.01)

    plt.subplots_adjust(hspace=0.3,wspace=0.1)
    plt.savefig('mdet_shear_variations_focal_plane_stacked_north_south.pdf', bbox_inches='tight')


def plot_shear_vaiations_ccd(x_side, y_side, ccdres):

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

    drawDECamCCDs_Plot(x0,y0,ccdres,'e1',rotate=False,label=False,color='k',lw=0.5,ls='-')
    drawDECamCCDs_Plot(x0,y0,ccdres,'e2',rotate=False,label=False,color='k',lw=0.5,ls='-')

def main(argv):

    individual_tiles = True
    make_per_ccd_files = True
    work_mdet = '/global/project/projectdirs/des/myamamot/metadetect'
    work_mdet_cuts = '/global/project/projectdirs/des/myamamot/metadetect/cuts_v2'
    work = '/global/cscratch1/sd/myamamot'
    ccd_x_min = 48
    ccd_x_max = 2000
    ccd_y_min = 48
    ccd_y_max = 4048
    cell_side = 32
    x_side = int(np.ceil((ccd_x_max - ccd_x_min)/cell_side))
    y_side = int(np.ceil((ccd_y_max - ccd_y_min)/cell_side))
    num_ccd = 62

    # NEED TO WRITE THE CODE TO BE ABLE TO RUN FROM BOTH MASTER FLAT AND INDIVIDUAL FILES. 
    mdet_f = open('/global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt', 'r')
    mdet_fs = mdet_f.read().split('\n')[:-1]
    mdet_filenames = [fname.split('/')[-1] for fname in mdet_fs]
    tilenames = [d.split('_')[0] for d in mdet_filenames]

    if not individual_tiles:
        from mpi4py import MPI
        import galsim
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print('mpi', rank, size)
        
        # Obtain file, tile, band information from information file queried from desoper. 
        coadd_info = fio.read(os.path.join(work, 'pizza-slice/pizza-cutter-coadds-info.fits'))
        coadd_files = {t: [] for t in tilenames}
        bands = {t: [] for t in tilenames}
        for coadd in coadd_info:
            tname = coadd['FILENAME'].split('_')[0]
            fname = coadd['FILENAME'] + coadd['COMPRESSION']
            bandname = coadd['FILENAME'].split('_')[2]
            if tname in list(coadd_files.keys()):
                coadd_files[tname].append(fname)
                bands[tname].append(bandname)

        # Accumulate raw sums of shear and number of objects in each bin for each tile and save as a pickle file. 
        # When not using MPI, you can use for-loops (for t in tilenames)
        split_tilenames = np.array_split(tilenames, size)
        for t in tqdm(split_tilenames[rank]):
            ccdres = {}
            obj_num = 0
            if not os.path.exists('/global/project/projectdirs/des/myamamot/metadetect/shear_variations/mdet_shear_focal_plane_'+t+'.pickle'):
                d = fio.read(os.path.join(work_mdet_cuts, mdet_filenames[np.where(np.in1d(tilenames, t))[0][0]]))
                # msk = ((d['flags']==0) & (d['mask_flags']==0) & (d['mdet_s2n']>10) & (d['mdet_s2n']<100) & (d['mfrac']<0.02) & (d['mdet_T_ratio']>0.5) & (d['mdet_T']<1.2))
                ccdres = spatial_variations(ccdres, d, coadd_files[t], ccd_x_min, ccd_y_min, x_side, y_side, cell_side, t, bands[t])
                for c in list(ccdres.keys()):
                    obj_num += np.sum(ccdres[c]['num_g1'])
                print('number of objects in this tile, ', obj_num)
                with open('/global/cscratch1/sd/myamamot/metadetect/shear_variations/mdet_shear_focal_plane_'+t+'.pickle', 'wb') as raw:
                    pickle.dump(ccdres, raw, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print('Already made this tile.', t)
        comm.Barrier()
    else:
        # if make_per_ccd_files:
        #     print('Converting tiles to CCDs...')
        #     ccdres_all_ccd = {}
        #     for t in tqdm(tilenames):
        #         with open('/global/cscratch1/sd/myamamot/metadetect/shear_variations/mdet_shear_focal_plane_'+t+'.pickle', 'rb') as handle:
        #             ccdres = pickle.load(handle)
        #         ccdres_all_ccd = _accum_shear_per_ccd(ccdres_all_ccd, ccdres, t)
        #     for c in list(ccdres_all_ccd):
        #         with open('/global/cscratch1/sd/myamamot/metadetect/shear_variations/mdet_shear_focal_plane_ccd_'+str(c)+'.pickle', 'wb') as raw:
        #             pickle.dump(ccdres_all_ccd[c], raw, protocol=pickle.HIGHEST_PROTOCOL)

        # # Add raw sums for all the tiles from individual tile file. 
        # if not os.path.exists('/global/cscratch1/sd/myamamot/metadetect/shear_variations/mdet_shear_focal_plane_all.pickle'):
        #     ccdres_all = {}
        #     for t in tqdm(tilenames):
        #         with open('/global/cscratch1/sd/myamamot/metadetect/shear_variations/mdet_shear_focal_plane_'+t+'.pickle', 'rb') as handle:
        #             ccdres = pickle.load(handle)
        #         ccdres_all = _accum_shear_from_file(ccdres_all, ccdres, x_side, y_side)
        #     print(list(ccdres_all))
        #     with open('/global/cscratch1/sd/myamamot/metadetect/shear_variations/mdet_shear_focal_plane_all.pickle', 'wb') as raw:
        #         pickle.dump(ccdres_all, raw, protocol=pickle.HIGHEST_PROTOCOL)

        # else:
        #     with open('/global/cscratch1/sd/myamamot/metadetect/shear_variations/mdet_shear_focal_plane_all.pickle', 'rb') as raw:
        #         ccdres_all = pickle.load(raw)
            # plot for all the CCDs. 
            # print('Plotting...')
            # plot_shear_vaiations_ccd(x_side, y_side, ccdres_all) ->  now coded-up in jupyter-notebook. 
            # plot_stacked_ccd_north_south(x_side, y_side, ccdres_all) ->  now coded-up in jupyter-notebook. 

        # Compute jackknife error estimate. 
        print('Computing jackknife error')
        jk_sample = len(tilenames)
        xbin = 25
        ybin = 12

        # Set the mpi size to be 63 so that 1 CCD per task, save rank=0 for all the processing. 
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print('mpi', rank, size)

        all_ccd = {c:{'jk_x_g1':[], 'jk_y_g1':[], 'jk_x_g2':[], 'jk_y_g2':[]} for c in range(1,num_ccd+1)}
        all_ccd.pop(31)
        all_ccd.pop(61)
        for c in range(1,4): #num_ccd+1)):
            if c != rank:
                continue
            if c in [31, 61]:
                continue

            jk_x_g1 = np.zeros((jk_sample, xbin))
            jk_y_g1 = np.zeros((jk_sample, ybin))
            jk_x_g2 = np.zeros((jk_sample, xbin))
            jk_y_g2 = np.zeros((jk_sample, ybin))

            with open('/global/cscratch1/sd/myamamot/metadetect/2000tiles_test1/shear_variations/mdet_shear_focal_plane_ccd_'+str(c)+'.pickle', 'rb') as handle:
                ccdres = pickle.load(handle)

            for j,t in enumerate(list(ccdres)):
                print(t, rank)
                res = ccdres.copy()
                res.pop(t)
                ccdres_jk = {}
                ccdres_jk = _accum_shear_from_file(ccdres_jk, res, x_side, y_side, per_ccd=True)
            
                x_data, y_data = plot_stacked_xy(x_side, y_side, ccdres_jk, xbin, ybin, plot=False, per_ccd=True)
                jk_x_g1[j, :] = x_data[0] 
                jk_y_g1[j, :] = y_data[0]
                jk_x_g2[j, :] = x_data[1]
                jk_y_g2[j, :] = y_data[1]
                
            all_ccd[c]['jk_x_g1'].append(jk_x_g1)
            all_ccd[c]['jk_y_g1'].append(jk_y_g1)
            all_ccd[c]['jk_x_g2'].append(jk_x_g2)
            all_ccd[c]['jk_y_g2'].append(jk_y_g2)
        print('before comm barrier', rank)
        comm.Barrier()

        if rank != 0:
            if rank not in [31,61]:
                comm.send(all_ccd, dest=0)
        print('info sent', rank)
        comm.Barrier()
        if rank == 0:
            print('receiving info', rank)
            for r in range(1, size):
                if r not in [31,61]:
                    tmp_res = comm.recv(source=r)
                    all_ccd[r]['jk_x_g1'] = tmp_res[r]['jk_x_g1']
                    all_ccd[r]['jk_y_g1'] = tmp_res[r]['jk_y_g1']
                    all_ccd[r]['jk_x_g2'] = tmp_res[r]['jk_x_g2']
                    all_ccd[r]['jk_y_g2'] = tmp_res[r]['jk_y_g2']
        print(all_ccd)
        sys.exit()
        jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2 = _compute_jackknife_cov(jk_x_g1, jk_y_g1, jk_x_g2, jk_y_g2, len(tilenames))
        print('jackknife error estimate', jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2)
        jc = [jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2]
        jk_dict = {'jc_x_g1': jc_x_g1, 'jc_y_g1': jc_y_g1, 'jc_x_g2': jc_x_g2, 'jc_y_g2': jc_y_g2}
        with open('/global/cscratch1/sd/myamamot/metadetect/shear_variations/jk_cov_v2.pickle', 'wb') as handle:
            pickle.dump(jk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit()
        
        with open('/global/cscratch1/sd/myamamot/metadetect/shear_variations/mdet_shear_focal_plane_all.pickle', 'rb') as handle:
            ccdres = pickle.load(handle)
        # plot_stacked_xy(x_side, y_side, ccdres, xbin, ybin, plot=True, jc=jc)

    
if __name__ == "__main__":
    main(sys.argv)