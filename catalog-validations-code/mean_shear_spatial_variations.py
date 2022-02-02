
import fitsio as fio
import numpy as np
import galsim
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
from mpi4py import MPI

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

def _accum_shear_from_file(ccdres_all, ccdres, x_side, y_side):

    for ccdnum in list(ccdres):
        if ccdnum not in list(ccdres_all):
            ccdres_all[ccdnum] = {}

    step_names = ['noshear', 'g1p', 'g1m', 'g2p', 'g2m']
    for ccdnum in list(ccdres):
        for cname in step_names:
            if cname not in list(ccdres_all[ccdnum]):
                ccdres_all[ccdnum][cname] = np.zeros((y_side, x_side))
                ccdres_all[ccdnum]["num_" + cname] = np.zeros((y_side, x_side))
            if cname in list(ccdres[ccdnum]):
                ccdres_all[ccdnum][cname] += ccdres[ccdnum][cname]
                ccdres_all[ccdnum]["num_"+cname] += ccdres[ccdnum]["num_"+cname]

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

def _compute_jackknife_cov(jk_x_g1, jk_y_g1, jk_x_g2, jk_y_g2, N):

    # compute jackknife average. 
    jk_x_g1_ave = np.sum(jk_x_g1, axis=0)/N
    jk_y_g1_ave = np.sum(jk_y_g1, axis=0)/N
    jk_x_g2_ave = np.sum(jk_x_g2, axis=0)/N
    jk_y_g2_ave = np.sum(jk_y_g2, axis=0)/N

    x_cov_g1 = np.sqrt(np.sum((jk_x_g1 - jk_x_g1_ave)**2, axis=0)/(N*(N-1)))
    y_cov_g1 = np.sqrt(np.sum((jk_y_g1 - jk_y_g1_ave)**2, axis=0)/(N*(N-1)))
    x_cov_g2 = np.sqrt(np.sum((jk_x_g2 - jk_x_g2_ave)**2, axis=0)/(N*(N-1)))
    y_cov_g2 = np.sqrt(np.sum((jk_y_g2 - jk_y_g2_ave)**2, axis=0)/(N*(N-1)))

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
        epochs = coadd['epochs_info'].read()
        image_info = coadd['image_info'].read()

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
            return (x_data, y_data)
        else:
            # ax = plt.gca()
            fig, ax1 = plt.subplots(2,2,figsize=(35,18))
            # plt.style.use('default')
            matplotlib.rcParams.update({'font.size': 28})
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='k', alpha=1.)
            piece_side = 32
            X, Y = np.meshgrid(np.linspace(1, 4001, (4000//piece_side)+1), np.linspace(1, 1953, (1952//piece_side)+1))
            
            mesh = ax1[0,0].pcolormesh(X,Y,mean_g1, vmin=-0.05, vmax=0.05, cmap=cmap)
            ax1[0,0].set_aspect(1)
            ax1[0,0].set_title(r'$\langle e_{1} \rangle$', fontsize=22)
            ax1[0,0].set_xticks([])
            ax1[0,0].set_yticks([])
            plt.colorbar(mesh, orientation='horizontal', ax=ax1[0], pad=0.03)

            mesh = ax1[0,1].pcolormesh(X,Y,mean_g2, vmin=-0.05, vmax=0.05, cmap=cmap)
            ax1[0,1].set_aspect(1)
            ax1[0,1].set_title(r'$\langle e_{2} \rangle$', fontsize=22)
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
            ax1[1,0].set_ylabel(r'$\langle e_{1} \rangle$')
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
            ax1[1,1].set_ylabel(r'$\langle e_{2} \rangle$')
            ax1[1,1].set_xticks([])

            plt.legend(fontsize='large')
            plt.savefig('mdet_shear_variations_focal_plane_stacked_'+name+'.pdf')
            return None


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
    data = stack_CCDs(ccdres, 'all', x_side, y_side)
    if jk:
        return data[0], data[1]
    else:
        return None

# def main(argv):

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     print('mpi', rank, size)

#     just_plot = False
#     work_mdet = '/global/cscratch1/sd/myamamot/metadetect'
#     work = '/global/cscratch1/sd/myamamot'
#     ccd_x_min = 48
#     ccd_x_max = 2000
#     ccd_y_min = 48
#     ccd_y_max = 4048
#     piece_side = 32
#     x_side = int(np.ceil((ccd_x_max - ccd_x_min)/piece_side))
#     y_side = int(np.ceil((ccd_y_max - ccd_y_min)/piece_side))
#     num_ccd = 62

#     # NEED TO WRITE THE CODE TO BE ABLE TO RUN FROM BOTH MASTER FLAT AND INDIVIDUAL FILES. 
#     mdet_f = open('/global/cscratch1/sd/myamamot/metadetect/mdet_files.txt', 'r')
#     mdet_fs = mdet_f.read().split('\n')[:-1]
#     mdet_filenames = [fname.split('/')[-1] for fname in mdet_fs]
#     tilenames = [d.split('_')[0] for d in mdet_filenames]

#     if not just_plot:
        
#         # Obtain file, tile, band information from information file queried from desoper. 
#         coadd_info = fio.read(os.path.join(work, 'pizza-slice/pizza-cutter-coadds-info.fits'))
#         coadd_files = {t: [] for t in tilenames}
#         bands = {t: [] for t in tilenames}
#         for coadd in coadd_info:
#             tname = coadd['FILENAME'].split('_')[0]
#             fname = coadd['FILENAME'] + coadd['COMPRESSION']
#             bandname = coadd['FILENAME'].split('_')[2]
#             if tname in list(coadd_files.keys()):
#                 coadd_files[tname].append(fname)
#                 bands[tname].append(bandname)

#         # Accumulate raw sums of shear and number of objects in each bin for each tile and save as a pickle file. 
#         # When not using MPI, you can use for-loops (for t in tilenames)
#         split_tilenames = np.array_split(tilenames, size)
#         for t in tqdm(split_tilenames[rank]):
#             ccdres = {}
#             d = fio.read(os.path.join(work_mdet, mdet_filenames[np.where(np.in1d(tilenames, t))[0][0]]))
#             msk = ((d['flags']==0) & (d['mask_flags']==0) & (d['mdet_s2n']>10) & (d['mdet_s2n']<100) & (d['mfrac']<0.02) & (d['mdet_T_ratio']>0.5) & (d['mdet_T']<1.2))
#             ccdres = spatial_variations(ccdres, d[msk], coadd_files[t], ccd_x_min, ccd_y_min, x_side, y_side, piece_side, t, bands[t])
#             with open('/global/cscratch1/sd/myamamot/metadetect/mdet_shear_focal_plane_'+t+'.pickle', 'wb') as raw:
#                 pickle.dump(ccdres, raw, protocol=pickle.HIGHEST_PROTOCOL)
#     else:
#         print('Plotting...')

#         # Add raw sums for all the tiles from individual tile file. 
#         ccdres_all = {}
#         for t in tqdm(tilenames):
#             with open('/global/cscratch1/sd/myamamot/metadetect/mdet_shear_focal_plane_'+t+'.pickle', 'rb') as handle:
#                 ccdres = pickle.load(handle)
#             ccdres_all = _accum_shear_from_file(ccdres_all, ccdres, x_side, y_side)
#         with open('/global/cscratch1/sd/myamamot/metadetect/mdet_shear_focal_plane_all.pickle', 'wb') as raw:
#             pickle.dump(ccdres_all, raw, protocol=pickle.HIGHEST_PROTOCOL)
#         # plot for all the CCDs. 
#         plot_shear_vaiations_ccd(x_side, y_side, ccdres_all, num_ccd, jk=False)
        
#         # Compute jackknife error estimate. 
#         jk_sample = len(tilenames)
#         jk_x_g1 = []
#         jk_y_g1 = []
#         jk_x_g2 = []
#         jk_y_g2 = []
#         for i in tqdm(range(len(tilenames))):
#             x_g1 = []
#             y_g1 = []
#             x_g2 = []
#             y_g2 = []
#             for j,t in enumerate(tilenames):
#                 if i == j:
#                     continue
#                 with open('/global/cscratch1/sd/myamamot/metadetect/mdet_shear_focal_plane_'+t+'.pickle', 'rb') as handle:
#                     ccdres = pickle.load(handle)
#                 x_data, y_data = plot_shear_vaiations_ccd(x_side, y_side, ccdres, num_ccd, jk=True)
#                 x_g1.append(x_data[0])
#                 y_g1.append(y_data[0])
#                 x_g2.append(x_data[1])
#                 y_g2.append(y_data[1])

#             jk_x_g1.append(np.sum(x_g1, axis=0)/(jk_sample-1))
#             jk_y_g1.append(np.sum(y_g1, axis=0)/(jk_sample-1))
#             jk_x_g2.append(np.sum(x_g2, axis=0)/(jk_sample-1))
#             jk_y_g2.append(np.sum(y_g2, axis=0)/(jk_sample-1))
#         jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2 = _compute_jackknife_cov(jk_x_g1, jk_y_g1, jk_x_g2, jk_y_g2, len(tilenames))
#         print('jackknife error estimate', jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2)

#         with open('/global/cscratch1/sd/myamamot/metadetect/mdet_shear_focal_plane_all.pickle', 'rb') as handle:
#             ccdres = pickle.load(handle)
#             plot_shear_vaiations_ccd(x_side, y_side, ccdres, num_ccd, jk=False, jc=[jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2])
#     comm.Barrier()

    
# if __name__ == "__main__":
#     main(sys.argv)


just_plot = False
work_mdet = '/global/cscratch1/sd/myamamot/metadetect'
work = '/global/cscratch1/sd/myamamot'
ccd_x_min = 48
ccd_x_max = 2000
ccd_y_min = 48
ccd_y_max = 4048
piece_side = 32
x_side = int(np.ceil((ccd_x_max - ccd_x_min)/piece_side))
y_side = int(np.ceil((ccd_y_max - ccd_y_min)/piece_side))
num_ccd = 62

# NEED TO WRITE THE CODE TO BE ABLE TO RUN FROM BOTH MASTER FLAT AND INDIVIDUAL FILES. 
mdet_f = open('/global/cscratch1/sd/myamamot/metadetect/mdet_files.txt', 'r')
mdet_fs = mdet_f.read().split('\n')[:-1]
mdet_filenames = [fname.split('/')[-1] for fname in mdet_fs]
tilenames = [d.split('_')[0] for d in mdet_filenames]

if not just_plot:
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
tile_count = 0
num_tiles = len(tilenames)
while tile_count < num_tiles:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('mpi', rank, size)

    ccdres = {}
    if tile_count+rank < num_tiles:
        t = tilenames[rank]
        # d = fio.read(os.path.join(work_mdet, mdet_filenames[np.where(np.in1d(tilenames, t))[0][0]]))
        # msk = ((d['flags']==0) & (d['mask_flags']==0) & (d['mdet_s2n']>10) & (d['mdet_s2n']<100) & (d['mfrac']<0.02) & (d['mdet_T_ratio']>0.5) & (d['mdet_T']<1.2))
        # ccdres = spatial_variations(ccdres, d[msk], coadd_files[t], ccd_x_min, ccd_y_min, x_side, y_side, piece_side, t, bands[t])
        # with open('/global/cscratch1/sd/myamamot/metadetect/mdet_shear_focal_plane_'+t+'.pickle', 'wb') as raw:
        #     pickle.dump(ccdres, raw, protocol=pickle.HIGHEST_PROTOCOL)
        tile_count += size
        comm.bcast(tile_count, root=0)
        comm.Barrier()