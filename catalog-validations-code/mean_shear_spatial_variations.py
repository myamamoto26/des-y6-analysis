from statistics import mean
import fitsio as fio
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from esutil import stat
import esutil as eu
from tqdm import tqdm
import json
# from joblib import Parallel, delayed
import time
import pickle
from skimage.measure import block_reduce
import drawDECam.drawDECam as dDECam
import matplotlib
import math
from des_y6utils import mdet
import galsim

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
                    np.add.at(ccdres_all[ccdnum][cname], (rows, cols), ccdres[ccdnum][cname][rows, cols])
                    np.add.at(ccdres_all[ccdnum]["num_"+cname], (rows, cols), ccdres[ccdnum]["num_"+cname][rows, cols])
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

def _accum_shear_for_jk(ccdres_all, ccdres):
    
    cnames = ['g1', 'g2', 'g1p', 'g1m', 'g2p', 'g2m']
    for cname in cnames:
        rows,cols = np.where(~np.isnan(ccdres[cname]))
        np.negative.at(ccdres_all[cname], (rows, cols), ccdres[cname][rows, cols])
        np.negative.at(ccdres_all["num_"+cname], (rows, cols), ccdres["num_"+cname][rows, cols])
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

def comb_rows(d, nbin):
    
    ncol = d['g1'].shape[1]
    binds = np.array_split(np.arange(ncol), nbin) 
    cnames = ['g1', 'g2', 'g1p', 'g1m', 'g2p', 'g2m']
    bin_shear = {}
    for cname in cnames:
        if cname not in bin_shear:
            bin_shear[cname] = np.zeros(nbin)
            bin_shear["num_"+cname] = np.zeros(nbin)
        for i,bins in enumerate(binds):
            bin_shear[cname][i] = np.nansum(d[cname][:,bins])
            bin_shear["num_"+cname][i] = np.nansum(d["num_"+cname][:,bins])
            
    g1, g2 = _compute_g1_g2_per_ccd(bin_shear)  
    return g1, g2

def comb_cols(d, nbin):
    
    nrow = d['g1'].shape[0]
    binds = np.array_split(np.arange(nrow), nbin) 
    cnames = ['g1', 'g2', 'g1p', 'g1m', 'g2p', 'g2m']
    bin_shear = {}
    for cname in cnames:
        if cname not in bin_shear:
            bin_shear[cname] = np.zeros(nbin)
            bin_shear["num_"+cname] = np.zeros(nbin)
        for i,bins in enumerate(binds):
            bin_shear[cname][i] = np.nansum(d[cname][bins,:])
            bin_shear["num_"+cname][i] = np.nansum(d["num_"+cname][bins,:])
    
    g1, g2 = _compute_g1_g2_per_ccd(bin_shear)  
    return g1, g2

def _compute_jackknife_cov(jk_x_g1, jk_y_g1, jk_x_g2, jk_y_g2, N):

    # Make zero entries nan values.
    msk = np.where(~jk_x_g1.any(axis=1))[0]
    jk_x_g1 = np.delete(jk_x_g1, msk, axis=0)
    msk = np.where(~jk_y_g1.any(axis=1))[0]
    jk_y_g1 = np.delete(jk_y_g1, msk, axis=0)
    msk = np.where(~jk_x_g2.any(axis=1))[0]
    jk_x_g2 = np.delete(jk_x_g2, msk, axis=0)
    msk = np.where(~jk_y_g2.any(axis=1))[0]
    jk_y_g2 = np.delete(jk_y_g2, msk, axis=0)

    # compute jackknife average. 
    jk_x_g1_ave = np.mean(jk_x_g1, axis=0)
    jk_y_g1_ave = np.mean(jk_y_g1, axis=0)
    jk_x_g2_ave = np.mean(jk_x_g2, axis=0)
    jk_y_g2_ave = np.mean(jk_y_g2, axis=0)

    x_cov_g1 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_x_g1 - jk_x_g1_ave)**2, axis=0))
    y_cov_g1 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_y_g1 - jk_y_g1_ave)**2, axis=0))
    x_cov_g2 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_x_g2 - jk_x_g2_ave)**2, axis=0))
    y_cov_g2 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_y_g2 - jk_y_g2_ave)**2, axis=0))

    return x_cov_g1, y_cov_g1, x_cov_g2, y_cov_g2

def _categorize_obj_in_ccd(piece_side, nx, ny, ccd_x_min, ccd_y_min, x, y, msk_obj):

    """Computes which 32x32 cell the objects are in."""

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

def find_objects_in_ccd_and_sum_shears(ccdres, mdet_obj, coadd_files, ccd_x_min, ccd_y_min, x_side, y_side, piece_side, mdet_mom):

    """
    Computes x,y coordinates in single-epoch image frame from RA,DEC in metadetection catalogs, and sums up the raw shear in each cell for each CCD. 

    Parameters
    ----------
    ccdres: a dictionary that contains keys for all CCDs and sums of raw shear
    mdet_obj: a metadetection catalog
    coadd_files: a list of filenames of pizza-cutter meds files
    ccd_x_min: minimum x-value for CCD coordinates
    ccd_y_min: minimum y-value for CCD coordinates
    x_side: the number of cells in a row
    y_side: the number of cells in a column
    piece_side: the size of each cell
    bands: a list of bandnames of pizza-cutter meds files
    mdet_mom: which estimator to use
    """

    # How this function works: Collect info (id, ra, dec, CCD coord, mean property values), save it, and plot later. 
    for pizza_f in coadd_files:
        if pizza_f.split('_')[2] == 'r': # just do r-band for now. 
            coadd = fio.FITS(os.path.join('/global/cfs/cdirs/des/myamamot/pizza-slice/data', pizza_f))
        else:
            continue
        try:
            epochs = coadd['epochs_info'].read()
            image_info = coadd['image_info'].read()
        except OSError:
            print('Corrupt file.?', pizza_f)
            raise OSError

        # For each single-epoch image to create the coadd tile (10,000x10,000), find the WCS that can convert sky coordinates in coadd slices that used the single-epoch image to the image coordinates in single-epoch image, find objects in the slices
        # image_info['image_id']: single-epoch image ID. 
        # epochs_info['image_id']: single-epoch image ID. 
        # epochs_info['id']: coadd slice ID (200x200). 
        # mdet['slice_id']: coadd slice ID where objects are detected (200x200). 

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
            mdet_step = mdet_obj["mdet_step"][msk_obj]

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
        
            ccdres = _accum_shear(ccdres, ccdnum, "g1", "noshear", mdet_step, xind, yind, mdet_obj[mdet_mom+"_g_1"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g2", "noshear", mdet_step, xind, yind, mdet_obj[mdet_mom+"_g_2"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g1p", "1p", mdet_step, xind, yind, mdet_obj[mdet_mom+"_g_1"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g1m", "1m", mdet_step, xind, yind, mdet_obj[mdet_mom+"_g_1"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g2p", "2p", mdet_step, xind, yind, mdet_obj[mdet_mom+"_g_2"][msk_obj], x_side, y_side)
            ccdres = _accum_shear(ccdres, ccdnum, "g2m", "2m", mdet_step, xind, yind, mdet_obj[mdet_mom+"_g_2"][msk_obj], x_side, y_side)

    return ccdres

def compute_shear_stack_CCDs(ccdres, x_side, y_side, out_path, stack_north_south=False, block=False):
    
    """
    Stack sums of raw shears per CCD into the sums of north/south CCDs and the sums of all CCDs.
    """
    shear = {}
    for direct in ['north', 'south']:
        cnames = ['g1', 'g2', 'g1p', 'g1m', 'g2p', 'g2m']
        shear_sum_all = {}
        
        if direct == 'north':
            ccd_list = np.arange(1,32)
            ccd_list = np.delete(ccd_list, -1)
        elif direct == 'south':
            ccd_list = np.arange(32,63)
            ccd_list = np.delete(ccd_list, -2)
            
        for ccdnum in list(ccd_list):
            for cname in cnames:
                if cname not in list(shear_sum_all):
                    shear_sum_all[cname] = np.zeros((y_side, x_side))
                    shear_sum_all["num_" + cname] = np.zeros((y_side, x_side))
                if cname in list(shear_sum_all):
                    rows,cols = np.where(~np.isnan(ccdres[ccdnum][cname]))
                    np.add.at(shear_sum_all[cname], (rows, cols), ccdres[ccdnum][cname][rows, cols])
                    np.add.at(shear_sum_all["num_"+cname], (rows, cols), ccdres[ccdnum]["num_"+cname][rows, cols])
                    
        if direct == 'north':
            shear['north'] = shear_sum_all
        elif direct == 'south':
            shear['south'] = shear_sum_all
        
        if not os.path.exists(os.path.join(out_path, 'mdet_shear_focal_plane_north_south.pickle')):
            with open(os.path.join(out_path, 'mdet_shear_focal_plane_north_south.pickle'), 'wb') as raw:
                pickle.dump(shear, raw, protocol=pickle.HIGHEST_PROTOCOL)

    if not stack_north_south:
        mean_north_g1, mean_north_g2  = _compute_g1_g2_per_ccd(shear['north'])
        mean_south_g1, mean_south_g2  = _compute_g1_g2_per_ccd(shear['south'])
        
        mean_north_g1 = np.rot90(mean_north_g1, 3)
        mean_north_g2 = np.rot90(mean_north_g2, 3)
        mean_south_g1 = np.rot90(mean_south_g1, 3)
        mean_south_g2 = np.rot90(mean_south_g2, 3)
        mean_g1 = [mean_north_g1, mean_south_g1]
        mean_g2 = [mean_north_g2, mean_south_g2]
        print(mean_g1, mean_g2)
    else:
        # Stack north and south but be careful of the directions of stacking.
        shear_stack = {}
        for cname in cnames:
            shear_stack[cname] = np.zeros((y_side, x_side))
            shear_stack["num_" + cname] = np.zeros((y_side, x_side))
            rows,cols = np.where(~np.isnan(shear['north'][cname]))
            np.add.at(shear_stack[cname], (rows, cols), shear['north'][cname][rows, cols])
            np.add.at(shear_stack["num_"+cname], (rows, cols), shear['north']["num_"+cname][rows, cols])
            
            shear_flip = np.flip(shear['south'][cname],0)
            shear_flip_num = shear['south']["num_"+cname]
            rows,cols = np.where(~np.isnan(shear_flip))
            np.add.at(shear_stack[cname], (rows, cols), shear_flip[rows, cols])
            np.add.at(shear_stack["num_"+cname], (rows, cols), shear_flip_num[rows, cols])
        
        if block: # trim cells around edges. 
            for cname in list(shear_stack):
                shear_stack[cname] = shear_stack[cname][2:-2, 2:-2]

            if not os.path.exists(os.path.join(out_path, 'mdet_shear_focal_plane_stacked.pickle')):
                with open(os.path.join(out_path, 'mdet_shear_focal_plane_stacked.pickle'), 'wb') as raw:
                    pickle.dump(shear_stack, raw, protocol=pickle.HIGHEST_PROTOCOL)
            return shear_stack
        else:
            g1, g2  = _compute_g1_g2_per_ccd(shear_stack)
            mean_g1 = np.rot90(g1, 3)
            mean_g2 = np.rot90(g2, 3)
            print(mean_g1, mean_g2)

def compute_mean_shear_variations(mdet_input_filepaths, mdet_tilename_filepath, pizza_coadd_info, shear_variations_path, mdet_mom, mdet_cuts, individual_tiles = False, make_per_ccd_files = False):

    """
    
    Parameters
    ----------
    mdet_input_filepaths: The input filepath for the metadetection catalog
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v2

    mdet_tilename_filepath: The input filepath for the metadetection catalog tilenames
    Example) /global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt

    pizza_coadd_info: The input filepath for the pizza-cutter coadd information
    Example) /global/cscratch1/sd/myamamot/pizza-slice/pizza-cutter-coadds-info.fits

    shear_variations_path: The folder where shear variations files are stored. 
    Example) /global/cscratch1/sd/myamamot/metadetect/shear_variations/wmom

    mdet_mom: which estimator to use

    individual_tiles: the argument whether you want to create pickle files that contain raw sums of shear in each CCD for each coadd tile. This will make # of tiles worth of files (Boolean)
    make_per_ccd_tiles: the argument whether you want to create pickle files per CCD. This will make # of CCDs worth of files and the tile information is contained in each file. This is necessary for computing jackknife errors. (Boolean)
    """

    ccd_x_min = 48
    ccd_x_max = 2000
    ccd_y_min = 48
    ccd_y_max = 4048
    cell_side = 32
    x_side = int(np.ceil((ccd_x_max - ccd_x_min)/cell_side))
    y_side = int(np.ceil((ccd_y_max - ccd_y_min)/cell_side))
    num_ccd = 62

    # for individual catalogs
    mdet_f = open(mdet_tilename_filepath, 'r')
    mdet_fs = mdet_f.read().split('\n')[:-1]
    mdet_filenames = [fname.split('/')[-1] for fname in mdet_fs]
    tilenames = [d.split('_')[0] for d in mdet_filenames]
    # for patch catalogs
    # mdet_fs = glob.glob(mdet_input_filepaths)
    # mdet_filenames = [fname.split('/')[-1] for fname in mdet_fs]
    # tilenames = [d.split('_')[0] for d in mdet_filenames]

    # Accumulate raw sums of shear and number of objects in each bin for each tile and save as a pickle file. 
    # When not using MPI, you can use for-loops (for t in tilenames)
    if individual_tiles:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print('mpi', rank, size)
        
        # Obtain file, tile, band information from information file queried from desoper. 
        coadd_info = fio.read(pizza_coadd_info)
        coadd_files = {t: [] for t in tilenames}
        for coadd in coadd_info:
            tname = coadd['FILENAME'].split('_')[0]
            fname = coadd['FILENAME']+'.fz'
            if tname in list(coadd_files.keys()):
                coadd_files[tname].append(fname)

        split_tilenames = np.array_split(tilenames, size)
        for t in tqdm(split_tilenames[rank]):
            ccdres = {}
            obj_num = 0
            if not os.path.exists(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_'+t+'.pickle')):
                try:
                    d = fio.read(os.path.join(mdet_input_filepaths, mdet_filenames[np.where(np.in1d(tilenames, t))[0][0]]))
                    msk = mdet.make_mdet_cuts(d, mdet_cuts)
                    d = d[msk]
                except:
                    print(t, " does not exist. skipping")
                    continue
                # msk = if additional cuts are necessary. 
                ccdres = find_objects_in_ccd_and_sum_shears(ccdres, d, coadd_files[t], ccd_x_min, ccd_y_min, x_side, y_side, cell_side, mdet_mom)
                for c in list(ccdres.keys()):
                    obj_num += np.sum(ccdres[c]['num_g1'])
                print('number of objects in this tile, ', obj_num)
                with open(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_'+t+'.pickle'), 'wb') as raw:
                    pickle.dump(ccdres, raw, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print('Already made this tile.', t)
        comm.Barrier()
    else:
        if make_per_ccd_files:
            print('Converting tiles to CCDs...')
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            print('mpi', rank, size)

            ccdres_all_ccd = {}
            split_tilenames = np.array_split(tilenames, size)
            for t in tqdm(split_tilenames[rank]):
                try:
                    with open(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_'+t+'.pickle'), 'rb') as handle:
                        ccdres = pickle.load(handle)
                        handle.close()
                    ccdres_all_ccd = _accum_shear_per_ccd(ccdres_all_ccd, ccdres, t)
                except:
                    print(t, 'this tile cannot be found.')
                    continue
            comm.Barrier()
            for c in list(ccdres_all_ccd):
                with open(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_ccd_'+str(c)+'.pickle'), 'wb') as raw:
                    pickle.dump(ccdres_all_ccd[c], raw, protocol=pickle.HIGHEST_PROTOCOL)

        # Add raw sums for all the tiles from individual tile file. 
        if not os.path.exists(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_all.pickle')):
            ccdres_all = {}
            for t in tqdm(tilenames):
                with open(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_'+t+'.pickle'), 'rb') as handle:
                    ccdres = pickle.load(handle)
                ccdres_all = _accum_shear_from_file(ccdres_all, ccdres, x_side, y_side)
            print(list(ccdres_all))
            with open(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_all.pickle'), 'wb') as raw:
                pickle.dump(ccdres_all, raw, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_all.pickle'), 'rb') as raw:
                ccdres_all = pickle.load(raw) 
        
        # Saves accumulated pickle file for north/south and all, and then compute the mean shear for x-stack and y-stack.
        bin_num = 20
        d_shear = compute_shear_stack_CCDs(ccdres_all, x_side, y_side, shear_variations_path, stack_north_south=True, block=True)
        mean_row_g1, mean_row_g2 = comb_rows(d_shear, bin_num)
        mean_col_g1, mean_col_g2 = comb_cols(d_shear, bin_num)

        # Compute jackknife error estimate. 
        print('Computing jackknife error')
        jk_sample = len(tilenames)

        all_ccd_shear = {}
        cnames = ['g1', 'g2', 'g1p', 'g1m', 'g2p', 'g2m']
        for c in tqdm(range(1, num_ccd+1)):
            if not os.path.exists(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_ccd_'+str(c)+'.pickle')):
                print(c, 'ccd not found')
                continue

            with open(os.path.join(shear_variations_path, 'mdet_shear_focal_plane_ccd_'+str(c)+'.pickle', 'rb')) as handle:
                ccdres = pickle.load(handle)

            # Accumulate all the shears first. 
            ccdres_jk = {}
            ccdres_jk = _accum_shear_from_file(ccdres_jk, ccdres, x_side, y_side, per_ccd=True)
            for t in list(ccdres):
                res_jk = ccdres_jk.copy()
                res_jk = _accum_shear_for_jk(res_jk, ccdres[t])
                if t not in list(all_ccd_shear):
                    all_ccd_shear[t] = {}
                    for cname in cnames:
                        all_ccd_shear[t][cname] = np.zeros((y_side, x_side))
                        all_ccd_shear[t]["num_" + cname] = np.zeros((y_side, x_side))
                        rows,cols = np.where(~np.isnan(res_jk[cname]))
                        np.add.at(all_ccd_shear[t][cname], (rows, cols), res_jk[cname][rows, cols])
                        np.add.at(all_ccd_shear[t]["num_"+cname], (rows, cols), res_jk["num_"+cname][rows, cols])
                else:
                    for cname in cnames:
                        rows,cols = np.where(~np.isnan(res_jk[cname]))
                        np.add.at(all_ccd_shear[t][cname], (rows, cols), res_jk[cname][rows, cols])
                        np.add.at(all_ccd_shear[t]["num_"+cname], (rows, cols), res_jk["num_"+cname][rows, cols])
        
        jk_x_g1 = np.zeros((jk_sample, bin_num))
        jk_y_g1 = np.zeros((jk_sample, bin_num))
        jk_x_g2 = np.zeros((jk_sample, bin_num))
        jk_y_g2 = np.zeros((jk_sample, bin_num))
        unused_tile = 0
        for j,t in tqdm(enumerate(tilenames)):
            if t not in list(all_ccd_shear):
                unused_tile += 1 
            mean_row_g1_jk, mean_row_g2_jk = comb_rows(all_ccd_shear[t], bin_num)
            mean_col_g1_jk, mean_col_g2_jk = comb_cols(all_ccd_shear[t], bin_num)

            jk_x_g1[j, :] = mean_row_g1_jk
            jk_y_g1[j, :] = mean_col_g1_jk
            jk_x_g2[j, :] = mean_row_g2_jk
            jk_y_g2[j, :] = mean_col_g2_jk

        jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2 = _compute_jackknife_cov(jk_x_g1, jk_y_g1, jk_x_g2, jk_y_g2, len(tilenames))
        print('jackknife error estimate', jc_x_g1, jc_y_g1, jc_x_g2, jc_y_g2)
        print(mean_row_g1)
        jk_dict = {'x_g1': mean_row_g1, 'y_g1': mean_col_g1, 'x_g2': mean_row_g2, 'y_g2': mean_col_g2, 
                    'jc_x_g1': jc_x_g1, 'jc_y_g1': jc_y_g1, 'jc_x_g2': jc_x_g2, 'jc_y_g2': jc_y_g2}
        with open(os.path.join(shear_variations_path, 'mean_shear_jk_cov.pickle'), 'wb') as handle:
            pickle.dump(jk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(argv):

    mdet_input_filepaths = sys.argv[1]
    mdet_tilename_filepath = sys.argv[2]
    pizza_coadd_info = sys.argv[3]
    shear_variations_path = sys.argv[4]
    shear_per_tile = False
    shear_per_ccd = True
    mdet_mom = sys.argv[7]
    mdet_cuts = int(sys.argv[8])

    compute_mean_shear_variations(mdet_input_filepaths,mdet_tilename_filepath, pizza_coadd_info, shear_variations_path, mdet_mom, mdet_cuts, individual_tiles=shear_per_tile, make_per_ccd_files=shear_per_ccd)
    
if __name__ == "__main__":
    main(sys.argv)