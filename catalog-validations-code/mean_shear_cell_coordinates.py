from statistics import mean
import fitsio as fio
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from esutil import stat
import esutil as eu
from tqdm import tqdm
import json
import time
import pickle
from des_y6utils import mdet
import h5py as h5
from pizza_cutter.slice_utils.locate import build_slice_locations

def get_shear_weights(d, weight_type, wgt_filepath, mask=None):
    """
    Get shear weights. The shear weight could be computed from
    objects S/N and size ratio, shape error or uniformly.
    """
    import numpy as np
    import pickle

    def _assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
        from math import log10
        # return x and y indices of data (x,y) on a log-spaced grid that runs from
        # [xy]min to [xy]max in [xy]steps

        logstepx = log10(xmax / xmin) / xsteps
        logstepy = log10(ymax / ymin) / ysteps

        indexx = (np.log10(x / xmin) / logstepx).astype(int)
        indexy = (np.log10(y / ymin) / logstepy).astype(int)

        indexx = np.maximum(indexx, 0)
        indexx = np.minimum(indexx, xsteps - 1)
        indexy = np.maximum(indexy, 0)
        indexy = np.minimum(indexy, ysteps - 1)

        return indexx, indexy

    def _find_shear_weight(d, wgt_dict, snmin, snmax, sizemin, sizemax, steps, mask=None):

        if wgt_dict is None:
            weights = np.ones(len(d))
            return weights

        shear_wgt = wgt_dict['weight']
        smoothing = True
        if smoothing:
            from scipy.ndimage import gaussian_filter
            smooth_response = gaussian_filter(wgt_dict['response'], sigma=2.0)
            shear_wgt = (smooth_response/wgt_dict['meanes'])**2
        if mask is None:
            indexx, indexy = _assign_loggrid(
                np.array(d['gauss_s2n']), np.array(d['gauss_T_ratio']), snmin, snmax, steps, sizemin, sizemax, steps)
        else:
            indexx, indexy = _assign_loggrid(
                np.array(d['gauss_s2n'])[mask], np.array(d['gauss_T_ratio'])[mask], snmin, snmax, steps, sizemin, sizemax, steps)
        weights = np.array([shear_wgt[x, y] for x, y in zip(indexx, indexy)])

        return weights

    # import pdb ; pdb.set_trace()
    if weight_type == 's2n_sizer':
        # pickle file that defines w(S/N, size)
        with open(wgt_filepath, 'rb') as handle:
            wgt_dict = pickle.load(handle)
        # TO-DO: make snmin, snmax, sizemin, sizemax available in config file.
        shear_wgt = _find_shear_weight(d, wgt_dict, 10, 1000, 0.5, 5.0, 20, mask=mask)
    elif weight_type == 'shape_err':
        if mask is None:
            shear_wgt = 1 / (0.22**2 + 0.5 * (np.array(d['gauss_g_cov_1_1']) + np.array(d['gauss_g_cov_2_2'])))
        else:
            shear_wgt = 1 / (0.22**2 + 0.5 * (np.array(d['gauss_g_cov_1_1'])[mask] + np.array(d['gauss_g_cov_2_2'])[mask]))
    elif weight_type == 'uniform':
        shear_wgt = np.ones(len(d['ra']))

    shear_wgt[np.isnan(shear_wgt)] = 0.

    return shear_wgt

def _process_arr(x, central_size, buffer_size, tile_size):
    slice_x = np.zeros_like(x)
    mskl = x < buffer_size
    if np.any(mskl):
        slice_x[mskl] = x[mskl]
    mskh = x > tile_size - buffer_size
    if np.any(mskh):
        slice_x[mskh] = buffer_size*2 + central_size + x[mskh] - tile_size
    msk = ~(mskl | mskh)
    if np.any(msk):
        slice_x[msk] = np.mod(x[msk] - buffer_size, central_size) + buffer_size
    return slice_x

def coaddpos2slicepos(x, y, central_size=100, buffer_size=50, tile_size=10000): 
    return _process_arr(x, central_size, buffer_size, tile_size), _process_arr(y, central_size, buffer_size, tile_size)


def read_mdet_h5(d, d_1p, d_1m, d_2p, d_2m, keys, weight_scheme, wgt_path, cell=False):

    gal_data = {'noshear': {}, '1p': {}, '1m': {}, '2p': {}, '2m': {}}
    shear_data = {'noshear': d, '1p': d_1p, '1m': d_1m, '2p': d_2p, '2m': d_2m}
    for step in shear_data.keys():
        d_ = shear_data[step]

        weight = get_shear_weights(d_, weight_scheme, wgt_path)
        gal_data[step]['w'] = weight
        for k in keys:
            gal_data[step][k] = d_[k][:]
        
        if cell:
            mdet_sx, mdet_sy = coaddpos2slicepos(gal_data[step]['x'], gal_data[step]['y'], central_size=100, buffer_size=50)
            gal_data[step]['x'] = mdet_sx
            gal_data[step]['y'] = mdet_sy
    
    return gal_data


def load_mdet_h5(datafile, keys, wgt_type, wgt_filepath, response=False, subtract_mean_shear=False, mask=None, color_split=False):

    def _wmean(q,w):
        return np.sum(q*w)/np.sum(w)
    def _make_cut(d, mask_map):
        hmap = healsparse.HealSparseMap.read(mask_map)
        in_footprint = hmap.get_values_pos(np.array(d['ra']), np.array(d['dec']), valid_mask=True)
        return in_footprint
    def _make_color_cut(d, cmin, cmax):
        gmi =  mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_g"]), 0) - mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_i"]), 2)
        msk = ((gmi > cmin) & (gmi <= cmax))
        return msk
    
    import h5py as h5
    f = h5.File(datafile, 'r')
    d = f.get('/mdet/noshear')
    nrows = len(np.array( d['ra'] ))
    formats = []
    for key in keys:
        if key == 'tilename':
            formats.append('S12')
        else:
            formats.append('f4')
    data = np.recarray(shape=(nrows,), formats=formats, names=keys)
    for key in keys:  
        if key == 'w':
            data['w'] = get_shear_weights(d, wgt_type, wgt_filepath)
        elif key in ('g1', 'g2'):
            data[key] = np.array(d['gauss_'+key[0]+'_'+key[1]])
        else:
            data[key] = np.array(d[key])
    print('made recarray with hdf5 file')
    if mask is not None:
        print('additional mask included')
        msk_footprint = _make_cut(data, mask)
        data = data[msk_footprint]
    if color_split:
        print('splitting color samples')
        # color splits: blue-[-2.00, 0.76], mid-[0.76, 1.49], red-[1.49, 4.00]
        gmi =  mdet._compute_asinh_mags(data["pgauss_band_flux_g"], 0) - mdet._compute_asinh_mags(data["pgauss_band_flux_i"], 2)
        dmin = 1.49
        dmax = 4.00
        msk_color = ((gmi > dmin) & (gmi <= dmax))
        data = data[msk_color]
    
    # response correction
    if response:
        d_2p = f.get('/mdet/2p')
        d_1p = f.get('/mdet/1p')
        d_2m = f.get('/mdet/2m')
        d_1m = f.get('/mdet/1m')
        if mask is None:
            # compute response with weights
            if color_split:
                print('color split included')
                msk_1p = _make_color_cut(d_1p, dmin, dmax)
                g1p = _wmean(np.array(d_1p["gauss_g_1"])[msk_1p], get_shear_weights(d_1p, wgt_type, wgt_filepath, mask=msk_1p))
                msk_1m = _make_color_cut(d_1m, dmin, dmax)
                g1m = _wmean(np.array(d_1m["gauss_g_1"])[msk_1m], get_shear_weights(d_1m, wgt_type, wgt_filepath, mask=msk_1m))
                msk_2p = _make_color_cut(d_2p, dmin, dmax)
                g2p = _wmean(np.array(d_2p["gauss_g_2"])[msk_2p], get_shear_weights(d_2p, wgt_type, wgt_filepath, mask=msk_2p))
                msk_2m = _make_color_cut(d_2m, dmin, dmax)
                g2m = _wmean(np.array(d_2m["gauss_g_2"])[msk_2m], get_shear_weights(d_2m, wgt_type, wgt_filepath, mask=msk_2m))
            else:
                g1p = _wmean(np.array(d_1p["gauss_g_1"]), get_shear_weights(d_1p, wgt_type, wgt_filepath))
                g1m = _wmean(np.array(d_1m["gauss_g_1"]), get_shear_weights(d_1m, wgt_type, wgt_filepath))
                g2p = _wmean(np.array(d_2p["gauss_g_2"]), get_shear_weights(d_2p, wgt_type, wgt_filepath))
                g2m = _wmean(np.array(d_2m["gauss_g_2"]), get_shear_weights(d_2m, wgt_type, wgt_filepath))
        else:
            print('additional mask included')
            msk_1p = _make_cut(d_1p, mask)
            g1p = _wmean(np.array(d_1p["gauss_g_1"])[msk_1p], get_shear_weights(d_1p, wgt_type, wgt_filepath, mask=msk_1p))
            msk_1m = _make_cut(d_1m, mask)
            g1m = _wmean(np.array(d_1m["gauss_g_1"])[msk_1m], get_shear_weights(d_1m, wgt_type, wgt_filepath, mask=msk_1m))
            msk_2p = _make_cut(d_2p, mask)
            g2p = _wmean(np.array(d_2p["gauss_g_2"])[msk_2p], get_shear_weights(d_2p, wgt_type, wgt_filepath, mask=msk_2p))
            msk_2m = _make_cut(d_2m, mask)
            g2m = _wmean(np.array(d_2m["gauss_g_2"])[msk_2m], get_shear_weights(d_2m, wgt_type, wgt_filepath, mask=msk_2m))

        R11 = (g1p - g1m) / 0.02
        R22 = (g2p - g2m) / 0.02
        R = (R11 + R22)/2.
        print(R)
        data['g1'] /= R
        data['g2'] /= R

    # mean shear subtraction
    if subtract_mean_shear:
        if color_split:
            data['g1'] -= 0.00020496
            data['g2'] -= -0.00003086
        else:
            g1 = _wmean(data["g1"],data["w"])
            g2 = _wmean(data["g2"],data["w"])
            print('mean g1 g2 =(%1.8f,%1.8f)'%(g1, g2))          
            data['g1'] -= g1
            data['g2'] -= g2   

    return data
            

def annulus(p_, n_, R0_):
    rp_ = np.sqrt(p_/n_) * R0_
    return rp_

def accum_shear(res, binn, dat, r0, rp, step):

    r = np.sqrt(dat[step]['x']**2 + dat[step]['y']**2)
    msk = ((r > r0) & (r <= rp))

    g1_masked = dat[step]['gauss_g_1'][msk]*dat[step]['w'][msk] 
    g2_masked = dat[step]['gauss_g_2'][msk]*dat[step]['w'][msk]

    np.add.at(
        res[step], 
        (binn, 0), 
        np.sum(g1_masked)
    )

    np.add.at(
        res[step], 
        (binn, 1), 
        np.sum(g2_masked)
    )

    np.add.at(
        res["num_"+step], 
        (binn, 0), 
        np.sum(dat[step]['w'][msk])
    )

    np.add.at(
        res["num_"+step], 
        (binn, 1), 
        np.sum(dat[step]['w'][msk])
    )

    return res

def _compute_g1_g2(res_, binnum, method='bin1d'):
    
    if method == 'bin1d':
        sample_mean = np.zeros(binnum, dtype=[('e1', 'f8'), ('e2', 'f8')])
        for b in range(binnum):
            g1 = res_['noshear'][b, 0]/res_['num_noshear'][b, 0]
            g2 = res_['noshear'][b, 1]/res_['num_noshear'][b, 1]
            g1p = res_['1p'][b, 0]/res_['num_1p'][b, 0]
            g1m = res_['1m'][b, 0]/res_['num_1m'][b, 0]
            g2p = res_['2p'][b, 1]/res_['num_2p'][b, 1]
            g2m = res_['2m'][b, 1]/res_['num_2m'][b, 1]

            R11 = (g1p - g1m)/2/0.01
            R22 = (g2p - g2m)/2/0.01
            R = (R11+R22)/2.
            sample_mean[b]['e1'] = g1/R
            sample_mean[b]['e2'] = g2/R
    elif method == 'grid':
        sample_mean = {'e1': np.zeros_like(res_['e1']['noshear']), 'e2': np.zeros_like(res_['e1']['noshear'])}
        g1 = res_['e1']['noshear']/res_['e1']['num_noshear']
        g2 = res_['e2']['noshear']/res_['e2']['num_noshear']
        g1p = res_['e1']['1p']/res_['e1']['num_1p']
        g1m = res_['e1']['1m']/res_['e1']['num_1m']
        g2p = res_['e2']['2p']/res_['e2']['num_2p']
        g2m = res_['e2']['2m']/res_['e2']['num_2m']

        R11 = (g1p - g1m)/2/0.01
        R22 = (g2p - g2m)/2/0.01
        R = (R11+R22)/2. 

        sample_mean['e1'][:,:] = g1/R
        sample_mean['e2'][:,:] = g2/R

    return sample_mean

def _compute_shear_per_jksample_radial(gdata, rad_bin, res_jk, ith_tilename, tilenames, binnum):

    # Compute mean shear for each jackknife sample. 
    # For each jackknife sample, you leave one tile out, sums the shears in N-1 tiles, and compute the mean. 
    
    for i,step in enumerate(['noshear', '1p', '1m', '2p', '2m']):
        d_ = gdata[step]
        msk_tile = (d_['patch_num'] == ith_tilename)
        for bin in range(binnum):
            r = np.sqrt(d_['x']**2 + d_['y']**2)
            if bin == 0:
                r0 = 0
            else:
                r0 = rad_bin[bin-1]
            msk_bin = ((r > r0) & (r <= rad_bin[bin]))

            weight = d_['w']
            msk = (msk_tile & msk_bin)
            g1_masked = d_['gauss_g_1'][msk]*weight[msk] 
            g2_masked = d_['gauss_g_2'][msk]*weight[msk]

            np.add.at(
                res_jk[step], 
                (bin, 0), 
                np.sum(g1_masked)
            )
            np.add.at(
                res_jk[step], 
                (bin, 1), 
                np.sum(g2_masked)
            )
            np.add.at(
                res_jk["num_" + step], 
                (bin, 0), 
                np.sum(weight[msk])
            )
            np.add.at(
                res_jk["num_" + step], 
                (bin, 1), 
                np.sum(weight[msk])
            )

    jk_sample_mean = _compute_g1_g2(res_jk, binnum)

    return jk_sample_mean

def _compute_shear_per_jksample(dat, tilename, x_side, y_side, xind, yind):

    res_ = {'e1': {}, 'e2': {}}
    for step in ['noshear', '1p', '1m', '2p', '2m']:
        if step not in list(res_['e1']):
            res_['e1'][step] = np.zeros((y_side, x_side))
            res_['e1']["num_" + step] = np.zeros((y_side, x_side))
            res_['e2'][step] = np.zeros((y_side, x_side))
            res_['e2']["num_" + step] = np.zeros((y_side, x_side))
        
        msk_tile = ~(dat[step]['patch_num'] == tilename)
        g1 = dat[step]['gauss_g_1'][msk_tile]
        g2 = dat[step]['gauss_g_2'][msk_tile]
        new_xind = xind[step][msk_tile]
        new_yind = yind[step][msk_tile]
        new_w = dat[step]['w'][msk_tile]

        np.add.at(
            res_['e1'][step], 
            (new_yind, new_xind), 
            g1*new_w,
            )
        np.add.at(
            res_['e1']["num_"+step], 
            (new_yind, new_xind), 
            new_w,
            )
        np.add.at(
            res_['e2'][step], 
            (new_yind, new_xind), 
            g2*new_w,
            )
        np.add.at(
            res_['e2']["num_"+step], 
            (new_yind, new_xind), 
            new_w,
            )

    jk_sample_mean = _compute_g1_g2(res_, 0, method='grid')

    return jk_sample_mean

def _compute_jackknife_error_estimate(stats, binnum, N, method='bin1d'):

    if method == 'bin1d':
        jk_cov = np.zeros((binnum, 2))
        for bin in range(binnum):

            jk_all_g1_ave = np.mean(stats[bin]['e1'])
            jk_all_g2_ave = np.mean(stats[bin]['e2'])

            cov_g1 = np.sqrt((N-1)/N)*np.sqrt(np.sum((stats[bin]['e1'] - jk_all_g1_ave)**2))
            cov_g2 = np.sqrt((N-1)/N)*np.sqrt(np.sum((stats[bin]['e2'] - jk_all_g2_ave)**2))

            jk_cov[bin, 0] = cov_g1
            jk_cov[bin, 1] = cov_g2
    elif method == 'grid':
        jk_cov = {'e1':{}, 'e2':{}}
        e1_ = np.stack([stats[i]['e1'] for i in range(N)])
        e2_ = np.stack([stats[i]['e2'] for i in range(N)])

        jk_all_g1_ave = np.mean(e1_, axis=0)
        jk_all_g2_ave = np.mean(e2_, axis=0)

        cov_g1 = np.sqrt((N-1)/N)*np.sqrt(np.sum((e1_ - jk_all_g1_ave)**2, axis=0))
        cov_g2 = np.sqrt((N-1)/N)*np.sqrt(np.sum((e2_ - jk_all_g2_ave)**2, axis=0))
        print(np.sum((e1_ - jk_all_g1_ave)**2, axis=0).shape)
        print(np.sum((e1_ - jk_all_g1_ave)**2, axis=0))
        print(cov_g1)

        jk_cov['e1'] = cov_g1
        jk_cov['e2'] = cov_g2

    return jk_cov

def _categorize_obj_in_ccd(cell_side, nx, ny, ccd_x_min, ccd_y_min, x, y):

    """Computes which cell the objects are in."""

    xind = np.floor((x-ccd_x_min)/cell_side).astype(int)
    yind = np.floor((y-ccd_y_min)/cell_side).astype(int)

    msk_cut = np.where(
        (xind >= 0)
        & (xind < nx)
        & (yind >= 0)
        & (yind < ny)
    )[0]
    print('objects cut ', len(x)-len(msk_cut))
    if len(msk_cut) == 0:
        return None

    xind = xind[msk_cut]
    yind = yind[msk_cut]

    return xind, yind, msk_cut

def _accum_shear_grid(ccdres, cname, step, xind, yind, g, wgt, x_side, y_side):

    if step not in list(ccdres[cname]):
        ccdres[cname][step] = np.zeros((y_side, x_side))
        ccdres[cname]["num_" + step] = np.zeros((y_side, x_side))
    
    # see https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html#numpy.ufunc.at
    np.add.at(
        ccdres[cname][step], 
        (yind, xind), 
        g*wgt,
    )
    np.add.at(
        ccdres[cname]["num_" + step], 
        (yind, xind), 
        wgt,
    )

    return ccdres

def accum_shear_tile(res_, dat, radbin, nbin):

    for iid in dat.keys():
        r = np.sqrt(dat[iid]['x']**2 + dat[iid]['y']**2)
        bind_ = np.ditigize(r, radbin, right=True)
        for step in ['noshear', '1p', '1m', '2p', '2m']:

            msk_step = (dat[iid]['mdet_step'] == step)
            g1 = dat[iid]['g1'][msk_step]
            g2 = dat[iid]['g2'][msk_step]
            w = dat[iid]['w'][msk_step]

            bind = bind_[msk_step]
            msk_cut = np.where((bind >= 0) & (bind < nbin))[0]

            np.add.at(
                res_[step], 
                (bind[msk_cut], 0), 
                g1[msk_cut]*w[msk_cut],
            )
            np.add.at(
                res_[step], 
                (bind[msk_cut], 1), 
                g2[msk_cut]*w[msk_cut],
            )
            np.add.at(
                res_["num_"+step], 
                (bind[msk_cut], 0), 
                w[msk_cut],
            )
            np.add.at(
                res_["num_"+step], 
                (bind[msk_cut], 1), 
                w[msk_cut],
            )
    
    return res_

def _get_cell_center(tiles):

    import galsim
    from pizza_cutter.slice_utils.locate import build_slice_locations
    rows, cols, srows, scols = build_slice_locations(central_size=100, buffer_size=50, image_width=10000)
    res_all = []
    for d in tqdm(tiles):
        head = {}
        d_ = np.zeros(len(rows), dtype=[('ra', 'f8'), ('dec', 'f8')])
        for nm in d.dtype.names:
            head[nm] = d[nm]
        wcs = galsim.FitsWCS(header=head)
        ra, dec = wcs.xyToradec(cols, rows, units="degrees")
        d_['ra'] = ra
        d_['dec'] = dec
        res_all.append(d_)

    res_all = np.concatenate(res_all)
    return res_all

def mean_shear_in_cell_grid(mdet_input_filepath, outpath, weight_scheme, wgt_path):

    # Read in cat. Convert coadd coordinates to cell coordinates. 
    print('reading in catalog...')
    f = h5.File(mdet_input_filepath, 'r')
    d = f.get('/mdet/noshear')
    d_2p = f.get('/mdet/2p')
    d_1p = f.get('/mdet/1p')
    d_2m = f.get('/mdet/2m')
    d_1m = f.get('/mdet/1m')
    keys = ['x', 'y', 'gauss_g_1', 'gauss_g_2', 'patch_num']
    gal_data = read_mdet_h5(d, d_1p, d_2p, d_1m, d_2m, keys, weight_scheme, wgt_path, cell=True)

    # Define grids first. 
    ccd_x_min = 25
    ccd_x_max = 175
    ccd_y_min = 25
    ccd_y_max = 175
    cell_side = 15
    x_side = int(np.ceil((ccd_x_max - ccd_x_min)/cell_side))
    y_side = int(np.ceil((ccd_y_max - ccd_y_min)/cell_side))

    # Allocate objects in grids.
    print('allocating objects in grid...')
    ccdres = {'e1': {}, 'e2': {}}
    xcellid = {}; ycellid = {}
    for step in tqdm(['noshear', '1p', '1m', '2p', '2m']):
        d_ = gal_data[step]
        xind, yind, msk_cut = _categorize_obj_in_ccd(cell_side, x_side, y_side, ccd_x_min, ccd_y_min, d_['x'], d_['y'])
        xcellid[step] = xind; ycellid[step] = yind
        for k in d_.keys():
            d_[k] = d_[k][msk_cut]
        ccdres = _accum_shear_grid(ccdres, 'e1', step, xind, yind, d_['gauss_g_1'], d_['w'], x_side, y_side)
        ccdres = _accum_shear_grid(ccdres, 'e2', step, xind, yind, d_['gauss_g_2'], d_['w'], x_side, y_side)

    # Compute shear response & get mean shear.
    mean_shear = _compute_g1_g2(ccdres, 0, method='grid')

    # Compute jackknife errors
    print('jackknife process in progress...')
    tilenames = np.unique(np.array(d['patch_num']))
    jk_sample = len(tilenames)
    jk_stats = {i: {'e1': np.zeros((y_side, x_side)), 'e2': np.zeros((y_side, x_side))} for i in range(jk_sample)}
    for i,ith in tqdm(enumerate(tilenames)):
        jk_sample_mean = _compute_shear_per_jksample(gal_data, ith, x_side, y_side, xcellid, ycellid)
        jk_stats[i]['e1'] = jk_sample_mean['e1']
        jk_stats[i]['e2'] = jk_sample_mean['e2']
    jk_cov = _compute_jackknife_error_estimate(jk_stats, 0, jk_sample, method='grid')
    mean_shear['e1_cov'] = jk_cov['e1']
    mean_shear['e2_cov'] = jk_cov['e2']

    with open(os.path.join(outpath, 'mean_shear_around_cell_coordinates_grid_side15.pickle'), 'wb') as raw:
        pickle.dump(mean_shear, raw, protocol=pickle.HIGHEST_PROTOCOL)

def mean_shear_in_coadd_grid(mdet_input_filepath, outpath, weight_scheme, wgt_path):

    # Read in cat. 
    print('reading in catalog...')
    f = h5.File(mdet_input_filepath, 'r')
    d = f.get('/mdet/noshear')
    d_2p = f.get('/mdet/2p')
    d_1p = f.get('/mdet/1p')
    d_2m = f.get('/mdet/2m')
    d_1m = f.get('/mdet/1m')
    keys = ['x', 'y', 'gauss_g_1', 'gauss_g_2', 'patch_num']
    gal_data = read_mdet_h5(d, d_1p, d_2p, d_1m, d_2m, keys, weight_scheme, wgt_path)

    # Define grids first. 
    ccd_x_min = 0
    ccd_x_max = 10000
    ccd_y_min = 0
    ccd_y_max = 10000
    cell_side = 250
    x_side = int(np.ceil((ccd_x_max - ccd_x_min)/cell_side))
    y_side = int(np.ceil((ccd_y_max - ccd_y_min)/cell_side))

    # Allocate objects in grids.
    print('allocating objects in grid...')
    ccdres = {'e1': {}, 'e2': {}}
    xcellid = {}; ycellid = {}
    for step in tqdm(['noshear', '1p', '1m', '2p', '2m']):
        d_ = gal_data[step]
        xind, yind, msk_cut = _categorize_obj_in_ccd(cell_side, x_side, y_side, ccd_x_min, ccd_y_min, d_['x'], d_['y'])
        xcellid[step] = xind; ycellid[step] = yind
        for k in d_.keys():
            d_[k] = d_[k][msk_cut]
        ccdres = _accum_shear_grid(ccdres, 'e1', step, xind, yind, d_['gauss_g_1'], d_['w'], x_side, y_side)
        ccdres = _accum_shear_grid(ccdres, 'e2', step, xind, yind, d_['gauss_g_2'], d_['w'], x_side, y_side)

    # Compute shear response & get mean shear.
    mean_shear = _compute_g1_g2(ccdres, 0, method='grid')

    # Compute jackknife errors
    print('jackknife process in progress...')
    tilenames = np.unique(np.array(d['patch_num']))
    jk_sample = len(tilenames)
    jk_stats = {i: {'e1': np.zeros((y_side, x_side)), 'e2': np.zeros((y_side, x_side))} for i in range(jk_sample)}
    for i,ith in tqdm(enumerate(tilenames)):
        jk_sample_mean = _compute_shear_per_jksample(gal_data, ith, x_side, y_side, xcellid, ycellid)
        jk_stats[i]['e1'] = jk_sample_mean['e1']
        jk_stats[i]['e2'] = jk_sample_mean['e2']
    jk_cov = _compute_jackknife_error_estimate(jk_stats, 0, jk_sample, method='grid')
    mean_shear['e1_cov'] = jk_cov['e1']
    mean_shear['e2_cov'] = jk_cov['e2']


    with open(os.path.join(outpath, 'mean_shear_around_coadd_coordinates_grid.pickle'), 'wb') as raw:
        pickle.dump(mean_shear, raw, protocol=pickle.HIGHEST_PROTOCOL)


def mean_shear_around_focal(mdet_focal_filepath, outpath, weight_scheme, wgt_path, n_bin):

    # Read in info. 
    import glob
    fs = glob.glob(os.path.join(mdet_focal_filepath, 'focal_plane_coords_*.pickle'))

    # Define radial binning
    n = n_bin
    R0 = 4096*7*0.26#*np.sqrt(2) # arcsec 
    rad = []
    for p in range(1,n+1):
        rad.append(annulus(p, n, R0))

    # For each radial bin, accumulate shear from each tile info. 
    print('data processing in progress...')
    binnum = n
    res = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
            '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
            '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
            '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
            '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
    for tile in tqdm(fs):
        with open(tile, 'rb') as raw:
            d = pickle.load(raw)
        res = accum_shear_tile(res, d, rad, n_bin)


def mean_shear_around_coadd(mdet_input_filepath, outpath, weight_scheme, wgt_path, n_bin):

    # Read in catalog
    f = h5.File(mdet_input_filepath, 'r')
    d = f.get('/mdet/noshear')
    d_2p = f.get('/mdet/2p')
    d_1p = f.get('/mdet/1p')
    d_2m = f.get('/mdet/2m')
    d_1m = f.get('/mdet/1m')
    keys = ['x', 'y', 'gauss_g_1', 'gauss_g_2', 'patch_num']
    gal_data = read_mdet_h5(d, d_1p, d_2p, d_1m, d_2m, keys, weight_scheme, wgt_path)

    # Define radial binning
    n = n_bin
    R0 = 5000*0.26#*np.sqrt(2) # arcsec 
    rad = []
    for p in range(1,n+1):
        rad.append(annulus(p, n, R0))

    # For each radial bin, accumulate shear for each sheared catalog. 
    print('data processing in progress...')
    binnum = n
    res = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
            '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
            '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
            '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
            '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
    for ri in tqdm(range(n)):
        if ri == 0:
            r0 = 0
            rp = rad[0]
        else:
            r0 = rad[ri-1]
            rp = rad[ri]

        res = accum_shear(res, ri, gal_data, r0, rp, 'noshear')
        res = accum_shear(res, ri, gal_data, r0, rp, '1p')
        res = accum_shear(res, ri, gal_data, r0, rp, '1m')
        res = accum_shear(res, ri, gal_data, r0, rp, '2p')
        res = accum_shear(res, ri, gal_data, r0, rp, '2m')
        
    # Compute shear response and mean shear
    stats = np.zeros(n, dtype=[('rp', 'f8'), ('e1', 'f8'), ('e2', 'f8'), ('e1_cov', 'f8'), ('e2_cov', 'f8')])
    mean_shear = _compute_g1_g2(res, n)
    stats['rp'] = rad
    stats['e1'] = mean_shear['e1']
    stats['e2'] = mean_shear['e2']
    

    # Compute jackknife errors
    print('jackknife process in progress...')
    tilenames = np.unique(np.array(d['patch_num']))
    jk_sample = len(tilenames)
    jk_stats = {b: {'e1': np.zeros(jk_sample), 'e2': np.zeros(jk_sample)} for b in range(n)}
    dset = [d, d_1p, d_1m, d_2p, d_2m]
    for i,ith in tqdm(enumerate(tilenames)):
        res_jk = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
            '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
            '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
            '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
            '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
        jk_sample_mean = _compute_shear_per_jksample_radial(gal_data, rad, res_jk, ith, tilenames, binnum)
        for b in range(n):
            jk_stats[b]['e1'][i] = jk_sample_mean[b]['e1']
            jk_stats[b]['e2'][i] = jk_sample_mean[b]['e2']
    jk_cov = _compute_jackknife_error_estimate(jk_stats, binnum, jk_sample)
    stats['e1_cov'] = jk_cov[:, 0]
    stats['e2_cov'] = jk_cov[:, 1]

    fio.write(os.path.join(outpath, 'mean_shear_around_cell_coordinates.fits'), stats)

def tan_shear_around_cell_coords(mdet_input_filepath, outpath, weight_scheme, wgt_path, random_point_map, f_pc, var_method, color_split=False):

    import treecorr
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Read in catalog
    print('reading in catalog...')
    keys = ['ra', 'dec', 'g1', 'g2', 'w', 'tilename', 'pgauss_band_flux_g', 'pgauss_band_flux_i']
    gal_data = load_mdet_h5(mdet_input_filepath, keys, weight_scheme, wgt_path, response=True, subtract_mean_shear=True, color_split=color_split)

    # Read in coadd center tile & convert tile center into cell center. 
    coadd_cent = fio.read('/pscratch/sd/m/myamamot/coadd_tile.fits')
    tnames = coadd_cent['TILENAME']
    destile_cent = coadd_cent[np.in1d(tnames, gal_data['tilename'])]
    cell_cent = _get_cell_center(destile_cent)
    print('number of centers ', len(cell_cent))
    
    # set up treecorr call
    bin_config = dict(
                sep_units = 'arcmin',
                bin_slop = 0.01,

                min_sep = 0.1,
                max_sep = 1.25,
                nbins = 15,

                var_method = var_method,
                output_dots = False,
                )
    cat1 = treecorr.Catalog(ra=cell_cent['ra'], dec=cell_cent['dec'], ra_units='deg', dec_units='deg', patch_centers=f_pc)
    # random point subtraction. 
    cat1r_file = random_point_map
    cat1r = treecorr.Catalog(cat1r_file, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg', patch_centers=f_pc)
    ng_rand = treecorr.NGCorrelation(bin_config, verbose=2)
    ng = treecorr.NGCorrelation(bin_config, verbose=2)

    print('treecorr starting...')
    cat2 = treecorr.Catalog(ra=gal_data['ra'], dec=gal_data['dec'], ra_units='deg', dec_units='deg', g1=gal_data['g1'], g2=gal_data['g2'], w=gal_data['w'], patch_centers=f_pc)
    ng.process(cat1, cat2, low_mem=True, comm=comm)
    ng_rand.process(cat1r, cat2, low_mem=True, comm=comm)

    if rank==0:
        print('writing out files...')
        ng.write(os.path.join(outpath, 'cell_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.1_max1.25_norand.fits'))
        ng_rand.write(os.path.join(outpath, 'cell_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.1_max1.25_randonly.fits'))
        ng.write(os.path.join(outpath, 'cell_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.1_max1.25.fits'), rg=ng_rand)
        ng.calculateXi(rg=ng_rand)
        ng_cov = ng.cov
        np.save(os.path.join(outpath, 'cell_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.1_max1.25_cov.npy'), ng_cov)


def tan_shear_around_coadd_coords(mdet_input_filepath, outpath, weight_scheme, wgt_path, random_point_map, f_pc, var_method, color_split=False):

    import treecorr
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Read in catalog
    print('reading in catalog...')
    keys = ['ra', 'dec', 'g1', 'g2', 'w', 'tilename', 'pgauss_band_flux_g', 'pgauss_band_flux_i']
    gal_data = load_mdet_h5(mdet_input_filepath, keys, weight_scheme, wgt_path, response=True, subtract_mean_shear=True, color_split=color_split)

    # Read in coadd center tile
    print('reading in coadd center...')
    coadd_cent = fio.read('/pscratch/sd/m/myamamot/coadd_tile.fits')
    tnames = coadd_cent['TILENAME']
    destile_cent = coadd_cent[np.in1d(tnames, gal_data['tilename'])]
    
    # set up treecorr call
    bin_config = dict(
                sep_units = 'arcmin',
                bin_slop = 0.01,

                min_sep = 0.5,
                max_sep = 30,
                nbins = 15,

                var_method = var_method,
                output_dots = False,
                )
    print('treecorr catalog...')
    cat1 = treecorr.Catalog(ra=destile_cent['RA_CENT'], dec=destile_cent['DEC_CENT'], ra_units='deg', dec_units='deg', patch_centers=f_pc)
    # random point subtraction. 
    cat1r_file = random_point_map
    cat1r = treecorr.Catalog(cat1r_file, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg', patch_centers=f_pc)
    ng_rand = treecorr.NGCorrelation(bin_config, verbose=2)
    ng = treecorr.NGCorrelation(bin_config, verbose=2)

    print('treecorr starting...')
    cat2 = treecorr.Catalog(ra=gal_data['ra'], dec=gal_data['dec'], ra_units='deg', dec_units='deg', g1=gal_data['g1'], g2=gal_data['g2'], w=gal_data['w'], patch_centers=f_pc)
    ng.process(cat1, cat2, low_mem=True, comm=comm)
    ng_rand.process(cat1r, cat2, low_mem=True, comm=comm)

    if rank==0:
        print('writing out files...')
        ng.write(os.path.join(outpath, 'coadd_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.5_max30_norand.fits'))
        ng_rand.write(os.path.join(outpath, 'coadd_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.5_max30_randonly.fits'))
        ng.write(os.path.join(outpath, 'coadd_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.5_max30.fits'), rg=ng_rand)
        ng.calculateXi(rg=ng_rand)
        ng_cov = ng.cov
        np.save(os.path.join(outpath, 'coadd_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.5_max30_cov.npy'), ng_cov)


def main(argv):
    
    mdet_input_filepath = sys.argv[1]
    weight_name = sys.argv[2]
    weight_path = sys.argv[3]
    out_path = sys.argv[4]
    n_bin = int(sys.argv[5])
    test_name = sys.argv[6]
    
    if test_name == 'mean_shear_radial':
        mean_shear_around_coadd(mdet_input_filepath, out_path, weight_name, weight_path, n_bin)
    elif test_name == 'mean_shear_cell_grid':
        mean_shear_in_cell_grid(mdet_input_filepath, out_path, weight_name, weight_path)
    elif test_name == 'mean_shear_coadd_grid':
        mean_shear_in_coadd_grid(mdet_input_filepath, out_path, weight_name, weight_path)
    elif test_name == 'tan_shear_cell':
        rand_map = sys.argv[7]
        patch_center = sys.argv[8]
        var_method = sys.argv[9]
        color_split = eval(sys.argv[10])
        tan_shear_around_cell_coords(mdet_input_filepath, out_path, weight_name, weight_path, rand_map, patch_center, var_method, color_split=color_split)
    elif test_name == 'tan_shear_coadd':
        rand_map = sys.argv[7]
        patch_center = sys.argv[8]
        var_method = sys.argv[9]
        color_split = eval(sys.argv[10])
        tan_shear_around_coadd_coords(mdet_input_filepath, out_path, weight_name, weight_path, rand_map, patch_center, var_method, color_split=color_split)

if __name__ == "__main__":
    main(sys.argv)