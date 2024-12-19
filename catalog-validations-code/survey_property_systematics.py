# from nis import match
import os, sys
# from tokenize import group
from tqdm import tqdm
import numpy as np
import fitsio as fio
import matplotlib as mpl
import glob
import pickle
import healsparse
import healpy as hp
from des_y6utils import mdet
import healpy as hp
import time
# import numpy_groupies as npg
from esutil import stat

def read_mdet_h5(datafile, gal_weight_file, keys, mdet_step, patch_id=None, response=False, subtract_mean_shear=False, ):

    def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
        """
        Computes indices of 2D grids. Only used when we use shear weight that is binned by S/N and size ratio. 
        """
        from math import log10
        # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps

        logstepx = log10(xmax/xmin)/xsteps
        logstepy = log10(ymax/ymin)/ysteps

        indexx = (np.log10(x/xmin)/logstepx).astype(int)
        indexy = (np.log10(y/ymin)/logstepy).astype(int)

        indexx = np.maximum(indexx,0)
        indexx = np.minimum(indexx, xsteps-1)
        indexy = np.maximum(indexy,0)
        indexy = np.minimum(indexy, ysteps-1)

        return indexx,indexy

    def _find_shear_weight(dat, mask, wgt_dict, snmin, snmax, sizemin, sizemax, steps, mdet_mom):

        """
        Assigns shear weights to the objects based on the grids. 
        """
        
        if wgt_dict is None:
            weights = np.ones(len(dat))
            return weights

        shear_wgt = wgt_dict['weight']
        smoothing = True
        if smoothing:
            from scipy.ndimage import gaussian_filter
            smooth_response = gaussian_filter(wgt_dict['response'], sigma=2.0)
            shear_wgt = (smooth_response/wgt_dict['meanes'])**2
        indexx, indexy = assign_loggrid(np.array(dat[mdet_mom+'_s2n'])[mask], np.array(dat[mdet_mom+'_T_ratio'])[mask], snmin, snmax, steps, sizemin, sizemax, steps)
        weights = np.array([shear_wgt[x, y] for x, y in zip(indexx, indexy)])
        
        return weights

    def _get_shear_weights(dat, mask, gal_weight_file, shape_err=False):
        if shape_err:
            return 1/(0.22**2 + 0.5*(np.array(dat['gauss_g_cov_1_1'])[mask] + np.array(dat['gauss_g_cov_2_2'])[mask]))
        else:
            with open(gal_weight_file, 'rb') as handle:
                wgt_dict = pickle.load(handle)
                snmin = wgt_dict['xedges'][0]
                snmax = wgt_dict['xedges'][-1]
                sizemin = wgt_dict['yedges'][0]
                sizemax = wgt_dict['yedges'][-1]
                steps = len(wgt_dict['xedges'])-1
            shear_wgt = _find_shear_weight(dat, mask, wgt_dict, snmin, snmax, sizemin, sizemax, steps, 'gauss')
            return shear_wgt

    def _wmean(q,w):
        return np.sum(q*w)/np.sum(w)
    
    import h5py as h5
    f = h5.File(datafile, 'r')
    d = f.get('/mdet/'+mdet_step)
    if patch_id is None:
        nrows = len(np.array( d['ra'] ))
        mask = np.ones(nrows)!=0
    else:
        mask = (np.array( d['patch_num'] ) == patch_id)
        nrows = len(np.array( d['ra'] )[mask])
    formats = []
    for key in keys:
        formats.append('f4')
    data = np.recarray(shape=(nrows,), formats=formats, names=keys)
    for key in keys:  
        if key == 'w':
            data['w'] = _get_shear_weights(d, mask, gal_weight_file)
        elif key in ('g1', 'g2'):
            data[key] = np.array(d['gauss_'+key[0]+'_'+key[1]])[mask]
        elif key == 'gauss_T':
            data[key] = np.array(d['gauss_psf_T'])[mask] * np.array(d['gauss_T_ratio'])[mask]
        elif key == 'gmi':
            mag_g = mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_g"])[mask], 0)
            mag_i = mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_i"])[mask], 2)
            data[key] = mag_g - mag_i
        else:
            data[key] = np.array(d[key])[mask]
    print('made recarray with hdf5 file')
    
    # response correction
    if response:
        d_2p = f.get('/mdet/2p')
        d_1p = f.get('/mdet/1p')
        d_2m = f.get('/mdet/2m')
        d_1m = f.get('/mdet/1m')
        # compute response with weights
        g1p = _wmean(np.array(d_1p["gauss_g_1"]), _get_shear_weights(d_1p, gal_weight_file))                                     
        g1m = _wmean(np.array(d_1m["gauss_g_1"]), _get_shear_weights(d_1m, gal_weight_file))
        R11 = (g1p - g1m) / 0.02

        g2p = _wmean(np.array(d_2p["gauss_g_2"]), _get_shear_weights(d_2p, gal_weight_file))
        g2m = _wmean(np.array(d_2m["gauss_g_2"]), _get_shear_weights(d_2m, gal_weight_file))
        R22 = (g2p - g2m) / 0.02

        R = (R11 + R22)/2.
        data['g1'] /= R
        data['g2'] /= R

        mean_g1 = _wmean(data['g1'], data['w'])
        mean_g2 = _wmean(data['g2'], data['w'])
        std_g1 = np.var(data['g1'])
        std_g2 = np.var(data['g2'])
        mean_shear = [mean_g1, mean_g2, std_g1, std_g2]
        # mean shear subtraction
        if subtract_mean_shear:
            print('subtracting mean shear')
            print('mean g1 g2 =(%1.8f,%1.8f)'%(mean_g1, mean_g2))          
            data['g1'] -= mean_g1
            data['g2'] -= mean_g2

    return data

def _compute_shear(mean_shear_output, group_shear_output, binnum, hist, bin_shear, pkdgrav=False):

    """
    Computes shear response and correct raw mean shear by response in each signal bin. 
    """

    if pkdgrav:
        for bind in range(binnum):
            g1 = (group_shear_output['noshear'][bind,0]/group_shear_output['num_noshear'][bind,0])
            g2 = (group_shear_output['noshear'][bind,1]/group_shear_output['num_noshear'][bind,1])
            tmp = np.concatenate(bin_shear[bind]['g1'])
            mean_shear_output['g1_cov'][bind] = np.std(tmp)/np.sqrt(len(tmp))
            tmp = np.concatenate(bin_shear[bind]['g2'])
            mean_shear_output['g2_cov'][bind] = np.std(tmp)/np.sqrt(len(tmp))

            mean_shear_output['g1'][bind] = g1
            mean_shear_output['g2'][bind] = g2
            mean_shear_output['mean_signal'][bind] = hist['mean'][bind]
    else:
        for bind in range(binnum):
            R11 = (group_shear_output['1p'][bind, 0]/group_shear_output['num_1p'][bind,0] - group_shear_output['1m'][bind, 0]/group_shear_output['num_1m'][bind,0])/0.02
            R22 = (group_shear_output['2p'][bind, 1]/group_shear_output['num_2p'][bind,1] - group_shear_output['2m'][bind, 1]/group_shear_output['num_2m'][bind,1])/0.02
            R = (R11+R22)/2.
            
            g1 = (group_shear_output['noshear'][bind,0]/group_shear_output['num_noshear'][bind,0])/R
            g2 = (group_shear_output['noshear'][bind,1]/group_shear_output['num_noshear'][bind,1])/R

            tmp = np.concatenate(bin_shear[bind]['g1'])
            mean_shear_output['g1_cov'][bind] = np.std(tmp/R)/np.sqrt(len(tmp))
            tmp = np.concatenate(bin_shear[bind]['g2'])
            mean_shear_output['g2_cov'][bind] = np.std(tmp/R)/np.sqrt(len(tmp))

            mean_shear_output['R'][bind] = R
            mean_shear_output['g1'][bind] = g1
            mean_shear_output['g2'][bind] = g2
            mean_shear_output['mean_signal'][bind] = hist['mean'][bind]

    return mean_shear_output


def _accum_shear_pixel(d, d_pix, total_shear_output, total_number_output):

    for i, step in enumerate(['noshear', '1p', '1m', '2p', '2m']):
        msk_s = np.where(d['mdet_step'] == step)[0]

        t0 = time.time()
        group_e1 = npg.aggregate(d_pix[msk_s], d[msk_s]['mdet_g_1'].astype('float'), func='sum', fill_value=0)
        group_e2 = npg.aggregate(d_pix[msk_s], d[msk_s]['mdet_g_2'].astype('float'), func='sum', fill_value=0)
        group_nume1 =  np.bincount(d_pix[msk_s])
        group_nume2 =  np.bincount(d_pix[msk_s])
        # print('grouping', time.time()-t0)

        t0 = time.time()
        msk_nonzero = (group_e1 != 0)
        index_pixel_e1 = np.where(msk_nonzero)[0]
        msk_nonzero = (group_e2 != 0)
        index_pixel_e2 = np.where(msk_nonzero)[0]
        msk_nonzero = (group_nume1 != 0)
        index_pixel_nume1 = np.where(msk_nonzero)[0]
        msk_nonzero = (group_nume2 != 0)
        index_pixel_nume2 = np.where(msk_nonzero)[0]
        # print('index', time.time()-t0)

        t0 = time.time()
        total_shear_output[i][index_pixel_e1, 0] += group_e1[index_pixel_e1]
        total_shear_output[i][index_pixel_e2, 1] += group_e2[index_pixel_e2]
        total_number_output[i][index_pixel_nume1, 0] += group_nume1[index_pixel_nume1]
        total_number_output[i][index_pixel_nume2, 1] += group_nume2[index_pixel_nume2]
        # print('accumulate', time.time()-t0)

    return total_shear_output, total_number_output


def _accum_shear_bin(dat, hmap, bin_edges, total_shear_output, mdet_mom):

    """
    Accumulates weighted shear and effective number count in each bin. 
    """

    for i,step in enumerate(['noshear', '1p', '1m', '2p', '2m']):
        d_bin_signal = hmap.get_values_pos(dat[i]['ra'], dat[i]['dec'], lonlat=True)
        bin_index = np.digitize(d_bin_signal, bin_edges, right=True) - 1
        shear_wgt = dat[i]['w']

        t0 = time.time()
        np.add.at(total_shear_output[step], (bin_index, 0), dat[i][mdet_mom+'_g_1']*shear_wgt) 
        np.add.at(total_shear_output[step], (bin_index, 1), dat[i][mdet_mom+'_g_2']*shear_wgt) 
        np.add.at(total_shear_output["num_"+step], (bin_index, 0), shear_wgt)
        np.add.at(total_shear_output["num_"+step], (bin_index, 1), shear_wgt)
        # print('accumulate', time.time()-t0)

    return total_shear_output


def _compute_std_bin(d, hmap, bin_edges, bin_shear, mdet_mom, pkdgrav=False):

    """
    Save raw shear to compute the standard deviation in each signal bin later. 
    """

    d_bin_signal = hmap.get_values_pos(d['ra'], d['dec'], lonlat=True)
    if not pkdgrav:
        bin_index = np.digitize(d_bin_signal, bin_edges, right=True) - 1
        for bind in range(len(bin_edges)-1):
            bind_mask = np.in1d(bin_index, bind)
            bin_shear[bind]['g1'].append(d[bind_mask][mdet_mom+'_g_1'])
            bin_shear[bind]['g2'].append(d[bind_mask][mdet_mom+'_g_2'])
    else:
        bin_index = np.digitize(d_bin_signal, bin_edges, right=True) - 1
        for bind in range(len(bin_edges)-1):
            bind_mask = np.in1d(bin_index, bind)
            bin_shear[bind]['g1'].append(d['e1'][bind_mask])
            bin_shear[bind]['g2'].append(d['e2'][bind_mask])
        
    return bin_shear


def _accum_shear_bin_pkdgrav(d, d_bin_signal, bin_edges, total_shear_output, shear_wgt):

    """
    Accumulates shear for simulations. 
    """

    bin_index = np.digitize(d_bin_signal, bin_edges, right=True) - 1

    np.add.at(total_shear_output['noshear'], (bin_index, 0), d['e1']*shear_wgt) 
    np.add.at(total_shear_output['noshear'], (bin_index, 1), d['e2']*shear_wgt) 
    np.add.at(total_shear_output['num_noshear'], (bin_index, 0), shear_wgt)
    np.add.at(total_shear_output['num_noshear'], (bin_index, 1), shear_wgt)

    return total_shear_output


def _divide_bins(data, prop, nperbin=1000000, dict_type=True):

    """
    Divide survey property signals into bins. The number of bins is set so that an equal number of objects falls into each bin. 
    """

    if dict_type:
        pix_val = np.array(list(data.values()))
    else:
        pix_val = data

    if prop == 'exptime':
        hist = stat.histogram(pix_val, binsize=90, max=1500, more=True)
        hist['high'][-1] = np.max(pix_val)
    elif prop == 'stellar':
        pix_val = pix_val[pix_val > 0]
        hist = stat.histogram(pix_val, nperbin=nperbin, more=True)
    else:
        hist = stat.histogram(pix_val, nperbin=nperbin, more=True)
    binnum = len(hist['mean'])
    bin_edges = np.insert(hist['high'], 0, hist['low'][0])
    
    return binnum, bin_edges, hist


def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
    """
    Computes indices of 2D grids. Only used when we use shear weight that is binned by S/N and size ratio. 
    """
    from math import log10
    # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps

    logstepx = log10(xmax/xmin)/xsteps
    logstepy = log10(ymax/ymin)/ysteps

    indexx = (np.log10(x/xmin)/logstepx).astype(int)
    indexy = (np.log10(y/ymin)/logstepy).astype(int)

    indexx = np.maximum(indexx,0)
    indexx = np.minimum(indexx, xsteps-1)
    indexy = np.maximum(indexy,0)
    indexy = np.minimum(indexy, ysteps-1)

    return indexx,indexy

def _find_shear_weight(d, wgt_dict, snmin, snmax, sizemin, sizemax, steps, mdet_mom):

    """
    Assigns shear weights to the objects based on the grids. 
    """
    
    if wgt_dict is None:
        weights = np.ones(len(d))
        return weights

    shear_wgt = wgt_dict['weight']
    smoothing = True
    if smoothing:
        from scipy.ndimage import gaussian_filter
        smooth_response = gaussian_filter(wgt_dict['response'], sigma=2.0)
        shear_wgt = (smooth_response/wgt_dict['meanes'])**2
    indexx, indexy = assign_loggrid(np.array(d['gauss_s2n']), np.array(d['gauss_T_ratio']), snmin, snmax, steps, sizemin, sizemax, steps)
    weights = np.array([shear_wgt[x, y] for x, y in zip(indexx, indexy)])

    # prior = ngmix.priors.GPriorBA(0.3, rng=np.random.RandomState())
    # pvals = prior.get_prob_array2d(d['wmom_g_1'], d['wmom_g_2'])
    # weights *= pvals
    
    return weights

def jk_function(input_, outpath, sample_variance_filepath, mdet_mom):

    """
    Computes mean shear and standard deviation for each signal bin for each pkdgrav simulations, and saves those information in fits file. 
    """

    [sim_id,hmap,band,bin_edges,binnum,hist] = input_

    bin_shear = {i: {'g1':[], 'g2':[]} for i in range(binnum)}
    group_shear_output = {'noshear': np.zeros((binnum, 2)), '1p': np.zeros((binnum, 2)), '1m': np.zeros((binnum, 2)), '2p': np.zeros((binnum, 2)), '2m': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
    with open(os.path.join(sample_variance_filepath, 'seed__fid_cosmogrid_'+str(sim_id+1)+'.pkl'), 'rb') as handle:
        res = pickle.load(handle)
        handle.close()
    d = res['sources'][0]
    shear_wgt = d['w']
    d_bin_signal = hmap.get_values_pos(d['ra'], d['dec'], lonlat=True)

    group_shear_output = _accum_shear_bin_pkdgrav(d, d_bin_signal, bin_edges, group_shear_output, shear_wgt)
    bin_shear = _compute_std_bin(d, hmap, bin_edges, bin_shear, mdet_mom, pkdgrav=True)

    mean_shear_output = np.zeros(binnum, dtype=[('mean_signal', 'f8'), ('g1', 'f8'), ('g2', 'f8'), ('g1_cov', 'f8'), ('g2_cov', 'f8')])
    mean_shear_output = _compute_shear(mean_shear_output, group_shear_output, binnum, hist, bin_shear, pkdgrav=True)
    
    fio.write(os.path.join(outpath, 'seed__fid_'+str(sim_id+1)+'_'+band+'.fits'), mean_shear_output)


def main(argv):

    """
    
    Parameters
    ----------
    mdet_input_filepaths: The input filepath for the metadetection catalogs
            Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3
    sample_variance_filepaths: The input filepath for the simulated (PKDGRAV) sample variance catalogs
            Example) /global/cscratch1/sd/myamamot/sample_variance
    map_file: a list of files that contain the values for survey properties in healpix map
    nperbin: a number of objects in a given object property bin (default=1000000)
    method: a method of how to compute the survey systematics ('bin' computes the shear response in each bin. 'pixel' copmutes the shear response in each healpixel. This option is very expensive.)
    num_sim: the number of PKDGRAV simulations
    outpath: the path to output directory
            Example) /global/cscratch1/sd/myamamot/survey_property_maps/airmass
    prop: the systematic map property
    band: the band for which survey systematic is measured
    mdet_mom: which estimator to use
    mdet_cuts: which version of selection cuts to use
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('mpi rank', rank, size)

    # properties: airmass, fwhm, exptime, skybrite, stellar, background, dcre1, dcre2, dcrra, dcrdec, SFD
    prop = sys.argv[1]
    map_file = sys.argv[2]
    mdet_input_filepaths = sys.argv[3]
    sample_variance_filepath = sys.argv[4]
    outpath = sys.argv[5] # airmass, exposure, fwhm, sky_brightness, sky_sigma
    mdet_mom = sys.argv[6]
    mdet_cuts = sys.argv[7]
    nperbin = sys.argv[8]
    weight_scheme = sys.argv[9]
    weight_filepath = sys.argv[10]
    if prop == 'background':
        band = map_file.split('/')[-1].split('_')[-1][0]
    elif prop == 'dust':
        band = 'sfd'
    else:
        band = map_file.split('/')[-1].split('_')[2]

    # bin up signals
    hmap = healsparse.HealSparseMap.read(map_file)
    print('healsparse map read...')
    # sys.stdout.flush()
    if prop == 'stellar':
        # mask_map = healsparse.HealSparseMap.read('/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hleda-gaiafull-des-stars-hsmap16384-nomdet-v3.fits')
        # vpix, ra, dec = hmap.valid_pixels_pos(return_pixels=True)
        # in_footprint = mask_map.get_values_pos(ra, dec, valid_mask=True)
        # hmap[vpix[~in_footprint]] = hmap._sentinel
        hmap = hmap.degrade(512)
        pix_signal = hmap.get_values_pix(hmap.valid_pixels)
    elif prop == 'dust':
        valid_pix = np.load('/global/cfs/cdirs/des/myamamot/survey_property_maps/SFD_unique_pix_4096.npy')
        pix_signal = hmap.get_values_pix(valid_pix)
    else:
        pix_signal = hmap.get_values_pix(hmap.valid_pixels)
    binnum, bin_edges, hist = _divide_bins(pix_signal, prop, nperbin=nperbin, dict_type=False)

    # spread out jobs
    group_shear_output = {'noshear': np.zeros((binnum, 2)), '1p': np.zeros((binnum, 2)), '1m': np.zeros((binnum, 2)), '2p': np.zeros((binnum, 2)), '2m': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
    # fs = sorted(glob.glob(mdet_input_filepaths))
    fids = np.arange(200)
    bin_shear = {i: {'g1':[], 'g2':[]} for i in range(binnum)}
    if not os.path.exists(os.path.join(outpath, 'accum_shear_patch_'+band+'_0.pickle')):
        for pname in fids:
            if pname % size != rank:
                continue
            if pname % 20 == 0:
                print('made it to ', pname)
            
            keys = ['ra', 'dec', mdet_mom+'_g_1', mdet_mom+'_g_2', 'w']
            d = read_mdet_h5(mdet_input_filepaths, weight_filepath, keys, 'noshear', patch_id=pname, response=False, subtract_mean_shear=True)
            d_1p = read_mdet_h5(mdet_input_filepaths, weight_filepath, keys, '1p', patch_id=pname, response=False, subtract_mean_shear=True)
            d_1m = read_mdet_h5(mdet_input_filepaths, weight_filepath, keys, '1m', patch_id=pname, response=False, subtract_mean_shear=True)
            d_2p = read_mdet_h5(mdet_input_filepaths, weight_filepath, keys, '2p', patch_id=pname, response=False, subtract_mean_shear=True)
            d_2m = read_mdet_h5(mdet_input_filepaths, weight_filepath, keys, '2m', patch_id=pname, response=False, subtract_mean_shear=True)
            d_all = [d, d_1p, d_1m, d_2p, d_2m]

            group_shear_output = _accum_shear_bin(d_all, hmap, bin_edges, group_shear_output, mdet_mom)
            bin_shear = _compute_std_bin(d, hmap, bin_edges, bin_shear, mdet_mom)

            with open(os.path.join(outpath, 'accum_shear_patch_'+band+'_'+str(pname)+'.pickle'), 'wb') as fpatch:
                pickle.dump(group_shear_output, fpatch, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(outpath, 'bin_shear_patch_'+band+'_'+str(pname)+'.pickle'), 'wb') as fpatch:
                pickle.dump(bin_shear, fpatch, protocol=pickle.HIGHEST_PROTOCOL)
        comm.Barrier()

    if not os.path.exists(os.path.join(outpath, prop+'_'+band+'_systematics.fits')):
        group_shear_output = {'noshear': np.zeros((binnum, 2)), '1p': np.zeros((binnum, 2)), '1m': np.zeros((binnum, 2)), '2p': np.zeros((binnum, 2)), '2m': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
        bin_shear = {i: {'g1':[], 'g2':[]} for i in range(binnum)}
        if rank == 0:
            print('accumulating info at rank 0')
            for pname in tqdm(fids):
                with open(os.path.join(outpath, 'accum_shear_patch_'+band+'_'+str(pname)+'.pickle'), 'rb') as f:
                    tmp_res = pickle.load(f)
                for step in ['noshear', '1p', '1m', '2p', '2m']:
                    group_shear_output[step] += tmp_res[step]
                    group_shear_output["num_"+step] += tmp_res["num_"+step]

                with open(os.path.join(outpath, 'bin_shear_patch_'+band+'_'+str(pname)+'.pickle'), 'rb') as f:
                    tmp_res_std = pickle.load(f)
                for b in range(binnum):
                    bin_shear[b]['g1'] += tmp_res_std[b]['g1']
                    bin_shear[b]['g2'] += tmp_res_std[b]['g2']
            
            print('writing output file...')
            mean_shear_output = np.zeros(binnum, dtype=[('mean_signal', 'f8'), ('R', 'f8'), ('g1', 'f8'), ('g2', 'f8'), ('g1_cov', 'f8'), ('g2_cov', 'f8')])
            mean_shear_output = _compute_shear(mean_shear_output, group_shear_output, binnum, hist, bin_shear)
            fio.write(os.path.join(outpath, prop+'_'+band+'_systematics.fits'), mean_shear_output)
        print('mean shear process done...', rank)
    else:
        print('mean shear output file already exists... moving to jackknife covariance estimation.')
    comm.Barrier()

    # Do jackknife; run each sim for each rank
    runs = []
    for sim_id in range(800):
        runs.append([sim_id, hmap, band, bin_edges, binnum, hist])
    for i in range(len(runs)):
        if i % size != rank:
            continue
        if i % 20 == 0:
            print('jk made it to ', i)
        if not os.path.exists(os.path.join(outpath, 'seed__fid_'+str(i+1)+'_'+band+'.fits')):
            jk_function(runs[i], outpath, sample_variance_filepath, mdet_mom)
    comm.Barrier()


if __name__ == "__main__":
    main(sys.argv)