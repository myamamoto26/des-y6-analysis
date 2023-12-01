from nis import match
import os, sys
from tokenize import group
from tqdm import tqdm
import numpy as np
import fitsio as fio
import matplotlib as mpl
import glob
np.random.seed(1738)
import matplotlib.pyplot as plt
from math import log10
import pickle
from des_y6utils import mdet

def inverse_variance_weight(d, shear_wgt_output_filepath, mdet_mom, steps, snmin, snmax, sizemin, sizemax, mdet_cuts):

    """
    Returns galaxy count, shear response, variance of e, shear weight as a function of S/N and size ratio.

    Parameters
    ----------
    mdet_input_filepaths: The file path to the directory in which the input metadetection catalogs exist
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3

    shear_wgt_output_filepath: The file path where the output pickle file is written
    Example) /global/cscratch1/sd/myamamot/metadetect/inverse_variance_weight_v3_Trcut_snmax1000.pickle

    mdet_mom: which measurement do we want to make cuts on
    steps: The bin number in S/N and size ratio
    snmin: The minimum S/N to be considered
    snmax: The maximum S/N to be considered
    sizemin: The minimum size ratio (T/Tpsf) to be considered
    sizemax: The maximum size ratio to be considered
    mdet_cuts: which cut ID do you want to use?
    """

    def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
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

    def apply_loggrid(x, y, grid, xmin=0, xmax=0, xsteps=0, ymin=0, ymax=0, ysteps=0):
        indexx,indexy = assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps)
        res = np.zeros(len(x))
        res = grid[indexx,indexy]
        return res

    def logmeshplot(data, xedges, yedges, label="quantity"):
        fig=plt.figure(figsize=(6,6))
        ax = plt.subplot(111)
        X, Y = np.meshgrid(yedges, xedges)
        plt.pcolormesh(X, Y, data)
        plt.xscale('log')
        plt.yscale('log')
        plt.colorbar(label=label)
        plt.ylabel("mcal snr")
        plt.xlabel("mcal size/psf_size")

        plt.minorticks_off() 
        ax.set_xticks(np.array([0.6,0.7,0.8,0.9,1.,2.,3.,4.,]))
        ax.set_xticklabels(np.array([r'5 x $10^{-1}$','','','',r'$10^{0}$',r'2 x $10^{0}$','',r'4 x $10^{0}$']))

    def mesh_average(m, quantity,indexx,indexy,steps,count):
        m = np.zeros((steps,steps)) # revised version, was -1 before
        np.add.at(m,(indexx,indexy),quantity)
        m /= count
        return m

    def find_assign_grid(d, mdet_step, mdet_mom, snmin, snmax, steps, sizemin, sizemax):

        mask = mdet_step
        mastercat_snr = d[mask][mdet_mom+'_s2n']
        mastercat_Tr = d[mask][mdet_mom+'_T_ratio']
        new_indexx,new_indexy = assign_loggrid(mastercat_snr, mastercat_Tr, snmin, snmax, steps, sizemin, sizemax, steps)
        
        return new_indexx, new_indexy

    def find_bincount_2d(indexx, indexy, steps):

        from collections import Counter
        index_tup = [(i,j) for i,j in zip(indexx, indexy)]
        count = Counter(index_tup)
        all_count = np.zeros((steps, steps))
        for i in range(steps):
            for j in range(steps):
                if count[(i,j)] != 0:
                    all_count[i,j] = count[(i,j)]
        
        return all_count

    def accumulate_shear_per_tile(res, d, mdet_mom, snmin, snmax, steps, sizemin, sizemax):

        g1p_indexx, g1p_indexy = find_assign_grid(d, '1p', mdet_mom, snmin, snmax, steps, sizemin, sizemax)
        g1p_count = find_bincount_2d(g1p_indexx, g1p_indexy, steps)
        g1m_indexx, g1m_indexy = find_assign_grid(d, '1m', mdet_mom, snmin, snmax, steps, sizemin, sizemax)
        g1m_count = find_bincount_2d(g1m_indexx, g1m_indexy, steps)
        g2p_indexx, g2p_indexy = find_assign_grid(d, '2p', mdet_mom, snmin, snmax, steps, sizemin, sizemax)
        g2p_count = find_bincount_2d(g2p_indexx, g2p_indexy, steps)
        g2m_indexx, g2m_indexy = find_assign_grid(d, '2m', mdet_mom, snmin, snmax, steps, sizemin, sizemax)
        g2m_count = find_bincount_2d(g2m_indexx, g2m_indexy, steps)

        np.add.at(res['g_1p'], (g1p_indexx, g1p_indexy), d['1p'][mdet_mom+'_g_1'])
        np.add.at(res['g_1m'], (g1m_indexx, g1m_indexy), d['1m'][mdet_mom+'_g_1'])
        np.add.at(res['g_2p'], (g2p_indexx, g2p_indexy), d['2p'][mdet_mom+'_g_2'])
        np.add.at(res['g_2m'], (g2m_indexx, g2m_indexy), d['2m'][mdet_mom+'_g_2'])
        
        np.add.at(res['g1p_count'], (), g1p_count)
        np.add.at(res['g1m_count'], (), g1m_count)
        np.add.at(res['g2p_count'], (), g2p_count)
        np.add.at(res['g2m_count'], (), g2m_count)

        return res

    def compute_mesh_response(res):
        
        g_1p = res['g_1p']/res['g1p_count']
        g_1m = res['g_1m']/res['g1m_count']
        g_2p = res['g_2p']/res['g2p_count']
        g_2m = res['g_2m']/res['g2m_count']

        R11 = (g_1p - g_1m)/0.02
        R22 = (g_2p - g_2m)/0.02
        new_response = (R11+R22)/2

        return new_response

    def mesh_response_master_cat(d, mdet_mom, snmin, snmax, steps, sizemin, sizemax):
        
        g1p_indexx, g1p_indexy, mask_1p = find_assign_grid(d, '1p', mdet_mom, snmin, snmax, steps, sizemin, sizemax)
        g1p_count = find_bincount_2d(g1p_indexx, g1p_indexy, steps)
        g1m_indexx, g1m_indexy, mask_1m = find_assign_grid(d, '1m', mdet_mom, snmin, snmax, steps, sizemin, sizemax)
        g1m_count = find_bincount_2d(g1m_indexx, g1m_indexy, steps)
        g2p_indexx, g2p_indexy, mask_2p = find_assign_grid(d, '2p', mdet_mom, snmin, snmax, steps, sizemin, sizemax)
        g2p_count = find_bincount_2d(g2p_indexx, g2p_indexy, steps)
        g2m_indexx, g2m_indexy, mask_2m = find_assign_grid(d, '2m', mdet_mom, snmin, snmax, steps, sizemin, sizemax)
        g2m_count = find_bincount_2d(g2m_indexx, g2m_indexy, steps)

        g_1p = np.zeros((steps, steps))
        g_1m = np.zeros((steps, steps))
        g_2p = np.zeros((steps, steps))
        g_2m = np.zeros((steps, steps))
        np.add.at(g_1p, (g1p_indexx, g1p_indexy), d[mask_1p][mdet_mom+'_g_1'])
        np.add.at(g_1m, (g1m_indexx, g1m_indexy), d[mask_1m][mdet_mom+'_g_1'])
        np.add.at(g_2p, (g2p_indexx, g2p_indexy), d[mask_2p][mdet_mom+'_g_2'])
        np.add.at(g_2m, (g2m_indexx, g2m_indexy), d[mask_2m][mdet_mom+'_g_2'])
        g_1p /= g1p_count
        g_1m /= g1m_count
        g_2p /= g2p_count
        g_2m /= g2m_count

        R11 = (g_1p - g_1m)/0.02
        R22 = (g_2p - g_2m)/0.02
        new_response = (R11+R22)/2

        return new_response

    save_data = True
    count_all = np.zeros((steps,steps))
    m = np.zeros((steps, steps))
    
    # f = open(mdet_tilename_filepath, 'r')
    # fs = f.read().split('\n')[:-1]
    # filenames = [fname.split('/')[-1] for fname in fs]
    # filenames = glob.glob(mdet_input_filepaths)
    # patch_names = [str(num).zfill(4) for num in range(200)]
    res = {'g_1p': np.zeros((steps, steps)),
           'g_1m': np.zeros((steps, steps)),
           'g_2p': np.zeros((steps, steps)),
           'g_2m': np.zeros((steps, steps)),
           'g1p_count': np.zeros((steps, steps)),
           'g1m_count': np.zeros((steps, steps)),
           'g2p_count': np.zeros((steps, steps)),
           'g2m_count': np.zeros((steps, steps))}
    
    # Accumulate raw sums of shear and mean shear corrected with response per tile. 
    total_count = 0
    total_count += len(d['noshear']['ra'])
    
    # Since we set upper limits on S/N (<1000) and Tratio (<3.0) which are not considered in cuts_and_save_catalogs.py, we need to make additional cuts here. 
    # d = d[((d[mdet_mom+'_s2n'] < snmax) & (d[mdet_mom+'_T_ratio'] < sizemax))]
    
    # mask_noshear = (d['mdet_step'] == 'noshear')
    mastercat_noshear_snr = d['noshear'][mdet_mom+'_s2n']
    mastercat_noshear_Tr = d['noshear'][mdet_mom+'_T_ratio']
    new_e1 = d['noshear'][mdet_mom+'_g_1']
    new_e2 = d['noshear'][mdet_mom+'_g_2']
    
    # Need raw sums of shear for shear response. 
    res = accumulate_shear_per_tile(res, d, mdet_mom, snmin, snmax, steps, sizemin, sizemax)
    new_indexx,new_indexy = assign_loggrid(mastercat_noshear_snr, mastercat_noshear_Tr, snmin, snmax, steps, sizemin, sizemax, steps)
    new_count = np.zeros((steps, steps))
    np.add.at(new_count,(new_indexx,new_indexy), 1)
    np.add.at(count_all,(), new_count)
    print(new_count, count_all)
    np.add.at(m,(new_indexx,new_indexy), (new_e1**2+new_e2**2)/2) # RMS of shear isn't corrected for the response. 
    # new_meanes = mesh_average(new_means, np.sqrt((new_e1**2+new_e2**2)/2),new_indexx,new_indexy,steps,new_count)

    H, xedges, yedges = np.histogram2d(mastercat_noshear_snr, mastercat_noshear_Tr, bins=[np.logspace(log10(snmin),log10(snmax),steps+1), np.logspace(log10(sizemin),log10(sizemax),steps+1)])
    # new_response = mesh_response_master_cat(d[msk], snmin, snmax, steps, sizemin, sizemax)
    new_response = compute_mesh_response(res)
    new_meanes = np.sqrt(m/count_all)
    new_shearweight = (new_response/new_meanes)**2


    res_measurement = {'xedges': xedges, 'yedges': yedges, 'count': count_all, 'meanes': new_meanes, 'response': new_response, 'weight': new_shearweight}

    with open(shear_wgt_output_filepath, 'wb') as dat:
        pickle.dump(res_measurement, dat, protocol=pickle.HIGHEST_PROTOCOL)

    print('total number count before cuts', total_count)
    print('total number count after cuts', np.sum(count_all))

def read_mdet_h5(datafile, keys, response=False, subtract_mean_shear=False):

    def _get_shear_weights(dat):
        return 1/(0.21**2 + 0.5*(np.array(dat['gauss_g_cov_1_1']) + np.array(dat['gauss_g_cov_2_2'])))
    def _wmean(q,w):
        return np.sum(q*w)/np.sum(w)
    
    mdet_steps = ['noshear', '1p', '1m', '2p', '2m']
    d_out = {}
    import h5py as h5
    f = h5.File(datafile, 'r')
    for mdet_step in mdet_steps:
        d = f.get('/mdet/'+mdet_step)
        nrows = len(np.array( d['ra'] ))
        formats = []
        for key in keys:
            formats.append('f4')
        data = np.recarray(shape=(nrows,), formats=formats, names=keys)
        for key in keys:  
            data[key] = np.array(d[key])
        d_out[mdet_step] = data
        print('made recarray with hdf5 file')
    
    # response correction
    if response:
        d_2p = f.get('/mdet/2p')
        d_1p = f.get('/mdet/1p')
        d_2m = f.get('/mdet/2m')
        d_1m = f.get('/mdet/1m')
        # compute response with weights
        g1p = _wmean(np.array(d_1p["gauss_g_1"]), _get_shear_weights(d_1p))                                     
        g1m = _wmean(np.array(d_1m["gauss_g_1"]), _get_shear_weights(d_1m))
        R11 = (g1p - g1m) / 0.02

        g2p = _wmean(np.array(d_2p["gauss_g_2"]), _get_shear_weights(d_2p))
        g2m = _wmean(np.array(d_2m["gauss_g_2"]), _get_shear_weights(d_2m))
        R22 = (g2p - g2m) / 0.02

        R = (R11 + R22)/2.
        data['g1'] /= R
        data['g2'] /= R

    # mean_g1 = _wmean(data['g1'], data['w'])
    # mean_g2 = _wmean(data['g2'], data['w'])
    # std_g1 = np.var(data['g1'])
    # std_g2 = np.var(data['g2'])
    # mean_shear = [mean_g1, mean_g2, std_g1, std_g2]
    # mean shear subtraction
    if subtract_mean_shear:
        print('subtracting mean shear')
        print('mean g1 g2 =(%1.8f,%1.8f)'%(mean_g1, mean_g2))          
        data['g1'] -= mean_g1
        data['g2'] -= mean_g2

    return d_out

def main(argv):

    mdet_input_filepaths = sys.argv[1]
    shear_wgt_output_filepath = sys.argv[2]
    mdet_mom = sys.argv[3]
    mdet_cuts = int(sys.argv[4])
    steps = int(sys.argv[5])
    snmin=int(sys.argv[6])
    snmax=int(sys.argv[7])
    if mdet_mom in ['pgauss', 'pgauss_reg0.90']:
        sizemin=0.5
        sizemax=3
    elif mdet_mom == 'wmom':
        sizemin=1.2
        sizemax=2.0 # to-do; check the histogram of Tratio to determine this value.
    elif mdet_mom == 'gauss':
        sizemin = float(sys.argv[8])
        sizemax = float(sys.argv[9])
    
    keys = ['ra', 'dec',  mdet_mom+'_g_1',  mdet_mom+'_g_2',  mdet_mom+'_s2n', mdet_mom+'_T_ratio']
    gal_data = read_mdet_h5(mdet_input_filepaths, keys, response=False, subtract_mean_shear=False)
    inverse_variance_weight(gal_data, shear_wgt_output_filepath, mdet_mom, steps, snmin, snmax, sizemin, sizemax, mdet_cuts)

if __name__ == "__main__":
    main(sys.argv)