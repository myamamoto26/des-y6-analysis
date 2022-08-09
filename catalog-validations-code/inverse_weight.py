from nis import match
import os, sys
from tokenize import group
from tqdm import tqdm
import numpy as np
import fitsio as fio
import matplotlib as mpl


def inverse_variance_weight(mdet_tilename_filepath, mdet_input_filepaths, shear_wgt_output_filepath, steps, snmin, snmax, sizemin, sizemax):

    """
    Returns galaxy count, shear response, variance of e, shear weight as a function of S/N and size ratio.

    Parameters
    ----------
    mdet_tilename_filepath: Text file of the list of filenames of the metadetection catalogs
    Example) /global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt

    mdet_input_filepaths: The file path to the directory in which the input metadetection catalogs exist
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3

    shear_wgt_output_filepath: The file path where the output pickle file is written
    Example) /global/cscratch1/sd/myamamot/metadetect/inverse_variance_weight_v3_Trcut_snmax1000.pickle

    steps: The bin number in S/N and size ratio
    snmin: The minimum S/N to be considered
    snmax: The maximum S/N to be considered
    sizemin: The minimum size ratio (T/Tpsf) to be considered
    sizemax: The maximum size ratio to be considered
    """

    import os
    np.random.seed(1738)
    import matplotlib.pyplot as plt
    from math import log10
    import pylab as mplot
    import matplotlib.ticker as ticker
    import pickle

    f = open(mdet_tilename_filepath, 'r')
    fs = f.read().split('\n')[:-1]

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

    def find_assign_grid(d, mdet_step, snmin, snmax, steps, sizemin, sizemax):

        mask = (d['mdet_step']==mdet_step)
        mastercat_snr = d[mask]['mdet_s2n']
        mastercat_Tr = d[mask]['mdet_T_ratio']
        new_indexx,new_indexy = assign_loggrid(mastercat_snr, mastercat_Tr, snmin, snmax, steps, sizemin, sizemax, steps)
        
        return new_indexx, new_indexy, mask

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

    def accumulate_shear_per_tile(res, d, snmin, snmax, steps, sizemin, sizemax):

        g1p_indexx, g1p_indexy, mask_1p = find_assign_grid(d, '1p', snmin, snmax, steps, sizemin, sizemax)
        g1p_count = find_bincount_2d(g1p_indexx, g1p_indexy, steps)
        g1m_indexx, g1m_indexy, mask_1m = find_assign_grid(d, '1m', snmin, snmax, steps, sizemin, sizemax)
        g1m_count = find_bincount_2d(g1m_indexx, g1m_indexy, steps)
        g2p_indexx, g2p_indexy, mask_2p = find_assign_grid(d, '2p', snmin, snmax, steps, sizemin, sizemax)
        g2p_count = find_bincount_2d(g2p_indexx, g2p_indexy, steps)
        g2m_indexx, g2m_indexy, mask_2m = find_assign_grid(d, '2m', snmin, snmax, steps, sizemin, sizemax)
        g2m_count = find_bincount_2d(g2m_indexx, g2m_indexy, steps)

        np.add.at(res['g_1p'], (g1p_indexx, g1p_indexy), d[mask_1p]['mdet_g_1'])
        np.add.at(res['g_1m'], (g1m_indexx, g1m_indexy), d[mask_1m]['mdet_g_1'])
        np.add.at(res['g_2p'], (g2p_indexx, g2p_indexy), d[mask_2p]['mdet_g_2'])
        np.add.at(res['g_2m'], (g2m_indexx, g2m_indexy), d[mask_2m]['mdet_g_2'])
        
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

    def mesh_response_master_cat(d, snmin, snmax, steps, sizemin, sizemax):
        
        g1p_indexx, g1p_indexy, mask_1p = find_assign_grid(d, '1p', snmin, snmax, steps, sizemin, sizemax)
        g1p_count = find_bincount_2d(g1p_indexx, g1p_indexy, steps)
        g1m_indexx, g1m_indexy, mask_1m = find_assign_grid(d, '1m', snmin, snmax, steps, sizemin, sizemax)
        g1m_count = find_bincount_2d(g1m_indexx, g1m_indexy, steps)
        g2p_indexx, g2p_indexy, mask_2p = find_assign_grid(d, '2p', snmin, snmax, steps, sizemin, sizemax)
        g2p_count = find_bincount_2d(g2p_indexx, g2p_indexy, steps)
        g2m_indexx, g2m_indexy, mask_2m = find_assign_grid(d, '2m', snmin, snmax, steps, sizemin, sizemax)
        g2m_count = find_bincount_2d(g2m_indexx, g2m_indexy, steps)

        g_1p = np.zeros((steps, steps))
        g_1m = np.zeros((steps, steps))
        g_2p = np.zeros((steps, steps))
        g_2m = np.zeros((steps, steps))
        np.add.at(g_1p, (g1p_indexx, g1p_indexy), d[mask_1p]['mdet_g_1'])
        np.add.at(g_1m, (g1m_indexx, g1m_indexy), d[mask_1m]['mdet_g_1'])
        np.add.at(g_2p, (g2p_indexx, g2p_indexy), d[mask_2p]['mdet_g_2'])
        np.add.at(g_2m, (g2m_indexx, g2m_indexy), d[mask_2m]['mdet_g_2'])
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
    
    filenames = [fname.split('/')[-1] for fname in fs]
    tilenames = [d.split('_')[0] for d in filenames]
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
    for fname in tqdm(filenames):
        fp = os.path.join(mdet_input_filepaths, fname)
        if os.path.exists(fp):
            d = fio.read(fp)
        else:
            continue
        total_count += len(d[d['mdet_step']=='noshear'])
        
        # Since we set upper limits on S/N (<1000) and Tratio (<3.0) which are not considered in cuts_and_save_catalogs.py, we need to make additional cuts here. 
        d = d[((d['mdet_s2n'] < snmax) & (d['mdet_T_ratio'] < sizemax))]
        
        mask_noshear = (d['mdet_step'] == 'noshear')
        mastercat_noshear_snr = d[mask_noshear]['mdet_s2n']
        mastercat_noshear_Tr = d[mask_noshear]['mdet_T_ratio']
        new_e1 = d[mask_noshear]['mdet_g_1']
        new_e2 = d[mask_noshear]['mdet_g_2']
        
        # Need raw sums of shear for shear response. 
        res = accumulate_shear_per_tile(res, d, snmin, snmax, steps, sizemin, sizemax)
        new_indexx,new_indexy = assign_loggrid(mastercat_noshear_snr, mastercat_noshear_Tr, snmin, snmax, steps, sizemin, sizemax, steps)
        new_count = np.zeros((steps, steps))
        np.add.at(new_count,(new_indexx,new_indexy), 1)
        np.add.at(count_all,(), new_count)
        np.add.at(m,(new_indexx,new_indexy), np.sqrt((new_e1**2+new_e2**2)/2)) # RMS of shear isn't corrected for the response. 
        # new_meanes = mesh_average(new_means, np.sqrt((new_e1**2+new_e2**2)/2),new_indexx,new_indexy,steps,new_count)

    H, xedges, yedges = np.histogram2d(mastercat_noshear_snr, mastercat_noshear_Tr, bins=[np.logspace(log10(snmin),log10(snmax),steps+1), np.logspace(log10(sizemin),log10(sizemax),steps+1)])
    # new_response = mesh_response_master_cat(d[msk], snmin, snmax, steps, sizemin, sizemax)
    new_response = compute_mesh_response(res)
    new_meanes = m/count_all
    new_shearweight = (new_response/new_meanes)**2

    res_measurement = {'xedges': xedges, 'yedges': yedges, 'count': count_all, 'meanes': new_meanes, 'response': new_response, 'weight': new_shearweight}

    with open(shear_wgt_output_filepath, 'wb') as dat:
        pickle.dump(res_measurement, dat, protocol=pickle.HIGHEST_PROTOCOL)

    print('total number count before cuts', total_count)
    print('total number count after cuts', np.sum(count_all))


def main(argv):

    mdet_tilename_filepath = sys.argv[1]
    mdet_input_filepaths = sys.argv[2]
    shear_wgt_output_filepath = sys.argv[3]
    steps = 20
    snmin=10
    snmax=1000
    sizemin=0.5
    sizemax=3
    
    inverse_variance_weight(mdet_tilename_filepath, mdet_input_filepaths, shear_wgt_output_filepath, steps, snmin, snmax, sizemin, sizemax)

if __name__ == "__main__":
    main(sys.argv)