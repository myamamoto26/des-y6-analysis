import fitsio as fio
import numpy as np
import glob
import os, sys
import pickle
from tqdm import tqdm

def _save_measurement_info(mdet_files, mdet_mom, outpath, stats_file): 
    """
    Make a flat catalog that contains information only needed to produce mean shear vs properties plot.
    """

    res = np.zeros(200000000, dtype=[(mdet_mom+'_psf_g_1', float), (mdet_mom+'_psf_g_2', float), (mdet_mom+'_psf_T', float), (mdet_mom+'_s2n', float), (mdet_mom+'_g_1', float), (mdet_mom+'_g_2', float), (mdet_mom+'_T', float), (mdet_mom+'_T_ratio', float)])

    start = 0
    for f in tqdm(mdet_files):
        d = fio.read(f)
        d = d[d['mdet_step'] == 'noshear']
        end = start+len(d)

        res[mdet_mom+'_psf_g_1'][start:end] = d[mdet_mom+'_psf_g_1']
        res[mdet_mom+'_psf_g_2'][start:end] = d[mdet_mom+'_psf_g_2']
        res[mdet_mom+'_psf_T'][start:end] = d[mdet_mom+'_psf_T']
        res[mdet_mom+'_s2n'][start:end] = d[mdet_mom+'_s2n']
        res[mdet_mom+'_g_1'][start:end] = d[mdet_mom+'_g_1']
        res[mdet_mom+'_g_2'][start:end] = d[mdet_mom+'_g_2']
        res[mdet_mom+'_T'][start:end] = d[mdet_mom+'_T']
        res[mdet_mom+'_T_ratio'][start:end] = d[mdet_mom+'_T_ratio']

        start = end

    # remove zero entry
    res = res[res['mdet_step'] != 0]
    print('number of objects ', len(res))
    fio.write(os.path.join(outpath, stats_file), res)

def _compute_bins(stats_file, outpath, bin_file, nperbin):
    """
    Compute the bin edges and mean from the flat catalog made by _save_measurement_info. 
    """

    from esutil import stat

    bin_dict = {}
    d = fio.read(os.path.join(outpath, stats_file))
    for col in list(d.dtype.names):
        prop = d[col]
        hist = stat.histogram(prop, nperbin=nperbin, more=True)
        bin_num = len(hist['hist'])
        print('number of bins', bin_num, 'in ', col)

        bin_dict[col] = hist

    with open(os.path.join(outpath, bin_file), 'wb') as handle:
        pickle.dump(bin_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def _compute_g1_g2(res, binnum, method='all', tile=None):

    # Compute mean shear with response. 
    # For 'jk' and 'tile', it is calculating mean shear for each jackknife sample/each tile. 
    # For 'all', it is computing mean shear from res['all'] which sums all the shears in all the catalogs. 

    corrected_g1g2 = np.zeros((binnum, 2))
    for bin in range(binnum):
        if method == 'jk':
            g1 = res['noshear'][bin][0] / res['num_noshear'][bin][0]
            g1p = res['1p'][bin][0] / res['num_1p'][bin][0]
            g1m = res['1m'][bin][0] / res['num_1m'][bin][0]
            R11 = (g1p - g1m) / 2 / 0.01

            g2 = res['noshear'][bin][1] / res['num_noshear'][bin][1]
            g2p = res['2p'][bin][1] / res['num_2p'][bin][1]
            g2m = res['2m'][bin][1] / res['num_2m'][bin][1]
            R22 = (g2p - g2m) / 2 / 0.01
        
        elif method == 'tile':
            g1 = res[tile]['noshear'][bin][0] / res[tile]['num_noshear'][bin][0]
            g1p = res[tile]['1p'][bin][0] / res[tile]['num_1p'][bin][0]
            g1m = res[tile]['1m'][bin][0] / res[tile]['num_1m'][bin][0]
            R11 = (g1p - g1m) / 2 / 0.01

            g2 = res[tile]['noshear'][bin][1] / res[tile]['num_noshear'][bin][1]
            g2p = res[tile]['2p'][bin][1] / res[tile]['num_2p'][bin][1]
            g2m = res[tile]['2m'][bin][1] / res[tile]['num_2m'][bin][1]
            R22 = (g2p - g2m) / 2 / 0.01

        elif method == 'all':
            g1 = res['all']['noshear'][bin][0] / res['all']['num_noshear'][bin][0]
            g1p = res['all']['1p'][bin][0] / res['all']['num_1p'][bin][0]
            g1m = res['all']['1m'][bin][0] / res['all']['num_1m'][bin][0]
            R11 = (g1p - g1m) / 2 / 0.01

            g2 = res['all']['noshear'][bin][1] / res['all']['num_noshear'][bin][1]
            g2p = res['all']['2p'][bin][1] / res['all']['num_2p'][bin][1]
            g2m = res['all']['2m'][bin][1] / res['all']['num_2m'][bin][1]
            R22 = (g2p - g2m) / 2 / 0.01

        corrected_g1g2[bin, 0] = g1/R11
        corrected_g1g2[bin, 1] = g2/R22
    return corrected_g1g2

def _compute_shear_per_jksample(res_jk, res, ith_tilename, tilenames, binnum):

    # Compute mean shear for each jackknife sample. 
    # For each jackknife sample, you leave one tile out, sums the shears in N-1 tiles, and compute the mean. 
    
    for t in tilenames:
        if t == ith_tilename:
            continue
        else:
            for step in ['noshear', '1p', '1m', '2p', '2m']:
                
                for bin in range(binnum):
                    np.add.at(
                        res_jk[step], 
                        (bin, 0), 
                        res[t][step][bin][0],
                    )
                    np.add.at(
                        res_jk[step], 
                        (bin, 1), 
                        res[t][step][bin][1],
                    )
                    np.add.at(
                        res_jk["num_" + step], 
                        (bin, 0), 
                        res[t]["num_" + step][bin][0],
                    )
                    np.add.at(
                        res_jk["num_" + step], 
                        (bin, 1), 
                        res[t]["num_" + step][bin][1],
                    )
    jk_sample_mean = _compute_g1_g2(res_jk, binnum, method='jk')
    return jk_sample_mean

def _accum_shear_per_tile(res, tilename, g_step, g1, g2, g_qa, bin_low, bin_high, binnum):
    
    for step in ['noshear', '1p', '1m', '2p', '2m']:
        msk_s = np.where(g_step == step)[0]
        qa_masked = g_qa[msk_s]
        g1_masked = g1[msk_s]
        g2_masked = g2[msk_s]
        
        for bin in range(binnum):
            msk_bin = np.where(((qa_masked >= bin_low[bin]) & (qa_masked <= bin_high[bin])))[0]
            np.add.at(
                res[tilename][step], 
                (bin, 0), 
                np.sum(g1_masked[msk_bin]),
            )
            np.add.at(
                res[tilename][step], 
                (bin, 1), 
                np.sum(g2_masked[msk_bin]),
            )
            np.add.at(
                res[tilename]["num_" + step], 
                (bin, 0), 
                len(g1_masked[msk_bin]),
            )
            np.add.at(
                res[tilename]["num_" + step], 
                (bin, 1), 
                len(g2_masked[msk_bin]),
            )
    
    return res

def _accum_shear_all(res, tilename, binnum):

    # Sum all the raw sums in each tile. 

    for step in ['noshear', '1p', '1m', '2p', '2m']:
        
        for bin in range(binnum):
            np.add.at(
                res['all'][step], 
                (bin, 0), 
                res[tilename][step][bin][0],
            )
            np.add.at(
                res['all'][step], 
                (bin, 1), 
                res[tilename][step][bin][1],
            )
            np.add.at(
                res['all']["num_" + step], 
                (bin, 0), 
                res[tilename]["num_" + step][bin][0],
            )
            np.add.at(
                res['all']["num_" + step], 
                (bin, 1), 
                res[tilename]["num_" + step][bin][1],
            )
    return res

def _compute_jackknife_error_estimate(res_jk_mean, binnum, N):

    jk_cov = np.zeros((binnum, 2))
    for bin in range(binnum):
        # compute jackknife average. 
        jk_g1_ave = np.array([res_jk_mean[sample][bin][0] for sample in list(res_jk_mean)])
        jk_all_g1_ave = np.mean(jk_g1_ave)
        jk_g2_ave = np.array([res_jk_mean[sample][bin][1] for sample in list(res_jk_mean)])
        jk_all_g2_ave = np.mean(jk_g2_ave)

        # cov_g1 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_g1_ave - res_all_mean[bin][0])**2))
        # cov_g2 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_g2_ave - res_all_mean[bin][1])**2))
        cov_g1 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_g1_ave - jk_all_g1_ave)**2))
        cov_g2 = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_g2_ave - jk_all_g2_ave)**2))

        jk_cov[bin, 0] = cov_g1
        jk_cov[bin, 1] = cov_g2

    return jk_cov

def compute_mean_shear(mdet_input_filepaths, stats_file, bin_file, mdet_mom, outpath, nperbin, measurement_file):

    """
    Computes mean shear in the bins of several PSF and galaxy properties.

    mdet_input_filepaths: The file paths to the metadetection catalogs
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3/*_metadetect-v5_mdetcat_part0000.fits

    stats_file: The file name that contains the statistics to measure mean shear in bins of the statistics
    bin_file:  The file name that contains the binning information for each statistic
    mdet_mom: The shape measurement estimator
    outpath: The folder that contains measurement information
    nperbin: The number of objects per bin
    measurement_file: The file that contains information to plot the mean shear vs several properties
    """

    mdet_files = glob.glob(mdet_input_filepaths)
    print('there are ', len(mdet_files), ' to be processed.')
    tilenames = [fname.split('/')[-1].split('_')[0] for fname in mdet_files]
    if not os.path.exists(os.path.join(outpath, stats_file)):
        print('creating flat file. ')
        _save_measurement_info(mdet_files, mdet_mom, outpath, stats_file)
    else:
        if not os.path.exists(os.path.join(outpath, bin_file)):
            print('creating bin file.')
            _compute_bins(stats_file, outpath, bin_file, nperbin)
        else:
            with open(os.path.join(outpath, bin_file), 'rb') as handle:
                bin_dict = pickle.load(handle)

        measurement_result = {}
        for key in list(bin_dict.keys()):
            bins = bin_dict[key]
            res = {} # dictionary to accumulate raw sums. 
            res_tile_mean = {} # dictionary to accumulate mean shear for each tile. 
            num_objects = 0
            binnum = len(bins['hist'])
            # Accumulate raw sums of shear and mean shear corrected with response per tile. 
            for fname in tqdm(mdet_files):
                d = fio.read(fname)
                num_objects += len(d)
                tilename = fname.split('/')[-1].split('_')[0]
                res[tilename] = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
                                        '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
                                        '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
                                        '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
                                        '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
                res = _accum_shear_per_tile(res, tilename, d['mdet_step'], d[mdet_mom+'_g_1'], d[mdet_mom+'_g_2'], d[key], bins['low'], bins['high'], binnum)
                tile_mean = _compute_g1_g2(res, binnum, method='tile', tile=tilename)
                res_tile_mean[tilename] = tile_mean

            # Accumulate all the tiles shears. 
            res['all'] = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
                        '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
                        '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
                        '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
                        '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
            for fname in tqdm(mdet_files):
                tilename = fname.split('/')[-1].split('_')[0]
                res = _accum_shear_all(res, tilename, binnum)
            print(num_objects)

            # Compute the mean g1 and g2 over all the tiles. 
            res_all_mean = _compute_g1_g2(res, binnum)
            print(res['all'])
            print("mean shear over all tiles: ", res_all_mean)

            # Compute jackknife samples.
            print('computing jackknife errors')
            res_jk_mean = {} 
            for sample, fname in tqdm(enumerate(mdet_files)):
                res_jk = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
                        '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
                        '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
                        '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
                        '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
                tilename = fname.split('/')[-1].split('_')[0]
                jk_sample_mean = _compute_shear_per_jksample(res_jk, res, tilename, tilenames, binnum)
                res_jk_mean[sample] = jk_sample_mean
            
            # Compute jackknife error estimate.
            jk_error = _compute_jackknife_error_estimate(res_jk_mean, res_all_mean, binnum, len(tilenames))
            print("jackknife error estimate: ", jk_error)

            measurement_result[key] = {'bin_mean': bins['mean'], 'g1': res_all_mean[:,0], 'g2': res_all_mean[:,1], 'g1_cov': jk_error[:,0], 'g2_cov': jk_error[:,1]}
        
        with open(os.path.join(outpath, measurement_file), 'wb') as handle:
            pickle.dump(measurement_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(argv):

    mdet_input_filepaths = sys.argv[1]
    stats_file = sys.argv[2]
    bin_file = sys.argv[3]
    mdet_mom = sys.argv[4]
    outpath = sys.argv[5]
    nperbin = int(sys.argv[6])
    measurement_file = sys.argv[7]

    compute_mean_shear(mdet_input_filepaths, stats_file, bin_file, mdet_mom, outpath, nperbin, measurement_file)
    
if __name__ == "__main__":
    main(sys.argv)
