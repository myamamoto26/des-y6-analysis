import fitsio as fio
import numpy as np
import glob
import os, sys
import pickle
from tqdm import tqdm
from des_y6utils import mdet

def flux2mag(flux, zero_pt=30):
    return zero_pt - 2.5 * np.log10(flux)

def _save_measurement_info(mdet_files, outpath, stats_file, mdet_cuts, mdet_mom, add_cuts=False): 
    """
    Make a flat catalog that contains information only needed to produce mean shear vs properties plot.
    """

    res = np.zeros(200000000, dtype=[('ra', float), ('psfrec_g_1', float), ('psfrec_g_2', float), ('psfrec_T', float), (mdet_mom+'_s2n', float), (mdet_mom+'_T', float), (mdet_mom+'_T_ratio', float), ('pgauss_T', float), ('gmi', float)])
    
    start = 0
    for f in tqdm(mdet_files):
        d = fio.read(f)
        msk = mdet.make_mdet_cuts(d, mdet_cuts) 
        d = d[msk]

        d = d[d['mdet_step'] == 'noshear']
        if add_cuts:
            for cut in add_cuts:
                if cut == mdet_mom+'_s2n':
                    d = d[d[cut] < 200]
                elif cut == mdet_mom+'_T_ratio':
                    d = d[d[cut] > 1.5]
                elif cut == 'nepoch_g':
                    d = d[d[cut] > 4]
        end = start+len(d)

        res['ra'][start:end] = d['ra']
        res['psfrec_g_1'][start:end] = d['psfrec_g_1']
        res['psfrec_g_2'][start:end] = d['psfrec_g_2']
        res['psfrec_T'][start:end] = d['psfrec_T']
        res['pgauss_T'][start:end] = d['pgauss_T']
        res[mdet_mom+'_s2n'][start:end] = d[mdet_mom+'_s2n']
        res[mdet_mom+'_T'][start:end] = d[mdet_mom+"_T_ratio"]*d[mdet_mom+"_psf_T"]
        res[mdet_mom+'_T_ratio'][start:end] = d[mdet_mom+'_T_ratio']
        gmi = flux2mag(d['pgauss_band_flux_g']) - flux2mag(d['pgauss_band_flux_i'])
        imz = flux2mag(d['pgauss_band_flux_i']) - flux2mag(d['pgauss_band_flux_z'])
        res['gmi'][start:end] = gmi
        # res['imz'][start:end] = imz

        start = end

    # remove zero entry
    res = res[res['ra'] != 0]
    print('number of objects ', len(res))
    fio.write(os.path.join(outpath, stats_file), res)

def _compute_bins(stats_file, outpath, bin_file, nperbin):
    """
    Compute the bin edges and mean from the flat catalog made by _save_measurement_info. 
    """

    from esutil import stat

    bin_dict = {}
    d = fio.read(os.path.join(outpath, stats_file))
    for col in list(d.dtype.names)[1:]:
        prop = d[col]
        hist = stat.histogram(prop, nperbin=nperbin, more=True)
        bin_num = len(hist['hist'])
        print('number of bins', bin_num, 'in ', col)

        bin_dict[col] = hist

    with open(os.path.join(outpath, bin_file), 'wb') as handle:
        pickle.dump(bin_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return bin_dict

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
            g1 = res['noshear'][bin][0] / res['num_noshear'][bin][0]
            g1p = res['1p'][bin][0] / res['num_1p'][bin][0]
            g1m = res['1m'][bin][0] / res['num_1m'][bin][0]
            R11 = (g1p - g1m) / 2 / 0.01

            g2 = res['noshear'][bin][1] / res['num_noshear'][bin][1]
            g2p = res['2p'][bin][1] / res['num_2p'][bin][1]
            g2m = res['2m'][bin][1] / res['num_2m'][bin][1]
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

def _accum_shear_per_tile(res, g_step, g1, g2, g_qa, bin_low, bin_high, binnum, weight):
    
    for step in ['noshear', '1p', '1m', '2p', '2m']:
        msk_s = np.where(g_step == step)[0]
        qa_masked = g_qa[msk_s]
        g1_masked = g1[msk_s]*weight[msk_s] #- mean_shear_color['mean_g1'][msk_s]
        g2_masked = g2[msk_s]*weight[msk_s] #- mean_shear_color['mean_g2'][msk_s]
        
        for bin in range(binnum):
            msk_bin = np.where(((qa_masked >= bin_low[bin]) & (qa_masked <= bin_high[bin])))[0]
            np.add.at(
                res[step], 
                (bin, 0), 
                np.sum(g1_masked[msk_bin]),
            )
            np.add.at(
                res[step], 
                (bin, 1), 
                np.sum(g2_masked[msk_bin]),
            )
            np.add.at(
                res["num_" + step], 
                (bin, 0), 
                np.sum(weight[msk_s][msk_bin]),
            )
            np.add.at(
                res["num_" + step], 
                (bin, 1), 
                np.sum(weight[msk_s][msk_bin]),
            )
    
    return res

def _accum_shear_all(res, binnum):

    # Sum all the raw sums in each tile. 
    res_accum = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
            '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
            '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
            '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
            '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
    for tilename in res.keys():
        for step in ['noshear', '1p', '1m', '2p', '2m']:
            
            for bin in range(binnum):
                np.add.at(
                    res_accum[step], 
                    (bin, 0), 
                    res[tilename][step][bin][0],
                )
                np.add.at(
                    res_accum[step], 
                    (bin, 1), 
                    res[tilename][step][bin][1],
                )
                np.add.at(
                    res_accum["num_" + step], 
                    (bin, 0), 
                    res[tilename]["num_" + step][bin][0],
                )
                np.add.at(
                    res_accum["num_" + step], 
                    (bin, 1), 
                    res[tilename]["num_" + step][bin][1],
                )
    return res_accum

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

def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
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



def function(input_, mdet_cuts, binnum, mdet_mom, wgt_file, bins, outpath, weights='shape_err'):

    [key,pname,fname,bins,binnum] = input_
   
    d = fio.read(fname)
    msk = mdet.make_mdet_cuts(d, mdet_cuts) 
    # msk = mdet._make_mdet_cuts_gauss(d, n_terr=3) # if you need max_t cut, add it here. max_t = 0.689 (top 25% cut) for gauss, 0.466 for pgauss. 
    d = d[msk]
    
    
    ## ADD ADDITIONAL CUTS HERE. (e.g., size, color selections)
    # color splits: blue-[-2.00, 0.76], mid-[0.76, 1.49], red-[1.49, 4.00]
    # size splits: small-[0.095, 0.301], midsize-[0.301, 0.455], large-[0.454, 8000]
    # size S/N splits: bad-[1.63e-4, 7.30], ok-[7.30, 15], good-[15-35361]
    dcut = flux2mag(d['pgauss_band_flux_g']) - flux2mag(d['pgauss_band_flux_i'])
    # dcut = (d['gauss_T_ratio'] * d["gauss_psf_T"]) * d['gauss_T_err']
    # dcut2 = (d['gauss_T_ratio'] * d["gauss_psf_T"])/d['gauss_T_err']
    dmin = 1.49
    dmax = 4.00
    d = d[((dcut > dmin) & (dcut < dmax))] 
    # d = d[((dcut < 1) | (dcut2 > 10))] 
    
    
    res = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
            '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
            '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
            '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
            '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
    if weights == 's2n_sizer':
        with open(os.path.join(outpath, wgt_file), 'rb') as handle:
            wgt_dict = pickle.load(handle)
        shear_wgt = _find_shear_weight(d, wgt_dict, wgt_dict['xedges'][0], wgt_dict['xedges'][-1], wgt_dict['yedges'][0], wgt_dict['yedges'][-1], len(wgt_dict['xedges'])-1, mdet_mom)
    elif weights == 'shape_err':
        shear_wgt = 1/(0.17**2 + 0.5*(d[mdet_mom+'_g_cov_1_1'] + d[mdet_mom+'_g_cov_2_2']))
    
    if key not in ['gmi', 'imz', 'gauss_T']:
        res = _accum_shear_per_tile(res, d['mdet_step'], d[mdet_mom+'_g_1'], d[mdet_mom+'_g_2'], d[key], bins['low'], bins['high'], binnum, shear_wgt)
    elif key == 'gauss_T':
        gaussT = d[mdet_mom+"_T_ratio"]*d[mdet_mom+"_psf_T"]
        res = _accum_shear_per_tile(res, d['mdet_step'], d[mdet_mom+'_g_1'], d[mdet_mom+'_g_2'], gaussT, bins['low'], bins['high'], binnum, shear_wgt)
    else:
        color = flux2mag(d['pgauss_band_flux_'+key[0]]) - flux2mag(d['pgauss_band_flux_'+key[2]])
        res = _accum_shear_per_tile(res, d['mdet_step'], d[mdet_mom+'_g_1'], d[mdet_mom+'_g_2'], color, bins['low'], bins['high'], binnum, shear_wgt)

    output_fpath = os.path.join(outpath, '{0}_{1}'.format(key,pname)+'.pickle')
    with open(output_fpath, 'wb') as fp:
        pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)

def main(argv):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank, size)

    stats_file = sys.argv[2]
    bin_file = sys.argv[3]
    mdet_mom = sys.argv[4]
    outpath = sys.argv[5]
    nperbin = int(sys.argv[6])
    measurement_file = sys.argv[7]
    wgt_file=sys.argv[8]
    mdet_cuts = int(sys.argv[9])
    weight_scheme = sys.argv[10]

    mdet_files = sorted(glob.glob(sys.argv[1]))
    if not os.path.exists(os.path.join(outpath, stats_file)):
        if rank == 0:
            print('creating flat and bin file. ')
            _save_measurement_info(mdet_files, outpath, stats_file, mdet_cuts, mdet_mom) 
            bin_dict = _compute_bins(stats_file, outpath, bin_file, nperbin)
    comm.Barrier()
    print('finished binning up...')
    with open(os.path.join(outpath, bin_file), 'rb') as handle:
        bin_dict = pickle.load(handle)

    patch = True
    if patch:
        fids = [fname.split('/')[-1][6:10] for fname in mdet_files]
    else:
        fids = [fname.split('/')[-1].split('_')[0] for fname in mdet_files]
    
    runs = []
    for key in list(bin_dict.keys()):
        for pname,fname in zip(fids, mdet_files):
            bins = bin_dict[key]
            binnum = len(bins['hist'])
            runs.append([key,pname,fname,bins,binnum])

    if len(measurement_file.split('/')) == 2:
        outpath2 = os.path.join(outpath, measurement_file.split('/')[0])
    else:
        outpath2 = outpath
    for i in range(len(runs)):
        if i % size != rank:
            continue
        if i % 100 == 0:
            print('made it to ', i)
        function(runs[i], mdet_cuts, binnum, mdet_mom, wgt_file, bins, outpath2, weights=weight_scheme)
    comm.Barrier()

    # compute jackknife errors by leaving one tile/patch out for each rank. 
    measurement_result = {}
    for key in list(bin_dict.keys()):
        print(key)
        bins = bin_dict[key]
        binnum = len(bins['hist'])
        res_all = {}
        res_jk_mean = {} 
        if rank == 0:
            fpath = os.path.join(outpath2, f"""{key}_*.pickle""")
            res_files = sorted(glob.glob(fpath))
            for id,fname in zip(fids, res_files):
                with open(fname, 'rb') as f:
                    d = pickle.load(f)
                res_all[id] = d
                
            # copmute mean shear by combining all the files.
            res_accum = _accum_shear_all(res_all, binnum)
            print('number of objects is: ', np.sum(res_accum['num_noshear'], axis=0))
            # Compute the mean g1 and g2 over all the tiles. 
            res_all_mean = _compute_g1_g2(res_accum, binnum)
        res_all = comm.bcast(res_all, root=0)
        comm.Barrier()
        print('computing jackknife covariance...')
        for sample,pname in tqdm(enumerate(res_all.keys())):
            res_jk = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
                        '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
                        '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
                        '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
                        '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
            # pname = fids[sample]
            if sample % size != rank:
                continue
            jk_sample_mean = _compute_shear_per_jksample(res_jk, res_all, pname, fids, binnum)
            res_jk_mean[sample] = jk_sample_mean
        # comm.Barrier()
        # if rank != 0:
        #     comm.send(res_jk_mean, dest=0)
        # comm.Barrier()
        # if rank == 0:
        #     for i in tqdm(range(1,size)):
        #         tmp_res = comm.recv(source=i)
        #         res_jk_mean.update(tmp_res)
        # comm.Barrier()

        # Write out res_jk_mean in pickle file.
        with open(os.path.join(outpath, 'res_jk_mean_'+str(rank)+'.pickle'), 'wb') as handle:
            pickle.dump(res_jk_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)
        comm.Barrier()
        # Read in res_jk_mean pickled files.
        if rank == 0:
            for i in tqdm(range(1,size)):
                with open(os.path.join(outpath, 'res_jk_mean_'+str(i)+'.pickle'), 'rb') as handle:
                    tmp_res = pickle.load(handle)
                res_jk_mean.update(tmp_res)
        comm.Barrier()
        print('saving result...')
        if rank == 0:
            # Compute jackknife error estimate.
            jk_error = _compute_jackknife_error_estimate(res_jk_mean, binnum, len(fids))
            measurement_result[key] = {'bin_mean': bins['mean'], 'g1': res_all_mean[:,0], 'g2': res_all_mean[:,1], 'g1_cov': jk_error[:,0], 'g2_cov': jk_error[:,1]}
    
    if rank == 0:
        with open(os.path.join(outpath, measurement_file), 'wb') as handle:
            pickle.dump(measurement_result, handle, protocol=pickle.HIGHEST_PROTOCOL) 

if __name__ == "__main__":
    main(sys.argv)