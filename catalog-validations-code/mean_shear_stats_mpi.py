import fitsio as fio
import numpy as np
import glob
import os, sys
import pickle
from tqdm import tqdm
from des_y6utils import mdet
import h5py as h5

def read_mdet_h5(datafile, gal_weight_file, keys, mdet_step, patch_id=None, response=False, subtract_mean_shear=False, subtract_shear_color=False):

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

    def assign_grid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
        # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps
        
        xbins=np.linspace(xmin, xmax, xsteps+1)
        ybins=np.linspace(ymin, ymax, ysteps+1)

        indexx=np.digitize(x, xbins, right=True)
        indexy=np.digitize(y, ybins, right=True)

        indexx = indexx - 1
        indexy = indexy - 1

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
        elif key == 'imz':
            mag_i = mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_i"])[mask], 2)
            mag_z = mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_z"])[mask], 3)
            data[key] = mag_i - mag_z
        else:
            data[key] = np.array(d[key])[mask]
    # print('made recarray with hdf5 file')
    
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
    
    # option to subtract mean shear based on per-object color.
    if ((mdet_step == 'noshear') & subtract_shear_color):
        with open("/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v6/color_grid_gmi_imz.pickle", "rb") as f:
            color_grid = pickle.load(f)
            g1_color = color_grid['e1']/color_grid['count']
            g2_color = color_grid['e2']/color_grid['count']
        gmimin = -2.0; gmimax = 4.0; imzmin = -2.0; imzmax = 2.0; steps=20
        indexx, indexy = assign_grid(data['gmi'], data['imz'], gmimin, gmimax, steps, imzmin, imzmax, steps)
        mean_g1_color = np.array([g1_color[x, y] for x, y in zip(indexx, indexy)])
        mean_g2_color = np.array([g2_color[x, y] for x, y in zip(indexx, indexy)])
        data['g1'] -= mean_g1_color
        data['g2'] -= mean_g2_color

    return data

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

def _compute_bins_from_h5(gal_data, outpath, bin_file, nperbin):
    """
    Compute the bin edges and mean from the flat catalog made by _save_measurement_info. 
    """

    from esutil import stat

    bin_dict = {}
    d = gal_data
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

def _accum_shear_per_tile(res, dat, key, bin_low, bin_high, binnum):
    
    for i,step in enumerate(['noshear', '1p', '1m', '2p', '2m']):
        g1_masked = dat[i]['g1'] * dat[i]['w']
        g2_masked = dat[i]['g2'] * dat[i]['w']
        qa_masked = dat[i][key]
        
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
                np.sum(dat[i]['w'][msk_bin]),
            )
            np.add.at(
                res["num_" + step], 
                (bin, 1), 
                np.sum(dat[i]['w'][msk_bin]),
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



def function(input_, mdet_files, mdet_cuts, mdet_mom, wgt_file, outpath, weights='shape_err'):

    [key,pname,bins,binnum] = input_
    
    # if key == 'gmi':
    #     keys = ['g1', 'g2', 'w', 'imz', key]
    # elif key == 'imz':
    #     keys = ['g1', 'g2', 'w', 'gmi', key]
    # else:
    #     keys = ['g1', 'g2', 'w', 'gmi', 'imz', key]
    keys = ['g1', 'g2', 'w', 'gmi', key]
    subtract_color = False # option to subtract mean shear from color grid
    d = read_mdet_h5(mdet_files, wgt_file, keys, 'noshear', patch_id=pname, response=False, subtract_mean_shear=True, subtract_shear_color=subtract_color)
    d_1p = read_mdet_h5(mdet_files, wgt_file, keys, '1p', patch_id=pname, response=False, subtract_mean_shear=True, subtract_shear_color=subtract_color)
    d_1m = read_mdet_h5(mdet_files, wgt_file, keys, '1m', patch_id=pname, response=False, subtract_mean_shear=True, subtract_shear_color=subtract_color)
    d_2p = read_mdet_h5(mdet_files, wgt_file, keys, '2p', patch_id=pname, response=False, subtract_mean_shear=True, subtract_shear_color=subtract_color)
    d_2m = read_mdet_h5(mdet_files, wgt_file, keys, '2m', patch_id=pname, response=False, subtract_mean_shear=True, subtract_shear_color=subtract_color)
    d_all = [d, d_1p, d_1m, d_2p, d_2m]
    # msk = mdet._make_mdet_cuts_gauss(d, n_terr=3) # if you need max_t cut, add it here. max_t = 0.689 (top 25% cut) for gauss, 0.466 for pgauss. 
    
    ## ADD ADDITIONAL CUTS HERE. (e.g., size, color selections)
    # color splits: blue-[-2.00, 0.76], mid-[0.76, 1.49], red-[1.49, 4.00]
    # size splits: small-[0.095, 0.301], midsize-[0.301, 0.455], large-[0.454, 8000]
    # size S/N splits: bad-[1.63e-4, 7.30], ok-[7.30, 15], good-[15-35361]
    # dcut = flux2mag(d['pgauss_band_flux_g']) - flux2mag(d['pgauss_band_flux_i'])
    # dcut = (d['gauss_T_ratio'] * d["gauss_psf_T"]) * d['gauss_T_err']
    # dcut2 = (d['gauss_T_ratio'] * d["gauss_psf_T"])/d['gauss_T_err']
    dmin = 1.49
    dmax = 4.00
    for i,cat in enumerate(d_all):
        d_all[i] = cat[((cat['gmi'] > dmin) & (cat['gmi'] < dmax))] 
    
    
    res = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
            '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
            '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
            '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
            '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
    res = _accum_shear_per_tile(res, d_all, key, bins['low'], bins['high'], binnum)

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
    color_split = sys.argv[11]

    mdet_files = sys.argv[1]
    if not os.path.exists(os.path.join(outpath, bin_file)):
        if rank == 0:
            print('creating flat and bin file. ')
            keys = ['ra', 'psfrec_g_1', 'psfrec_g_2', 'psfrec_T', 'pgauss_T', 'gauss_psf_T', 'gauss_s2n', 'gauss_T_ratio', 'gauss_T', 'gmi', 'imz', 'mfrac']
            gal_data = read_mdet_h5(mdet_files, wgt_file, keys, 'noshear', response=False, subtract_mean_shear=True)
            bin_dict = _compute_bins_from_h5(gal_data, outpath, bin_file, nperbin)
            # _save_measurement_info(mdet_files, outpath, stats_file, mdet_cuts, mdet_mom) 
            # bin_dict = _compute_bins(stats_file, outpath, bin_file, nperbin)
    comm.Barrier()
    print('finished binning up...')
    with open(os.path.join(outpath, bin_file), 'rb') as handle:
        bin_dict = pickle.load(handle)

    patch_h5 = True
    if patch_h5:
        fids = np.arange(200)
    else:
        fids = [fname.split('/')[-1][6:10] for fname in mdet_files]
    
    runs = []
    if color_split:
        keys = ['psfrec_g_1', 'psfrec_g_2']
    else:
        keys = bin_dict.keys()
    for key in keys:
        for pname in fids:
            bins = bin_dict[key]
            binnum = len(bins['hist'])
            runs.append([key,pname,bins,binnum])

    if len(measurement_file.split('/')) == 2:
        outpath2 = os.path.join(outpath, measurement_file.split('/')[0])
    else:
        outpath2 = outpath

    for i in range(len(runs)):
        if i % size != rank:
            continue
        if i % 100 == 0:
            print('made it to ', i)
        function(runs[i], mdet_files, mdet_cuts, mdet_mom, wgt_file, outpath2, weights=weight_scheme)
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