
import fitsio as fio
import numpy as np
from matplotlib import pyplot as plt
import os, sys
import time
from scipy import stats
import meds
from esutil import stat
import esutil as eu
from scipy.optimize import curve_fit
from tqdm import tqdm
import json
import time
import pickle
import matplotlib
# from ../metadetect_mask import exclude_gold_mask_objects

def _make_cuts(d, shear, additional_cuts=None):

    if additional_cuts is None:
        # msk = (
        #     (d['flags'] == 0)
        #     & (d['mdet_s2n'] > 10)
        #     & (d['mdet_T_ratio'] > 1.2)
        #     & (d['mfrac'] < 0.1)
        #     & (d['mdet_step'] == shear)
        # )
        msk = (d['MDET_STEP'] == shear)
    else:
        # if additional_cuts['quantity'] == 'PSFREC_G':
        #     msk = (
        #         (d[additional_cuts['quantity']+'_'+additional_cuts['e']] > additional_cuts['cuts'][0])
        #         & (d[additional_cuts['quantity']+'_'+additional_cuts['e']] < additional_cuts['cuts'][1])
        #         & (d['MDET_STEP'] == shear)
        #     )
        # else:
        msk = (
            (d[additional_cuts['quantity']] > additional_cuts['cuts'][0])
            & (d[additional_cuts['quantity']] < additional_cuts['cuts'][1])
            & (d['MDET_STEP'] == shear)
        )
    mean_shear = [np.mean(d['MDET_G_1'][msk], axis=0), np.mean(d['MDET_G_2'][msk], axis=0)]
    mean_shear_err = bootstrap_sampling_error(200, d['MDET_G_1'][msk], d['MDET_G_2'][msk])
    len_cuts = len(d['MDET_G_1'][msk])
    return mean_shear, mean_shear_err, len_cuts

def _compute_response(d, additional_cuts=None):

    g_noshear, gerr_noshear, len_noshear = _make_cuts(d, 'noshear', additional_cuts=additional_cuts)
    g_1p, gerr_1p, len_1p = _make_cuts(d, '1p', additional_cuts=additional_cuts)
    g_1m, gerr_1m, len_1m = _make_cuts(d, '1m', additional_cuts=additional_cuts)
    g_2p, gerr_2p, len_2p = _make_cuts(d, '2p', additional_cuts=additional_cuts)
    g_2m, gerr_2m, len_2m = _make_cuts(d, '2m', additional_cuts=additional_cuts)

    R11 = (g_1p[0] - g_1m[0])/0.02
    R22 = (g_2p[1] - g_2m[1])/0.02
    R = [R11, R22]

    return R, g_noshear, gerr_noshear, len_noshear

def _compute_bootstrap_error_estimate(N, data1, data2):

    # data1 = data_set[:, 0]
    # data2 = data_set[:, 1]

    fi = []
    for n in range(N):
        sample1 = np.random.choice(np.arange(len(data1)),len(data1),replace=True)
        fi.append(np.mean(data1[sample1]))
    f_mean = np.sum(fi)/N 
    fi = np.array(fi)
    cov1 = np.sqrt(np.sum((fi-f_mean)**2)/(N-1))

    fi = []
    for n in range(N):
        sample2 = np.random.choice(np.arange(len(data2)),len(data2),replace=True)
        fi.append(np.mean(data2[sample2]))
    f_mean = np.sum(fi)/N 
    fi = np.array(fi)
    cov2 = np.sqrt(np.sum((fi-f_mean)**2)/(N-1))
    
    return [cov1, cov2]

def _compute_jackknife_error_estimate(res_jk_mean, res_all_mean, binnum, N):

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

def _compute_OLSfit(x, y, dy=None):
    """Find the best fitting parameters of a linear fit to the data through the
    method of ordinary least squares estimation. (i.e. find m and b for
    y = m*x + b)

    Args:
        x: Numpy array of independent variable data
        y: Numpy array of dependent variable data. Must have same size as x.
        dy: Numpy array of dependent variable standard deviations. Must be same
            size as y.

    Returns: A list with four floating point values. [m, dm, b, db]
    """
    if dy is None:
        #if no error bars, weight every point the same
        dy = np.ones(x.size)
    denom = np.sum(1 / dy**2) * np.sum((x / dy)**2) - (np.sum(x / dy**2))**2
    m = (np.sum(1 / dy**2) * np.sum(x * y / dy**2) -
         np.sum(x / dy**2) * np.sum(y / dy**2)) / denom
    b = (np.sum(x**2 / dy**2) * np.sum(y / dy**2) -
         np.sum(x / dy**2) * np.sum(x * y / dy**2)) / denom
    dm = np.sqrt(np.sum(1 / dy**2) / denom)
    db = np.sqrt(np.sum(x / dy**2) / denom)
    return [m, dm, b, db]


def mdet_shear_pairs_plotting_percentile(d, nperbin, cut_quantity):

    ## for linear fit
    def func(x,m,n):
        return m*x+n
    
    ## psf shape/area vs mean shear. 
    fig,axs = plt.subplots(5,2,figsize=(36,20))
    perc = [99,99,98,98,95,95,90,90,85,85]
    # def_mask = ((d['flags'] == 0) & (d['mdet_s2n'] > 10) & (d['mdet_T_ratio'] > 1.2) & (d['mfrac'] < 0.1))
    for q,ax in enumerate(axs.ravel()):
        d_max = np.percentile(d[cut_quantity], perc[q], axis=0)
        d_mask = (d[cut_quantity] < d_max)
        d_quant = d[cut_quantity][d_mask]
        d = d[d_mask]
        hist = stat.histogram(d_quant, nperbin=nperbin, more=True)
        bin_num = len(hist['hist'])
        g_obs = np.zeros(bin_num)
        gerr_obs = np.zeros(bin_num)
        for i in tqdm(range(bin_num)):
            additional_cuts = {'quantity': cut_quantity, 'cuts': [hist['low'][i], hist['high'][i]]}
            print(additional_cuts)
            R, g_mean, gerr_mean, bs = calculate_response(d, additional_cuts=additional_cuts)
            g_obs[i] = g_mean[q%2]/R[q%2]
            gerr_obs[i] = (gerr_mean[q%2]/R[q%2])
        
        params = curve_fit(func,hist['mean'],g_obs,p0=(0.,0.))
        m1,n1=params[0]
        x = np.linspace(hist['mean'][0], hist['mean'][bin_num-1], 100)

        ax.plot(x, func(x,m1,n1), label='linear fit w/ fit params: m='+str("{:2.4f}".format(m1))+', b='+str("{:2.4f}".format(n1)))
        ax.errorbar(hist['mean'], g_obs, yerr=gerr_obs, fmt='o', fillstyle='none', label=str(100-perc[q])+'percent cut, Tmax='+str("{:2.2f}".format(d_max)))
        # if q%2 == 0:
        #     ax.set_ylim(-3e-3, 5e-3)
        # elif q%2 == 1:
        #     ax.set_ylim(-1e-2, 2e-2)
        ax.set_xlabel("e1,PSF")
        ax.set_ylabel('<e'+str(q%2 + 1)+'>')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.tick_params(labelsize=15)
        ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('mdet_shear_e1cuts_v2.pdf', bbox_inches='tight')

def mdet_shear_pairs_plotting(d, nperbin):

    ## for linear fit
    def func(x,m,n):
        return m*x+n
    
    ## psf shape/area vs mean shear. 
    fig,axs = plt.subplots(3,2,figsize=(30,17))
    # exclude objects in healpix which is the same as the gold. 
    d = exclude_gold_mask_objects(d)
    d = exclude_hyperleda_objects(d)
    # fio.write('mdet_exclude_y3goldmask_and_hyperleda.fits', d)
    for q,ax in enumerate(axs.ravel()):
        if q==0 or q==1:
            psf_ = d['PSFREC_G_'+str(q+1)]
            hist = stat.histogram(psf_, nperbin=nperbin, more=True)
            bin_num = len(hist['hist'])
        elif q==2 or q==3:
            Tpsf = d['PSFREC_T']
            hist = stat.histogram(Tpsf, nperbin=nperbin, more=True)
            bin_num = len(hist['hist'])
        elif q==4 or q==5:
            Tr = d['MDET_T_RATIO'] # d['mdet_T'][T_mask]
            hist = stat.histogram(Tr, nperbin=2000000, more=True)
            bin_num = len(hist['hist'])
        g_obs = np.zeros(bin_num)
        gerr_obs = np.zeros(bin_num)
        print(len(hist['low']), len(hist['mean']))
        for i in tqdm(range(bin_num)):
            if q==0 or q==1:
                additional_cuts = {'quantity': 'PSFREC_G_'+str(q+1), 'cuts': [hist['low'][i], hist['high'][i]]}
            elif q==2 or q==3:
                additional_cuts = {'quantity': 'PSFREC_T', 'cuts': [hist['low'][i], hist['high'][i]]}
            elif q==4 or q==5:
                additional_cuts = {'quantity': 'MDET_T_RATIO', 'cuts': [hist['low'][i], hist['high'][i]]}
                print(i, hist['low'][i], hist['high'][i])

            R, g_mean, gerr_mean, bs = calculate_response(d, additional_cuts=additional_cuts)
            g_obs[i] = g_mean[q%2]/R[q%2]
            gerr_obs[i] = (gerr_mean[q%2]/R[q%2])
        
        # params = curve_fit(func,hist['mean'],g_obs,p0=(0.,0.))
        # m1,n1=params[0]
        params = OLSfit(hist['mean'], g_obs, dy=gerr_obs)
        x = np.linspace(hist['mean'][0], hist['mean'][bin_num-1], 100)
        print('parameters of the fit. ', params)

        ax.plot(x, func(x,params[0],params[2]), label='linear fit w/ fit params: m='+str("{:2.4f}".format(params[0]))+'+/-'+str(("{:2.4f}".format(params[1])))+', b='+str("{:2.4f}".format(params[2]))+'+/-'+str(("{:2.4f}".format(params[3]))))
        ax.errorbar(hist['mean'], g_obs, yerr=gerr_obs, fmt='o', fillstyle='none', label='Y6 metadetect test')
        if q==0 or q==1:
            ax.set_xlabel('e'+str(q+1)+',PSF', fontsize=20)
        elif q==2 or q==3:
            ax.set_xlabel(r'$T_{PSF}$', fontsize=20)
        elif q==4 or q==5:
            ax.set_xlabel(r'$T_{ratio}$', fontsize=20)
        ax.set_ylabel('<e'+str(q%2 + 1)+'>', fontsize=20)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.tick_params(labelsize=16)
    axs[0,0].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('mdet_psf_vs_shear_weightedfit_v2_y3goldmask_and_hyperleda.pdf', bbox_inches='tight')

def plot_null_tests(d, nperbin, x):

    def func(x,m,n):
        return m*x+n
    
    ## psf shape/area vs mean shear. 
    fig,axs = plt.subplots(2,1,figsize=(18,12))
    # exclude objects in healpix which is the same as the gold. 
    # d = exclude_hyperleda_objects(d)
    prop = d[x]
    prop = prop[prop < 1000]
    hist = stat.histogram(prop, nperbin=nperbin, more=True)
    bin_num = len(hist['hist'])
    g_obs = np.zeros(bin_num)
    gerr_obs = np.zeros(bin_num)
    print(len(hist['low']), len(hist['mean']))
    for i in tqdm(range(bin_num)):
        additional_cuts = {'quantity': x, 'cuts': [hist['low'][i], hist['high'][i]]}
        R, g_mean, gerr_mean, bs = calculate_response(d, additional_cuts=additional_cuts)
        print(i, hist['low'][i], hist['high'][i], bs)

    for q,ax in enumerate(axs.ravel()):
        g_obs[i] = g_mean[q]/R[q]
        gerr_obs[i] = (gerr_mean[q]/R[q])
        
        params = curve_fit(func,hist['mean'],g_obs,p0=(0.,0.))
        m1,n1=params[0]
        x = np.linspace(hist['mean'][0], hist['mean'][bin_num-1], 100)
        print('parameters of the fit. ', m1, n1)

        ax.plot(x, func(x,m1,n1), label='linear fit w/ fit params: m='+str("{:2.4f}".format(m1))+', b='+str("{:2.4f}".format(n1)))
        ax.errorbar(hist['mean'], g_obs, yerr=gerr_obs, fmt='o', fillstyle='none', label='Y6 metadetect test')
        ax.set_xlabel('S/N', fontsize=20)
        ax.set_xscale('log')
        ax.set_ylabel(r"$\langle e'+str(q+1)+'\rangle$", fontsize=20)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.tick_params(labelsize=16)
    axs[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('mdet_psf_vs_shear_fit_v2_SNR_1000.pdf', bbox_inches='tight')


def plot_null_tests2(fs, predef_bin, qa):

    def func(x,m,n):
        return m*x+n

    def _compute_g1_g2(res, binnum, method='all', tile=None):

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
                print('shear response: ', R11, R22)

            elif method == 'all':
                g1 = res['all']['noshear'][bin][0] / res['all']['num_noshear'][bin][0]
                g1p = res['all']['1p'][bin][0] / res['all']['num_1p'][bin][0]
                g1m = res['all']['1m'][bin][0] / res['all']['num_1m'][bin][0]
                R11 = (g1p - g1m) / 2 / 0.01

                g2 = res['all']['noshear'][bin][1] / res['all']['num_noshear'][bin][1]
                g2p = res['all']['2p'][bin][1] / res['all']['num_2p'][bin][1]
                g2m = res['all']['2m'][bin][1] / res['all']['num_2m'][bin][1]
                R22 = (g2p - g2m) / 2 / 0.01
                print('shear response: ', R11, R22)

            corrected_g1g2[bin, 0] = g1/R11
            corrected_g1g2[bin, 1] = g2/R22
        return corrected_g1g2

    def _accum_shear_per_tile(res, tilename, g_step, g, g_qa, bin_low, bin_high, binnum):
        
        for step in ['noshear', '1p', '1m', '2p', '2m']:
            msk_s = np.where(g_step == step)[0]
            qa_masked = g_qa[msk_s]
            g_masked = g[msk_s]
            
            for bin in range(binnum):
                msk_bin = np.where(((qa_masked >= bin_low[bin]) & (qa_masked <= bin_high[bin])))[0]
                np.add.at(
                    res[tilename][step], 
                    (bin, 0), 
                    np.sum(g_masked[msk_bin,0]),
                )
                np.add.at(
                    res[tilename][step], 
                    (bin, 1), 
                    np.sum(g_masked[msk_bin,1]),
                )
                np.add.at(
                    res[tilename]["num_" + step], 
                    (bin, 0), 
                    len(g_masked[msk_bin,0]),
                )
                np.add.at(
                    res[tilename]["num_" + step], 
                    (bin, 1), 
                    len(g_masked[msk_bin,1]),
                )
        
        return res
    
    def _compute_shear_per_jksample(res_jk, res, tilename, tilenames, binnum):
        
        for t in tilenames:
            if t == tilename:
                continue
            else:
                for step in ['noshear', '1p', '1m', '2p', '2m']:
                    
                    for bin in range(binnum):
                        np.add.at(
                            res_jk[step], 
                            (bin, 0), 
                            res[tilename][step][bin][0],
                        )
                        np.add.at(
                            res_jk[step], 
                            (bin, 1), 
                            res[tilename][step][bin][1],
                        )
                        np.add.at(
                            res_jk["num_" + step], 
                            (bin, 0), 
                            res[tilename]["num_" + step][bin][0],
                        )
                        np.add.at(
                            res_jk["num_" + step], 
                            (bin, 1), 
                            res[tilename]["num_" + step][bin][1],
                        )
        jk_sample_mean = _compute_g1_g2(res_jk, binnum, method='jk')
        return jk_sample_mean

    def _accum_shear_all(res, tilename, binnum):

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

    def _accum_shear_per_tile_without_bin(res, tilename, g_step, g, bin):

        for step in ['noshear', '1p', '1m', '2p', '2m']:
            msk_s = np.where(g_step == step)[0]
            
            np.add.at(
                res[tilename][step], 
                (bin, 0), 
                np.sum(g[msk_s,0]),
            )
            np.add.at(
                res[tilename][step], 
                (bin, 1), 
                np.sum(g[msk_s,1]),
            )
            np.add.at(
                res[tilename]["num_" + step], 
                (bin, 0), 
                len(g[msk_s,0]),
            )
            np.add.at(
                res[tilename]["num_" + step], 
                (bin, 1), 
                len(g[msk_s,1]),
            )
        return res
    
    res = {}
    res_tile_mean = {}
    num_objects = 0
    binnum = len(predef_bin['hist'])
    filenames = [fname.split('/')[-1] for fname in fs]
    tilenames = [d.split('_')[0] for d in filenames] 
    for fname in tqdm(filenames):
        mdet_all = fio.read(os.path.join('/global/cscratch1/sd/myamamot/metadetect', fname))
        msk_default = ((mdet_all['flags']==0) & (mdet_all['mdet_s2n']>10) & (mdet_all['mfrac']<0.02) & (mdet_all['mdet_T_ratio']>1.2) & (mdet_all['mask_flags']==0))
        mdet = mdet_all[msk_default]
        num_objects += len(mdet)
        res[fname.split('_')[0]] = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
                                '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
                                '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
                                '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
                                '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
        res = _accum_shear_per_tile(res, fname.split('_')[0], mdet['mdet_step'], mdet['mdet_g'], mdet[qa], predef_bin['low'], predef_bin['high'], binnum)
        # res = _accum_shear_per_tile_without_bin(res, fname.split('_')[0], mdet['mdet_step'], mdet['mdet_g'], 0)
        # res_mean = _compute_g1_g2(res, binnum, method='tile', tile=fname.split('_')[0])
        tile_mean = _compute_g1_g2(res, binnum, method='tile', tile=fname.split('_')[0])
        res_tile_mean[fname.split('_')[0]] = tile_mean

    # Accumulate all the tiles shears. 
    res['all'] = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
                  '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
                  '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
                  '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
                  '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
    for fname in tqdm(filenames):
        res = _accum_shear_all(res, fname.split('_')[0], binnum)
    print(num_objects, np.sum(res['all']['num_noshear'][:,0])+np.sum(res['all']['num_1p'][:,0])+np.sum(res['all']['num_1m'][:,0])+np.sum(res['all']['num_2p'][:,0])+np.sum(res['all']['num_2m'][:,0]))

    # Compute the mean g1 and g2 over all the tiles. 
    res_all_mean = _compute_g1_g2(res, binnum)
    print(res['all'])
    print("mean shear over all tiles: ", res_all_mean)

    # Compute bootstrap sampling errors.
    N_boot = 500
    bt_size = len(list(res_tile_mean))
    bt_sample1_mean = np.zeros((binnum, N_boot))
    bt_sample2_mean = np.zeros((binnum, N_boot))
    for bin in range(binnum):
        for n in N_boot:
            bt_g1_bin = np.array([res_tile_mean[t][bin][0] for t in list(res_tile_mean)])
            bt_g2_bin = np.array([res_tile_mean[t][bin][1] for t in list(res_tile_mean)])
            bt_sample1 = np.random.choice(bt_g1_bin, size=bt_size, replace=True)
            bt_sample2 = np.random.choice(bt_g2_bin, size=bt_size, replace=True)
            bt_sample1_mean[bin, n] = np.mean(bt_sample1)
            bt_sample2_mean[bin, n] = np.mean(bt_sample2)
    bt_error = np.zeros((binnum, 2))
    for bin in range(binnum):
        bt_error[bin, 0] = np.std(bt_sample1_mean[bin])
        bt_error[bin, 1] = np.std(bt_sample2_mean[bin])
    print(bt_error)

    # Compute jackknife samples.
    res_jk_mean = {} 
    for sample, fname in tqdm(enumerate(filenames)):
        res_jk = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
                  '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
                  '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
                  '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
                  '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
        jk_sample_mean = _compute_shear_per_jksample(res_jk, res, fname.split('_')[0], tilenames, binnum)
        res_jk_mean[sample] = jk_sample_mean
    
    # Compute jackknife error estimate.
    jk_error = _get_jackknife_cov(res_jk_mean, res_all_mean, binnum, len(tilenames))
    print("jackknife error estimate: ", jk_error)

    sys.exit()
    fig,axs = plt.subplots(1,2,figsize=(28,10))
    for ii in range(2):
        # params = curve_fit(func,predef_bin['mean'],res_all_mean[:,ii],p0=(0.,0.))
        # m1,n1=params[0]
        # params = OLSfit(predef_bin['mean'], res_all_mean[:,ii], dy=jk_error[:,ii])
        # x = np.linspace(predef_bin['mean'][0], predef_bin['mean'][binnum-1], 100)
        # print('parameters of the fit. ', params)

        # axs[ii].plot(x, func(x,params[0],params[2]), label='linear fit w/ fit params: m='+str("{:2.4f}".format(params[0]))+', b='+str("{:2.4f}".format(params[1])))
        axs[ii].errorbar(predef_bin['mean'], res_all_mean[:,ii], fmt='o', fillstyle='none', label='Y6 metadetect test')
        axs[ii].set_xlabel(r"$T_{ratio}$", fontsize=20)
        # axs[ii].set_xscale('log')
        axs[ii].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axs[ii].tick_params(labelsize=16)
    axs[0].set_ylabel(r"$\langle e_{1} \rangle$", fontsize=20)
    axs[1].set_ylabel(r"$\langle e_{2} \rangle$", fontsize=20)
    axs[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('mdet_psf_vs_shear_fit_v3_Tratio.pdf', bbox_inches='tight')