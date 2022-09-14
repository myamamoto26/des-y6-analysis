import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob
import os,sys
import fitsio as fio
import matplotlib as mpl
from tqdm import tqdm
import proplot as pplt
import pickle
import pandas as pd
import smatch

d_gold = fio.read('/global/cscratch1/sd/myamamot/gold/gold_stars_all.fits')
d_pgauss_g = fio.read('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/mdet_stats_pgauss_gonly.fits')
d_pgauss_riz = fio.read('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/mdet_stats_pgauss_riz_1267tiles.fits')
d_pgauss_griz = fio.read('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/mdet_stats_pgauss_griz_1267tiles.fits')

f = open('/global/cfs/cdirs/des/y6-shear-catalogs/metadetection/mdet_files_1267tiles.txt', 'r')
fs_riz = f.read().split('\n')[:-1]
tilenames_riz = np.array([fname.split('_')[0] for fname in fs_riz])
f = open('/global/cfs/cdirs/des/y6-shear-catalogs/metadetection/gband-only/mdet_files.txt', 'r')
fs_g = f.read().split('\n')[:-1]
tilenames_g = np.array([fname.split('_')[0] for fname in fs_g])

msk = np.in1d(tilenames_riz, tilenames_g)
matched_filename_riz = np.array(fs_riz)[msk]
msk = np.in1d(tilenames_g, tilenames_riz)
matched_filename_g = np.array(fs_g)[msk]

mdet_files_g = []
mdet_files_riz = []
for fname_g, fname_riz in zip(matched_filename_g, matched_filename_riz):
    fn = os.path.join('/global/cscratch1/sd/myamamot/metadetect/cuts_final/g/pgauss', fname_g)
    if os.path.exists(fn):
        mdet_files_g.append(fn)
    fn = os.path.join('/global/cscratch1/sd/myamamot/metadetect/cuts_final/riz/pgauss', fname_riz)
    if os.path.exists(fn):
        mdet_files_riz.append(fn)
    
print('there are ', len(mdet_files_g), ' to be processed.')

def save_result(res, d, mdet_mom, start, end):
    res['ra'][start:end] = d['ra']
    res['psfrec_g_1'][start:end] = d['psfrec_g_1']
    res['psfrec_g_2'][start:end] = d['psfrec_g_2']
    res['psfrec_T'][start:end] = d['psfrec_T']
    res[mdet_mom+'_s2n'][start:end] = d[mdet_mom+'_s2n']
    res[mdet_mom+'_T'][start:end] = d[mdet_mom+'_T']
    res[mdet_mom+'_T_ratio'][start:end] = d[mdet_mom+'_T_ratio']
    res[mdet_mom+'_g_1'][start:end] = d[mdet_mom+'_g_1']
    res[mdet_mom+'_g_2'][start:end] = d[mdet_mom+'_g_2']
    res['dec'][start:end] = d['dec']
    res[mdet_mom+'_g_cov_1_1'][start:end] = d[mdet_mom+'_g_cov_1_1']
    res[mdet_mom+'_g_cov_2_2'][start:end] = d[mdet_mom+'_g_cov_2_2']
    res[mdet_mom+'_T_err'][start:end] = d[mdet_mom+'_T_err']

    # mag_g = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_g"])
    # mag_r = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_r"])
    # mag_i = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_i"])
    # mag_z = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_z"])
    # gmr = mag_g - mag_r
    # rmi = mag_r - mag_i
    # imz = mag_i - mag_z
    # res['g-r'][start:end] = gmr
    # res['r-i'][start:end] = rmi
    # res['i-z'][start:end] = imz
    
    return res

nside = 4096
maxmatch = 1
radius = 0.263/3600 # degrees
start_g = 0
start_g_star = 0
start_griz = 0
start_riz = 0
start_riz_star = 0
mdet_mom = 'pgauss'
res_griz = np.zeros(100000000, dtype=[('ra', float), ('psfrec_g_1', float), ('psfrec_g_2', float), ('psfrec_T', float), (mdet_mom+'_s2n', float), (mdet_mom+'_T', float), (mdet_mom+'_T_ratio', float), (mdet_mom+'_g_1', float), (mdet_mom+'_g_2', float), ('dec', float), (mdet_mom+'_g_cov_1_1', float), (mdet_mom+'_g_cov_2_2', float), ('g-r', float), ('r-i', float), ('i-z', float), (mdet_mom+'_T_err', float)])
res_riz = np.zeros(100000000, dtype=[('ra', float), ('psfrec_g_1', float), ('psfrec_g_2', float), ('psfrec_T', float), (mdet_mom+'_s2n', float), (mdet_mom+'_T', float), (mdet_mom+'_T_ratio', float), (mdet_mom+'_g_1', float), (mdet_mom+'_g_2', float), ('dec', float), (mdet_mom+'_g_cov_1_1', float), (mdet_mom+'_g_cov_2_2', float), ('g-r', float), ('r-i', float), ('i-z', float), (mdet_mom+'_T_err', float)])
res_riz_star = np.zeros(100000000, dtype=[('ra', float), ('psfrec_g_1', float), ('psfrec_g_2', float), ('psfrec_T', float), (mdet_mom+'_s2n', float), (mdet_mom+'_T', float), (mdet_mom+'_T_ratio', float), (mdet_mom+'_g_1', float), (mdet_mom+'_g_2', float), ('dec', float), (mdet_mom+'_g_cov_1_1', float), (mdet_mom+'_g_cov_2_2', float), ('g-r', float), ('r-i', float), ('i-z', float), (mdet_mom+'_T_err', float)])
res_g = np.zeros(100000000, dtype=[('ra', float), ('psfrec_g_1', float), ('psfrec_g_2', float), ('psfrec_T', float), (mdet_mom+'_s2n', float), (mdet_mom+'_T', float), (mdet_mom+'_T_ratio', float), (mdet_mom+'_g_1', float), (mdet_mom+'_g_2', float), ('dec', float), (mdet_mom+'_g_cov_1_1', float), (mdet_mom+'_g_cov_2_2', float), ('g-r', float), ('r-i', float), ('i-z', float), (mdet_mom+'_T_err', float)])
res_g_star = np.zeros(100000000, dtype=[('ra', float), ('psfrec_g_1', float), ('psfrec_g_2', float), ('psfrec_T', float), (mdet_mom+'_s2n', float), (mdet_mom+'_T', float), (mdet_mom+'_T_ratio', float), (mdet_mom+'_g_1', float), (mdet_mom+'_g_2', float), ('dec', float), (mdet_mom+'_g_cov_1_1', float), (mdet_mom+'_g_cov_2_2', float), ('g-r', float), ('r-i', float), ('i-z', float), (mdet_mom+'_T_err', float)])

num_total_g = 0
num_match_g = 0
num_total_riz = 0
num_match_riz = 0
for i, f_g, f_riz in tqdm(enumerate(zip(mdet_files_g, mdet_files_riz))):
    if i%15 == 0:
        print(i, num_total_g, num_match_g, num_total_riz, num_match_riz)
    d_g = fio.read(f_g)
    d_g = d_g[d_g['mdet_step'] == 'noshear']
    num_total_g += len(d_g)
    d_riz = fio.read(f_riz)
    d_riz = d_riz[d_riz['mdet_step'] == 'noshear']
    num_total_riz += len(d_riz)
    print(f_g.split('/')[-1].split('_')[0], f_riz.split('/')[-1].split('_')[0])
    
    # match with g-band and gold stars
    matches = smatch.match(d_g['ra'], d_g['dec'], radius, d_gold['RA'], d_gold['DEC'], nside=nside, maxmatch=maxmatch)
    d_g_match = d_g[matches['i1']]
    num_match_g += len(d_g_match)
    print('stellar contamination in g-band catalog', len(d_g_match))
    
    nomatch_msk = np.in1d(np.arange(len(d_g)), matches['i1'], invert=True)
    d_g_nomatch = d_g[nomatch_msk]
    
    # save stars that were identified as galaxies.
    end_g_star = start_g_star + len(d_g_match)
    res_g_star = save_result(res_g_star, d_g_match, mdet_mom, start_g_star, end_g_star)
    start_g_star = end_g_star
    # save pure g-band galaxies. 
    end_g = start_g + len(d_g_nomatch)
    res_g = save_result(res_g, d_g_nomatch, mdet_mom, start_g, end_g)
    start_g = end_g
    
    
    # match with riz-band and gold stars
    matches = smatch.match(d_riz['ra'], d_riz['dec'], radius, d_gold['RA'], d_gold['DEC'], nside=nside, maxmatch=maxmatch)
    d_riz_match = d_riz[matches['i1']]
    num_match_riz += len(d_riz_match)
    print('stellar contamination in riz-band catalog', len(d_riz_match))
    
    nomatch_msk = np.in1d(np.arange(len(d_riz)), matches['i1'], invert=True)
    d_riz_nomatch = d_riz[nomatch_msk]
    
    # save stars that were identified as galaxies.
    end_riz_star = start_riz_star + len(d_riz_match)
    res_riz_star = save_result(res_riz_star, d_riz_match, mdet_mom, start_riz_star, end_riz_star)
    start_riz_star = end_riz_star
    # save pure riz-band galaxies. 
    end_riz = start_riz + len(d_riz_nomatch)
    res_riz = save_result(res_riz, d_riz_nomatch, mdet_mom, start_riz, end_riz)
    start_riz = end_riz

