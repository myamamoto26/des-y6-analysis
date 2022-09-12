import fitsio as fio
import numpy as np
import glob
import os, sys
from tqdm import tqdm

def _save_stats_g(mdet_files, mdet_mom, outpath, stats_file, add_cuts=None):

    res = np.zeros(200000000, dtype=[('ra', float), ('psfrec_g_1', float), ('psfrec_g_2', float), ('psfrec_T', float), (mdet_mom+'_s2n', float), (mdet_mom+'_T', float), (mdet_mom+'_T_ratio', float), (mdet_mom+'_g_1', float), (mdet_mom+'_g_2', float), ('dec', float), (mdet_mom+'_g_cov_1_1', float), (mdet_mom+'_g_cov_2_2', float), (mdet_mom+'_T_err', float), (mdet_mom+"_band_flux_g", float), ('nepoch_eff_g', int), ('nepoch_g', int)])

    start = 0
    for f in tqdm(mdet_files):
        d = fio.read(f)
        d = d[d['mdet_step'] == 'noshear']
        if add_cuts:
            for cut in add_cuts:
                if cut == mdet_mom+'_s2n':
                    d = d[d[cut] < 200]
                elif cut == mdet_mom+'_T_ratio':
                    d = d[d[cut] > 1.5]
        end = start+len(d)

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
        res[mdet_mom+'_band_flux_g'][start:end] = d[mdet_mom+'_band_flux_g']
        res['nepoch_eff_g'][start:end] = d['nepoch_eff_g']
        res['nepoch_g'][start:end] = d['nepoch_g']

        start = end

    # remove zero entry
    res = res[res['ra'] != 0]
    print('number of objects ', len(res))
    fio.write(os.path.join(outpath, stats_file), res)

def _save_stats(mdet_files, mdet_mom, outpath, stats_file, add_cuts=None):

    res = np.zeros(200000000, dtype=[('ra', float), ('psfrec_g_1', float), ('psfrec_g_2', float), ('psfrec_T', float), (mdet_mom+'_s2n', float), (mdet_mom+'_T', float), (mdet_mom+'_T_ratio', float), (mdet_mom+'_g_1', float), (mdet_mom+'_g_2', float), ('dec', float), (mdet_mom+'_g_cov_1_1', float), (mdet_mom+'_g_cov_2_2', float), ('g-r', float), ('r-i', float), ('i-z', float), (mdet_mom+'_T_err', float), (mdet_mom+"_band_flux_g", float), (mdet_mom+"_band_flux_r", float), (mdet_mom+"_band_flux_i", float), (mdet_mom+"_band_flux_z", float), ('nepoch_eff_g', int), ('nepoch_g', int)])

    start = 0
    for f in tqdm(mdet_files):
        d = fio.read(f)
        d = d[d['mdet_step'] == 'noshear']
        if add_cuts:
            for cut in add_cuts:
                if cut == mdet_mom+'_s2n':
                    d = d[d[cut] < 200]
                elif cut == mdet_mom+'_T_ratio':
                    d = d[d[cut] > 1.5]
        end = start+len(d)

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
        res[mdet_mom+'_band_flux_g'][start:end] = d[mdet_mom+'_band_flux_g']
        res[mdet_mom+'_band_flux_r'][start:end] = d[mdet_mom+'_band_flux_r']
        res[mdet_mom+'_band_flux_i'][start:end] = d[mdet_mom+'_band_flux_i']
        res[mdet_mom+'_band_flux_z'][start:end] = d[mdet_mom+'_band_flux_z']
        res['nepoch_eff_g'][start:end] = d['nepoch_eff_g']
        res['nepoch_g'][start:end] = d['nepoch_g']

        mag_g = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_g"])
        mag_r = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_r"])
        mag_i = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_i"])
        mag_z = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_z"])
        gmr = mag_g - mag_r
        rmi = mag_r - mag_i
        imz = mag_i - mag_z
        res['g-r'][start:end] = gmr
        res['r-i'][start:end] = rmi
        res['i-z'][start:end] = imz

        start = end

    # remove zero entry
    res = res[res['ra'] != 0]
    print('number of objects ', len(res))
    fio.write(os.path.join(outpath, stats_file), res)

def main(argv):

    mdet_input_filepaths = sys.argv[1]
    stats_file = sys.argv[2]
    shear_bands = sys.argv[3]
    mdet_mom = sys.argv[4]
    outpath = sys.argv[5]

    f = open(mdet_input_filepaths, 'r')
    fs = f.read().split('\n')[:-1]
    mdet_files = []
    for fname in fs:
        fn = os.path.join('/global/cscratch1/sd/myamamot/metadetect/cuts_final/'+shear_bands+'/'+mdet_mom, fname.split('/')[-1])
        if os.path.exists(fn):
            mdet_files.append(fn)
    print('there are ', len(mdet_files), ' to be processed.')

    if shear_bands == 'g':
        _save_stats_g(mdet_files, mdet_mom, outpath, stats_file)
    else:
        _save_stats(mdet_files, mdet_mom, outpath, stats_file)

if __name__ == "__main__":
    main(sys.argv)