
import fitsio as fio
import numpy as np
import os, sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from astropy.io import fits
from rho_stats import measure_rho, measure_tau, write_stats, write_stats_tau, plot_overall_rho, plot_overall_tao

def measure_rho_tau_stats(do_rho, do_tau, min_sep, max_sep, stats_name, mdet_input_flat, stats_out_dir, error_method):

    """
    Computes rho-statistics from PIFF catalogs and tau-statistics from PIFF and metadetection catalogs.

    Parameters
    ----------
    do_rho: whether or not do rho-stats (Boolean)
    do_tau: whether or not do tau-stats(Boolean)
    max_sep: maximum separation of the correlation function
    Example) 60

    stats_name: name of the statistics computed
    Example) all_JK

    mdet_input_flat: the file path where the flat catalog is stored

    stats_out_dir: 
    Example) /global/cscratch1/sd/myamamot/metadetect/rho_tau_stats
    """

    good_piffs_table = fits.open('/pscratch/sd/s/schutt20/y6a2_piff_testing/catalogs/y6a2_piff_v3_allres_v3_taustat_input_v5.fits')
    f_pc = '/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed8888.fits'
    # rho-stats -> Just need to pass the piff catalog.
    if do_rho: 
        print('Computing rho-stats...')
        max_mag = 0
        name = stats_name 
        tag = 'riz'
        stats = measure_rho(good_piffs_table, min_sep, max_sep, max_mag, stats_out_dir, f_pc, error_method=error_method, subtract_mean=True, do_rho0=True)
        stat_file = os.path.join(stats_out_dir, "rho_%s_%s.json"%(name, tag))
        write_stats(stat_file,*stats)
    else:
        print('Skipping rho-stats. ')

    # tau-stats. 
    if do_tau: 
        print('Computing tau-stats...')
        max_mag = 0
        name = stats_name
        tag = 'riz'
        # Now it uses flat catalog created in make_flag_catalog.py
        stats = measure_tau(good_piffs_table, min_sep, max_sep, max_mag, mdet_input_flat, stats_out_dir, f_pc, error_method=error_method, subtract_mean_psf=True, subtract_mean_shear=True)
        stat_file = os.path.join(stats_out_dir, "tau_%s_%s.json"%(name, tag))
        write_stats_tau(stat_file,*stats)
    else:
        print('Skipping tau-stats. ')

def main(argv):
    do_rho = eval(sys.argv[1])
    do_tau = eval(sys.argv[2])
    min_sep = float(sys.argv[3])
    max_sep = float(sys.argv[4])
    stats_name = sys.argv[5]
    mdet_input_flat = sys.argv[6] 
    stats_out_dir = sys.argv[7]
    error_method = sys.argv[8]

    measure_rho_tau_stats(do_rho, do_tau, min_sep, max_sep, stats_name, mdet_input_flat, stats_out_dir, error_method)    

if __name__ == "__main__":
    main(sys.argv)
