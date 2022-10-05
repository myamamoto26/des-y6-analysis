
import fitsio as fio
import numpy as np
import os, sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from rho_stats import measure_rho, measure_tau, write_stats, write_stats_tau, plot_overall_rho, plot_overall_tao

def measure_rho_tau_stats(do_rho, do_tau, max_sep, stats_name, mdet_input_flat, stats_out_dir):

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
    Example) /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3_snmax1000.pkl

    stats_out_dir: 
    Example) /global/cscratch1/sd/myamamot/metadetect/rho_tau_stats
    """

    good_piffs_table = fio.read('/project/projectdirs/des/schutt20/catalogs/y6a2_piff_v3_allres_v3_collated.fits')

    # rho-stats -> Just need to pass the piff catalog.
    if do_rho: 
        print('Computing rho-stats...')
        max_mag = 0
        name = stats_name 
        tag = 'griz'
        stats = measure_rho(good_piffs_table, max_sep, max_mag, subtract_mean=True, do_rho0=True)
        stat_file = os.path.join(stats_out_dir, "rho_%s_%s.json"%(name, tag))
        write_stats(stat_file,*stats)
    else:
        print('Skipping rho-stats. ')

    # tau-stats. 
    if do_tau: 
        print('Computing tau-stats...')
        max_mag = 0
        name = stats_name
        tag = 'griz'
        # Now it uses flat catalog created in make_flag_catalog.py
        stats = measure_tau(good_piffs_table, max_sep, max_mag, mdet_input_flat, stats_out_dir, subtract_mean=True)
        stat_file = os.path.join(stats_out_dir, "tau_%s_%s.json"%(name, tag))
        write_stats_tau(stat_file,*stats)
    else:
        print('Skipping tau-stats. ')

def main(argv):
    do_rho = sys.argv[1]
    do_tau = sys.argv[2]
    max_sep = int(sys.argv[3])
    stats_name = sys.argv[4]
    mdet_input_flat = sys.argv[5] 
    stats_out_dir = sys.argv[6]

    measure_rho_tau_stats(do_rho, do_tau, max_sep, stats_name, mdet_input_flat, stats_out_dir)    

if __name__ == "__main__":
    main(sys.argv)
