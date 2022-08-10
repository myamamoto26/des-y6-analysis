
import fitsio as fio
import numpy as np
import os, sys
from tqdm import tqdm
from matplotlib import pyplot as plt
import emcee
import pickle
import glob
from rho_stats import measure_rho, measure_tau, write_stats, write_stats_tau, plot_overall_rho, plot_overall_tao

def measure_rho_tau_stats(do_rho, do_tau, max_sep, stats_name, response_filepath, mdet_input_filepaths, stats_out_dir):

    """
    Computes rho-statistics from PIFF catalogs and tau-statistics from PIFF and metadetection catalogs.

    Parameters
    ----------
    do_rho: whether or not do rho-stats (Boolean)
    do_tau: whether or not do tau-stats(Boolean)
    max_sep: maximum separation of the correlation function
    Example) 60

    stats_name:
    Example) all_JK

    response_filepath: 
    Example) /global/cscratch1/sd/myamamot/metadetect/shear_response_v3.txt

    mdet_input_filepaths: 
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3/*_metadetect-v5_mdetcat_part0000.fits

    stats_out_dir: 
    Example) /global/cscratch1/sd/myamamot/metadetect/rho_tau_stats
    """

    good_piffs_table = fio.read('/global/project/projectdirs/des/schutt20/catalogs/y6a2_piff_v2_hsm_allres_collated.fits')

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
        # Right now, it computes the tau-stats from the response corrected g1, g2, and the response is for over all the tiles. 
        stats = measure_tau(good_piffs_table, max_sep, max_mag, response_filepath, mdet_input_filepaths, stats_out_dir, subtract_mean=True)
        stat_file = os.path.join(stats_out_dir, "tau_%s_%s.json"%(name, tag))
        write_stats_tau(stat_file,*stats)
    else:
        print('Skipping tau-stats. ')

def main(argv):
    do_rho = sys.argv[1]
    do_tau = sys.argv[2]
    max_sep = int(sys.argv[3])
    stats_name = sys.argv[4]
    response_filepath = sys.argv[5] 
    mdet_input_filepaths = sys.argv[6]
    stats_out_dir = sys.argv[7]
    measure_rho_tau_stats(do_rho, do_tau, max_sep, stats_name, response_filepath, mdet_input_filepaths, stats_out_dir)    

if __name__ == "__main__":
    main(sys.argv)
