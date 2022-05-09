
import fitsio as fio
import numpy as np
import os, sys
from tqdm import tqdm
from matplotlib import pyplot as plt
import emcee
import pickle
import glob
from rho_stats import measure_rho, measure_tau, write_stats, write_stats_tau, plot_overall_rho, plot_overall_tao

def main(argv):

    f = open('/global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt', 'r')
    fs = f.read().split('\n')[:-1]
    mdet_filenames = [fname.split('/')[-1] for fname in fs]
    tilenames = [d.split('_')[0] for d in mdet_filenames]
    good_piffs_table = fio.read('/global/project/projectdirs/des/schutt20/catalogs/y6a2_piff_v2_hsm_allres_collated.fits')

    # rho-stats -> Just need to pass the piff catalog.
    if not os.path.exists(os.path.join('/global/cscratch1/sd/myamamot/metadetect/rho_tau_stats', 'rho_all_JK_v2_griz.json')): 
        print('Computing rho-stats...')
        max_sep = 60
        max_mag = 0
        name = 'all_JK_v2' 
        tag = 'griz'
        stats = measure_rho(good_piffs_table, max_sep, max_mag, subtract_mean=True, do_rho0=True)
        stat_file = os.path.join('/global/cscratch1/sd/myamamot/metadetect/rho_tau_stats', "rho_%s_%s.json"%(name, tag))
        write_stats(stat_file,*stats)
    else:
        print('Skipping rho-stats. ')

    # tau-stats. 
    if not os.path.exists(os.path.join('/global/cscratch1/sd/myamamot/metadetect/rho_tau_stats', 'tau_all_JK_v3_griz.json')): 
        print('Computing tau-stats...')
        max_sep = 60
        max_mag = 0
        name = 'all_JK_v3' 
        tag = 'griz'
        stats = measure_tau(good_piffs_table, max_sep, max_mag, subtract_mean=True)
        stat_file = os.path.join('/global/cscratch1/sd/myamamot/metadetect/rho_tau_stats', "tau_%s_%s.json"%(name, tag))
        write_stats_tau(stat_file,*stats)
    else:
        print('Skipping tau-stats. ')
    

if __name__ == "__main__":
    main(sys.argv)
