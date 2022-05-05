import os, sys
import glob
import fitsio as fio
import numpy as np
import proplot as pplt
import treecorr
import pickle
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Set outpath
outpath = '/global/project/projectdirs/des/myamamot/2pt_corr/'
subtract_mean = True

# Load Y6 catalogs
# mdet_files = glob.glob('/global/cscratch1/sd/myamamot/metadetect/cuts_v2/*_metadetect-v5_mdetcat_part0000.fits')
with open('/global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted.pkl', 'rb') as handle:
    res = pickle.load(handle)
    handle.close()


bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 1.0,

        min_sep = 1,
        max_sep = 400,
        nbins = 1000,
        # bin_size = 0.2,

        output_dots = False,
    )

e1 = res[0]['e1']
e2 = res[0]['e2']
ra = res[0]['ra']
dec = res[0]['dec']

if subtract_mean:
    e1 -= np.mean(e1)
    e2 -= np.mean(e2)

if rank == 0:
    if not os.path.exists(outpath+'patch_centers.txt'):
        cat_patch = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2, npatch=50)
        cat_patch.write_patch_centers(outpath+'patch_centers.txt')
        print('patch center done')
    del cat_patch
comm.Barrier()

cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2, patch_centers=outpath+'patch_center.txt')
print('catalog done', rank)
gg = treecorr.GGCorrelation(bin_config, verbose=2)
gg.process(cat, comm=comm)
print('calculation done', rank)

if rank == 0:
    gg.write(outpath+'y6_shear2pt_nontomo_subtract_mean.fits')


