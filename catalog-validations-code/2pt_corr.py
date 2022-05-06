import os, sys
import glob
import fitsio as fio
import numpy as np
import treecorr
import pickle
sys.path.append('/global/project/projectdirs/des/myamamot/metadetect/')
from hybrideb import hybrideb
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def read_fpfm(fname): #, gammat, gammax):
    """Compute Xp, Xm using the already-defined B-mode estimator. 
       Each bandpower is computed using theta_min=1.0, theta_max=400, Ntheta=1000. """
    
    # Read Gauss EB
    with open(fname, 'rb') as f:
        res = pickle.load(f)
    f.close()
    
    theta_rad = res[0]
    fp = res[1]
    fm = res[2]
    
    return fp, fm

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
        cat_patch = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2, npatch=100)
        cat_patch.write_patch_centers(outpath+'patch_centers.txt')
        print('patch center done')
        del cat_patch
comm.Barrier()

cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2, patch_centers=outpath+'patch_centers.txt')
print('catalog done', rank)
gg = treecorr.GGCorrelation(bin_config, verbose=2)
gg.process(cat, comm=comm)
print('calculation done', rank)

if rank == 0:
    gg.write(outpath+'y6_shear2pt_nontomo_subtract_mean.fits')
    with open(outpath+'y6_shear2pt_nontomo_subtract_mean.pkl', 'wb') as f: # save gg as a pickle file, so that we can refer to all the results later. 
        pickle.dump(gg, f)

    cov_jk = gg.estimate_cov('jackknife')
    np.save(outpath+'y6_shear2pt_nontomo_JKcov.npy', cov_jk)

    # covariance for B-mode stats
    corr_fs = glob.glob('../2pt_corr/B_mode/geb_Y6_*.pkl') # B-mode estimator (each bandpower)
    for i,fname in enumerate(corr_fs):
        fp, fm = read_fpfm(fname)
        # func_Xp = lambda corrs: np.sum((fp*corrs[0] + fm*corrs[1])/2) # Xp
        # func_Xm = lambda corrs: np.sum((fp*corrs[0] - fm*corrs[1])/2) # Xm
        # cov_Xp = gg.estimate_cov(method='jackknife', func=func_Xp) # or 'bootstrap'
        # cov_Xm = gg.estimate_cov(method='jackknife', func=func_Xm) # or 'bootstrap'
        # np.save(outpath+'Xp_JKcov_%d.npy'%i, cov_Xp)
        # np.save(outpath+'Xm_JKcov_%d.npy'%i, cov_Xm)

        func = lambda corr: np.concatenate([np.sum((fp*corr.xip + fm*corr.xim)/2), # Xp
                                            np.sum((fp*corr.xip - fm*corr.xim)/2)] # Xm
                                          )
        cov_XpXm = gg.estimate_cov(method='jackknife', func=func) # or 'bootstrap'
        
    print('done')
