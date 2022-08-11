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

def _build_Bmode_estimator(mdet_2pt_input_filepath, out_path):
    """
    Builds bandpower estimator from the 2pt correlation function bin information.

    Parameters
    ----------
    mdet_2pt_input_filepath: The input filepath for the 2pt correlation function created in _compute_2pt_function function
    Example) /global/project/projectdirs/des/myamamot/2pt_corr/y6_shear2pt_nontomo_v3.fits
    
    out_path: The output filepath
    Example)  /global/project/projectdirs/des/myamamot/2pt_corr/B_mode/
    """

    import progressbar

    y6_xipm = fio.read(mdet_2pt_input_filepath)
    y6_theta = y6_xipm['meanr']
    y6_gammat = y6_xipm['xip']
    y6_gammax = y6_xipm['xim']

    theta_min = y6_theta[0] # arcmin
    theta_max = y6_theta[-1] # arcmin
    Ntheta    = len(y6_theta) # number of bins in log(theta) 

    heb = hybrideb.HybridEB(theta_min, theta_max, Ntheta)
    beb = hybrideb.BinEB(theta_min, theta_max, Ntheta)
    geb = hybrideb.GaussEB(beb, heb)

    # Write out Gauss EB information
    for n in range(20):
        with open(os.path.join(out_path, 'geb_Y6_'+str(n)+'.pkl'), 'wb') as f:
            pickle.dump(geb(n), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

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

def _compute_2pt_function(mdet_input_flatfile, out_path, corr_out_fits_filepath, corr_out_pickle_filepath, bmode, cov_method):
    """
    Computes 2pt correlation functions for B-mode estimation. 

    Parameters
    ----------
    mdet_input_flatfile: the input file path of the flat catalog created in make_flat_catalog.py
    Example) /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3.pkl

    out_path: the output file path
    Example) /global/project/projectdirs/des/myamamot/2pt_corr

    corr_out_fits_filepath: the output fits file path of the computed 2pt function
    Example) /global/project/projectdirs/des/myamamot/2pt_corr/y6_shear2pt_nontomo_v3.fits

    corr_out_pickle_filepath: the output pickle file path of the computed 2pt function
    Example) /global/project/projectdirs/des/myamamot/2pt_corr/y6_shear2pt_nontomo_v3.pkl

    bmode: whether or not do B-mode estiamtion (Boolean)

    cov_method: what covariance computation method
    Example) jackknife, bootstrap
    """

    subtract_mean = True
    # Load Y6 catalogs
    with open(mdet_input_flatfile, 'rb') as handle:
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
        if not os.path.exists(out_path+'patch_centers.txt'):
            cat_patch = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2, npatch=100)
            cat_patch.write_patch_centers(os.path.join(out_path, 'patch_centers.txt'))
            print('patch center done')
            del cat_patch
    comm.Barrier()


    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2, patch_centers=os.path.join(out_path,'patch_centers.txt'))
    print('catalog done', rank)
    gg = treecorr.GGCorrelation(bin_config, verbose=2)
    gg.process(cat, comm=comm)
    print('calculation done', rank)

    if rank == 0:
        gg.write(corr_out_fits_filepath)
        with open(corr_out_pickle_filepath, 'wb') as f: # save gg as a pickle file, so that we can refer to all the results later. 
            pickle.dump(gg, f)

        cov_jk = gg.estimate_cov('jackknife')
        np.save(os.path.join(out_path,'y6_shear2pt_nontomo_JKcov.npy'), cov_jk)
    comm.Barrier()

    if bmode: 
        if len(glob.glob(os.path.join(out_path,'geb_Y6_*.pkl'))) == 0:
            _build_Bmode_estimator(corr_out_fits_filepath, out_path)
        if rank == 0:
            if os.path.exists(corr_out_pickle_filepath):
                with open(corr_out_pickle_filepath, 'rb') as f:
                    gg = pickle.load(f)

                # covariance for B-mode stats
                print('computing B-mode stats')
                corr_fs = sorted(glob.glob(os.path.join(out_path,'geb_Y6_*.pkl'))) # B-mode estimator (each bandpower)
                allfp = []
                allfm = []
                for fname in corr_fs:
                    fp, fm = read_fpfm(fname)
                    allfp.append(fp)
                    allfm.append(fm)

                # Turn these into matrices
                fp = np.array(allfp)
                fm = np.array(allfm)

                func = lambda corr: np.concatenate([(fp.dot(corr.xip) + fm.dot(corr.xim))/2, # Xp
                                                (fp.dot(corr.xip) - fm.dot(corr.xim))/2] # Xm
                                            )
                XpXm = func(gg)
                cov_XpXm = gg.estimate_cov(method=cov_method, func=func) 
                np.save(os.path.join(out_path,'XpXm_BS.npy'), XpXm)
                np.save(os.path.join(out_path,'XpXm_BScov.npy'), cov_XpXm)
                print('done')
            else:
                print('please compute correlation function first')

def main(argv):

    mdet_input_flatfile = sys.argv[1]
    out_path = sys.argv[2]
    corr_out_fits_filepath = sys.argv[3]
    corr_out_pickle_filepath = sys.argv[4]
    bmode = sys.argv[5]
    cov_method = sys.argv[6]

    _compute_2pt_function(mdet_input_flatfile, out_path, corr_out_fits_filepath, corr_out_pickle_filepath, bmode, cov_method)

if __name__ == "__main__":
    main(sys.argv)