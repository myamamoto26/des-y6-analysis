import fitsio as fio
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from esutil import stat
import esutil as eu
from tqdm import tqdm
import json
# from joblib import Parallel, delayed
import time
import pickle
import treecorr

def get_cat_recarr(cat_path, band=None, n_rows=None):
    """takes path to the input Piff catalog as input. Applies optional filters
    on the catalog. Returns a recarray of the catalog with the relevant columns."""

    cols = ['RA', 'DEC', 'DELTA_G1_WMEANSUB_W_OUT', 'DELTA_G2_WMEANSUB_W_OUT', 'G1_MODEL_WMEANSUB_W_OUT', 'G2_MODEL_WMEANSUB_W_OUT',
            'DELTA_T_FRAC_WMEANSUB_W_OUT', 'T_MODEL_WMEANSUB_W_OUT', 'BAND', 'GI_COLOR', 'IZ_COLOR', 'STARGAL_COLOR_WEIGHT_W_OUTLIERS']
    piffcat = fio.read(cat_path, columns=cols)

    # optionally cut down to random `n_rows` entries to limit plot rendering time
    if n_rows is not None:
        print('Cutting to %i random entries. %f of total catalog.'%(n_rows, n_rows/len(piffcat)))
        rng = np.random.default_rng(1234)
        nstar = int(n_rows)
        idx_sample = rng.integers(low=0, high=len(piffcat), size=nstar) #high is exclusive
        piffcat = piffcat[idx_sample]

    # optionally add band filter
    if band is not None:
        print('Using only %s band.'%band)
        piffcat = piffcat[piffcat['BAND'] == band]

    return piffcat

def measure_tangential_shear(d, mask, cname, name, var_method, fpatch, ffcent, frand, out_path):

    # set up treecorr call
    bin_config = dict(
                sep_units = 'arcmin',
                bin_slop = 0.01,

                min_sep = 0.5,
                max_sep = 150,
                nbins = 20,

                var_method = var_method,
                output_dots = False,
                )
    
    print('treecorr catalog...')
    cat1_file = ffcent
    cat1 = treecorr.Catalog(cat1_file, ra_col='RA_CENT', dec_col='DEC_CENT', ra_units='deg', dec_units='deg', patch_centers=fpatch)
    # random point subtraction. 
    cat1r_file = frand
    cat1r = treecorr.Catalog(cat1r_file, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg', patch_centers=fpatch)
    ng_rand = treecorr.NGCorrelation(bin_config, verbose=2)
    ng = treecorr.NGCorrelation(bin_config, verbose=2)

    if name == 'model':
        cat2 = treecorr.Catalog(ra=d['ra'][mask], dec=d['dec'][mask], ra_units='deg', dec_units='deg', g1=d['e1'][mask], g2=d['e2'][mask], w=d['w'][mask], patch_centers=fpatch)
    elif name == 'residual':
        cat2 = treecorr.Catalog(ra=d['ra'][mask], dec=d['dec'][mask], ra_units='deg', dec_units='deg', g1=d['de1'][mask], g2=d['de2'][mask], w=d['w'][mask], patch_centers=fpatch)
    ng.process(cat1, cat2, low_mem=True)
    ng_rand.process(cat1r, cat2, low_mem=True)

    ng.write(os.path.join(out_path, 'psf_'+name+'_'+cname+'_star_color_field_centers_'+var_method+'_bins0.01_min0.5_max150.fits'), rg=ng_rand)
    ng.calculateXi(rg=ng_rand)
    ng_cov = ng.cov
    np.save(os.path.join(out_path, 'psf_'+name+'_'+cname+'_star_color_field_centers_'+var_method+'_bins0.01_min0.5_max150_cov.npy'), ng_cov)


def main(argv):
    # piff_cat = "/global/cfs/cdirs/des/schutt20/catalogs/y6a2_piff/v3_HOM_mdet/y6a2_piff_v3_HOMs_v1_hsmask-v3_mdet-v5b_w-v4.4_riz_rhotau_input_v2.fits"
    piff_cat = "/global/cfs/cdirs/des/schutt20/catalogs/y6a2_piff/v3_HOM_mdet/y6a2_piff_v3_HOMs_v1_STAR-COLORS_hsmask-v3_mdet-v5b_w-v4.4_riz_rhotau_input.fits"
    # band = "r"
    outpath = '/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/field_centers/'
    fpatch = '/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed8888.fits'
    ffcent = '/pscratch/sd/m/myamamot/pizza-slice/exposure_field_centers.fits'
    frand = "/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hsmap_random_v3_fcenters.fits"

    print('reading in piff cat...')
    dat = {}
    piffcat = get_cat_recarr(piff_cat)
    dat['ra'] = piffcat['RA']
    dat['dec'] = piffcat['DEC']
    dat['e1'] = piffcat['G1_MODEL_WMEANSUB_W_OUT']
    dat['e2'] = piffcat['G2_MODEL_WMEANSUB_W_OUT']
    dat['T'] = piffcat['T_MODEL_WMEANSUB_W_OUT']
    dat['de1'] = piffcat['DELTA_G1_WMEANSUB_W_OUT']
    dat['de2'] = piffcat['DELTA_G2_WMEANSUB_W_OUT']
    dat['dT/T'] = piffcat['DELTA_T_FRAC_WMEANSUB_W_OUT']
    dat['w'] = piffcat['STARGAL_COLOR_WEIGHT_W_OUTLIERS']

    print('measuring tangential projection of PSF shapes...')
    colors = [(-2.00, 0.76), (0.76, 1.49), (1.49, 4.00)]
    cnames = ['blue', 'mid', 'red']
    for i,color in tqdm(enumerate(colors)):
        cmin = color[0]; cmax = color[1]
        msk = ((piffcat['GI_COLOR'] > cmin) & (piffcat['GI_COLOR'] <= cmax))
        measure_tangential_shear(dat, msk, cnames[i], 'model', 'bootstrap', fpatch, ffcent, frand, outpath)
        measure_tangential_shear(dat, msk, cnames[i], 'residual', 'bootstrap', fpatch, ffcent, frand, outpath)

if __name__ == "__main__":
    main(sys.argv)