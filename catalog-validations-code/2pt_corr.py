import glob
import fitsio as fio
import numpy as np
import proplot as pplt
import treecorr
import pickle

# Set outpath
outpath = '/global/project/projectdirs/des/myamamot/2pt_corr/'

# Load Y6 catalogs
mdet_files = glob.glob('/global/cscratch1/sd/myamamot/metadetect/cuts_v2/*_metadetect-v5_mdetcat_part0000.fits')
with open('/global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted.pkl', 'rb') as handle:
    res = pickle.load(handle)
    handle.close()

bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 0.5,
        max_sep = 250,
        bin_size = 0.2,
    
        output_dots = False,
    )

gg = treecorr.GGCorrelation(bin_config, verbose=2)
cat1 = treecorr.Catalog(ra=res[0]['ra'], dec=res[0]['dec'], ra_units='deg', dec_units='deg', g1=res[0]['e1'], g2=res[0]['e2'])
cat2 = treecorr.Catalog(ra=res[0]['ra'], dec=res[0]['dec'], ra_units='deg', dec_units='deg', g1=res[0]['e1'], g2=res[0]['e2'])
gg.process(cat1,cat2)
gg.write(outpath+'shear2pt_nontomo_treecorrw.txt')