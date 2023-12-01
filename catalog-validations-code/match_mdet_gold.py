import fitsio as fio
import numpy as np
import smatch
import glob
from tqdm import tqdm
from des_y6utils import mdet

nside = 4096
maxmatch = 1
radius = 0.263/3600 # degrees

# mdet_files = glob.glob('/global/cfs/cdirs/des/y6-shear-catalogs/metadetection/*_metadetect.fits')
d_mdet = fio.read('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/y6_catalog4matching.fits') # cuts applied: msk = mdet._make_mdet_cuts(d, min_s2n=7, n_terr=0, min_t_ratio=1.1) 
# gold_files = glob.glob('/global/cscratch1/sd/myamamot/gold/gold_galaxies_*.fits')
d_match = fio.read('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/y3_mcal_catalog_matched.fits')

# matched = []
# for f in tqdm(mdet_files):
#     d_mdet = fio.read(f)
#     msk = mdet._make_mdet_cuts(d_mdet, min_s2n=7, n_terr=0, min_t_ratio=1.1) 
#     msk_noshear = (d_mdet['mdet_step']=='noshear')

#     d_mdet = d_mdet[msk & msk_noshear]

    # smatch.match method
    # matches = smatch.match(d_match['ra'], d_match['dec'], radius, d_mdet['ra'], d_mdet['dec'], nside=nside, maxmatch=maxmatch)
    # d_matched = d_match[matches['i1']]
    # d_match_mdet = d_mdet[matches['i2']]
    # np.save('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/y3_1arcmin_matched_index.npy', matches['i1'])
    # np.save('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/y6_1arcmin_matched_index.npy', matches['i2'])
    # np.save('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/matched_dist_1arcmin.npy', matches['cosdist'])

# knn method
matcher = smatch.matcher.Matcher(d_mdet['ra'], d_mdet['dec'])
idx, dist = matcher.query_knn(d_match['ra'], d_match['dec'], k=1, distance_upper_bound=None, return_distances=True)
np.save('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/knn_all_matched_index.npy', idx)
np.save('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/knn_all_matched_dist.npy', dist)

# res = np.zeros(len(d_match), dtype=[('ra', 'f8'), ('dec', 'f8'), ('s2n', 'f8'), ('T', 'f8'), ('size_ratio', 'f8'), ('mag_i', 'f8'), ('mag_z', 'f8')])
# res['ra'] = d_match['RA']
# res['dec'] = d_match['DEC']
# res['mag_g'] = d_match['BDF_MAG_G']
# res['mag_r'] = d_match['BDF_MAG_R']
# res['mag_i'] = d_match['BDF_MAG_I']
# res['mag_z'] = d_match['BDF_MAG_Z']
# res['s2n'] = d_match_mdet['s2n']

# matched.append(res)

# matched_all = np.concatenate(matched)
# fio.write('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/matched_gold_mdet.fits', matched_all)