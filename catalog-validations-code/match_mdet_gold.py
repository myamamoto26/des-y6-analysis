import fitsio as fio
import numpy as np
import smatch
import glob
from tqdm import tqdm

nside = 4096
maxmatch = 1
radius = 0.263/3600 # degrees

d_mdet = fio.read('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/marco_metadetection_v2.fits')
gold_files = glob.glob('/global/cscratch1/sd/myamamot/gold/gold_galaxies_*.fits')

matched = []
for f in tqdm(gold_files):
    d_gold = fio.read(f)
    d_gold = d_gold[d_gold['EXT_MASH'] == 4]

    matches = smatch.match(d_gold['RA'], d_gold['DEC'], radius, d_mdet['ra'], d_mdet['dec'], nside=nside, maxmatch=maxmatch)

    d_match = d_gold[matches['i1']]
    d_match_mdet = d_mdet[matches['i2']]
    res = np.zeros(len(d_match), dtype=[('ra', 'f8'), ('dec', 'f8'), ('s2n', 'f8'), ('mag_g', 'f8'), ('mag_r', 'f8'), ('mag_i', 'f8'), ('mag_z', 'f8')])
    res['ra'] = d_match['RA']
    res['dec'] = d_match['DEC']
    res['mag_g'] = d_match['BDF_MAG_G']
    res['mag_r'] = d_match['BDF_MAG_R']
    res['mag_i'] = d_match['BDF_MAG_I']
    res['mag_z'] = d_match['BDF_MAG_Z']
    res['s2n'] = d_match_mdet['s2n']

    matched.append(res)

matched_all = np.concatenate(matched)
fio.write('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/matched_gold_mdet.fits', matched_all)