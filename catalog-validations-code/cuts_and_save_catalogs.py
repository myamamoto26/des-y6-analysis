
import fitsio as fio
import os, sys
import numpy as np
from tqdm import tqdm
import healsparse

# Apply cuts + image masks, and save the catalogs.
hmap = healsparse.HealSparseMap.read('/global/project/projectdirs/des/myamamot/metadetect/y6-combined-hsmap16384-nomdet.fits')
f = open('/global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt', 'r')
fs = f.read().split('\n')[:-1]
mdet_filenames = [fname.split('/')[-1] for fname in fs]
tilenames = [d.split('_')[0] for d in mdet_filenames]

for fname in tqdm(mdet_filenames):

    if os.path.exists(os.path.join('/global/cscratch1/sd/myamamot/metadetect/cuts_v3', fname)):
        continue
    elif not os.path.exists(os.path.join('/global/project/projectdirs/des/myamamot/metadetect/v3', fname)):
        print(fname)
        continue
    try:
        d = fio.read(os.path.join('/global/project/projectdirs/des/myamamot/metadetect/v3', fname))
    except:
        print(fname)
        continue
    mag_g = 30.0 - 2.5*np.log10(d["mdet_g_flux"])
    mag_r = 30.0 - 2.5*np.log10(d["mdet_r_flux"])
    mag_i = 30.0 - 2.5*np.log10(d["mdet_i_flux"])
    mag_z = 30.0 - 2.5*np.log10(d["mdet_z_flux"])
    gmr = mag_g - mag_r
    rmi = mag_r - mag_i
    imz = mag_i - mag_z

    msk = ((d["flags"] == 0) & (d["mask_flags"] == 0) & (d["mdet_flux_flags"] == 0) & (d["mdet_T_ratio"] > 0.5) & (d["mdet_s2n"] > 10) & (d["mfrac"] < 0.1) 
            & (d["mdet_T"] < 1.9 - 2.8*d["mdet_T_err"]) & (np.abs(gmr) < 5) & (np.abs(rmi) < 5) & (np.abs(imz) < 5) & np.isfinite(mag_g) & np.isfinite(mag_r) 
            & np.isfinite(mag_i) & np.isfinite(mag_z) & (mag_g < 26.5) & (mag_r < 26.5) & (mag_i < 26.2) & (mag_z < 25.6))
    in_footprint = hmap.get_values_pos(d["ra"], d["dec"], valid_mask=True)

    total_msk = (msk & in_footprint)
    d_msk = d[total_msk]
    fio.write('/global/cscratch1/sd/myamamot/metadetect/cuts_v3/'+fname, d_msk)