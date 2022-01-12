
import fitsio as fio
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
import treecorr
import smatch
import os
from tqdm import tqdm

def exclude_gold_mask_objects(d, which_mask):

    import healpy as hp
    import h5py

    hpix = hp.ang2pix(4096, d['RA'], d['DEC'], lonlat=True, nest=True)

    if which_mask == 'y6':
        # Y6 mask
        gold_mask = fio.read('/data/des70.a/data/masaya/gold/y6a2_foreground_mask_v1.1.fits')
        exclude_pix = np.unique(gold_mask['PIXEL'])
        mask = np.in1d(hpix, exclude_pix, invert=True)
    elif which_mask == 'y3':
        # Y3 mask
        good_pix = fio.read('/home/s1/masaya/des-y6-analysis/y3_gold_mask_good_hpix.fits')
        mask = np.in1d(hpix, good_pix)
    
    print('excluded number of objects with gold mask, ', len(d)-len(d[mask]))
    return d[mask]

def exclude_hyperleda_objects(d):

    import smatch
    d_hyperleda = os.path.join('/home/s1/masaya/des-y6-analysis', 'asu.fit')
    hyperleda = fio.read(d_hyperleda)
    d_gold = os.path.join('/data/des70.a/data/masaya/gold', 'y6_gold_2_0_magnitudes.fits')
    gold = fio.read(d_gold)

    nside = 4096
    maxmatch = 1
    radius = 0.263/3600 # degrees
    matches = smatch.match(hyperleda['RAJ2000'], hyperleda['DEJ2000'], radius, gold['RA'], gold['DEC'], nside=nside, maxmatch=maxmatch)
    hyperleda_des = hyperleda[matches['i1']]
    print('hyperleda objects within DES footprint', len(hyperleda_des))
    print('mdet catalog', len(d))
    
    masked_obj = []
    for ii in tqdm(range(len(hyperleda_des))):

        radius = 2*(10**(hyperleda_des['logD25'][ii]))/600
        mask_limit = ((d['RA'] >= hyperleda_des['RAJ2000'][ii]-1.0) & 
                        (d['RA'] <= hyperleda_des['RAJ2000'][ii]+1.0) & 
                        (d['DEC'] >= hyperleda_des['DEJ2000'][ii]-1.0) & 
                        (d['DEC'] <= hyperleda_des['DEJ2000'][ii]+1.0))
        mdet_limit = d[mask_limit]
        if len(mdet_limit) == 0:
            continue
        # print(hyperleda_des['RAJ2000'][ii], hyperleda_des['DEJ2000'][ii], len(mdet_limit), mdet_limit)
        matches = smatch.match(hyperleda_des['RAJ2000'][ii], hyperleda_des['DEJ2000'][ii], radius, mdet_limit['RA'], mdet_limit['DEC'], nside=nside, maxmatch=0)
        masked_obj.append(mdet_limit[matches['i2']])
    total_mask = np.concatenate(masked_obj)
    mdet_mask = np.in1d(d['ID'], total_mask['ID'], invert=True)
    mdet_masked = d[mdet_mask]

    print('excluded number of metadetect objects around hyperleda galaxies', len(d)-len(mdet_masked))
    return mdet_masked

## Examples.
# metadetect = fio.read('/data/des70.a/data/masaya/metadetect/v2/mdet_test_all_v2.fits')
# mdet_noshear = metadetect[metadetect['MDET_STEP']=='noshear']
# hyperleda = fio.read('~/des-y6-analysis/asu.fit')
# # c = SkyCoord(hyperleda['RAJ2000'], hyperleda['DEJ2000'], unit=(u.deg, u.deg))
# gold_cat = os.path.join('/data/des70.a/data/masaya/gold', 'y6_gold_2_0_magnitudes.fits')
                    

# gold = fio.read(gold_cat)
# nside = 4096
# maxmatch = 1
# radius = 0.263/3600 # degrees
# matches = smatch.match(hyperleda['RAJ2000'], hyperleda['DEJ2000'], radius, gold['RA'], gold['DEC'], nside=nside, maxmatch=maxmatch)
# matched_data = hyperleda[matches['i1']]
# matched_data_mag = gold[matches['i2']]['BDF_MAG_R']
# print(len(matched_data_mag))

# matched_data_mag = matched_data_mag[matched_data_mag > 0]
# print(len(matched_data_mag))

# fig, ax = plt.subplots(figsize=(10,7)) 
# ax.hist(matched_data_mag, bins=40, histtype='step')
# ax.set_xlabel('Magnitude', fontsize=16)
# ax.set_title('r-band magnitude of HYPERLEDA objects matched with GOLD catalog')
# plt.tight_layout()
# plt.savefig('hyperleda_gold_mag.pdf')

# bin_config = dict(
#         sep_units = 'arcmin',
#         bin_slop = 0.1,

#         min_sep = 1.0,
#         max_sep = 250,
#         nbins = 20,
#     )

# cat1 = treecorr.Catalog(ra=mdet_noshear['RA'], dec=mdet_noshear['DEC'], ra_units='deg', dec_units='deg')
# cat2 = treecorr.Catalog(ra=c.galactic.l.degree, dec=c.galactic.b.degree, ra_units='deg', dec_units='deg')
# nn = treecorr.NNCorrelation(bin_config)
# nn.process(cat1,cat2)

# fig, ax = plt.subplots(figsize=(10,7)) 
# ax.errorbar(nn.meanr, nn.xi, yerr=np.sqrt(nn.varxi), fmt='o')
# ax.set_xlabel(r'$\theta [arcmin]$', fontsize='xx-large' )
# ax.set_xscale('log')

# plt.tight_layout()
# plt.savefig('hyperleda_metadetect_masking.pdf')
