
import fitsio as fio
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
import treecorr
import smatch
import os

metadetect = fio.read('/data/des70.a/data/masaya/metadetect/v2/mdet_test_all_v2.fits')
mdet_noshear = metadetect[metadetect['MDET_STEP']=='noshear']
hyperleda = fio.read('~/des-y6-analysis/asu.fit')
# c = SkyCoord(hyperleda['RAJ2000'], hyperleda['DEJ2000'], unit=(u.deg, u.deg))
gold_cat = os.path.join('/data/des70.a/data/masaya/gold', 'y6_gold_2_0_magnitudes.fits')
                    

gold = fio.read(gold_cat)
nside = 4096
maxmatch = 1
radius = 0.263/3600 # degrees
matches = smatch.match(hyperleda['RAJ2000'], hyperleda['DEJ2000'], radius, gold['RA'], gold['DEC'], nside=nside, maxmatch=maxmatch)
matched_data = hyperleda[matches['i1']]
matched_data_mag = gold[matches['i2']]['BDF_MAG_R']
print(len(matched_data_mag))

matched_data_mag = matched_data_mag[matched_data_mag['BDF_MAG_R'] > 0]
print(len(matched_data_mag))

fig, ax = plt.subplots(figsize=(10,7)) 
ax.hist(matched_data_mag, bins=40, histtype='step')
plt.tight_layout()
plt.savefig('hyperleda_gold_mag.pdf')

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
