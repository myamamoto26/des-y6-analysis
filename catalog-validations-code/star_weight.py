import healpy as hp
import healsparse as hsp
import skyproj
import fitsio as fio
import numpy as np

def apply_gold_mask(d, ra_col, dec_col, hleda=False):

    ra = np.array(d[ra_col])
    dec = np.array(d[dec_col])
    in_gold_footprint = footprint_map.get_values_pos(ra, dec, valid_mask=True)
    in_gold_mask = gold_mask_map.get_values_pos(ra, dec, valid_mask=True)
    msk = (in_gold_footprint & ~in_gold_mask)
    
    return msk

gal_data = fio.read('/global/cfs/cdirs/des/myamamot/gold/gold_galaxies_master.fits')
msk = ((gal_data['EXT_MASH'] == 3) | (gal_data['EXT_MASH'] == 4))
gal_data = gal_data[msk & (gal_data['FLAGS_FOREGROUND'] == 0)]
star_data = fio.read('/global/cfs/cdirs/des/myamamot/gold/stars/gold_stars_master.fits')

footprint_map = hsp.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6a2_decasu_griz_nexpgt2_and_fp2_footprint.hsp")
gold_mask_map = hsp.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6a2_foreground_mask_healsparse_nside16384.fits.gz")
msk_gold = apply_gold_mask(gal_data, 'RA', 'DEC'); msk_gold_star = apply_gold_mask(star_data, 'RA', 'DEC')
gal_data = gal_data[msk_gold]; star_data = star_data[msk_gold_star]

def _compute_gal_star_density(gal_ra, gal_dec, star_ra, star_dec, w, nside, mask=False):
    """
    computes the ratio of galaxy density to stellar density.
    """
    import copy
    import healpy as hp

    if mask:
        mdet_msk_map = hsp.HealSparseMap.read('/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hleda-gaiafull-des-stars-hsmap131k-mdet-extra-masks-v2.hsp')
        in_footprint_gal = mdet_msk_map.get_values_pos(gal_ra, gal_dec, valid_mask=True)
        in_footprint_star = mdet_msk_map.get_values_pos(star_ra, star_dec, valid_mask=True)
        gal_ra = gal_ra[in_footprint_gal]
        gal_dec = gal_dec[in_footprint_gal]
        star_ra = star_ra[in_footprint_star]
        star_dec = star_dec[in_footprint_star]

    nu = np.zeros(hp.nside2npix(nside))
    desy6_map = np.zeros(hp.nside2npix(nside))
    pix1 = hp.ang2pix(nside, gal_ra, gal_dec, nest=True, lonlat=True)

    unique_pix1, idx1, idx_rep1 = np.unique(pix1, return_index=True, return_inverse=True)
    nu[unique_pix1] += np.bincount(idx_rep1, weights=np.ones(len(pix1)))
    desy6_map[unique_pix1] += np.bincount(idx_rep1, weights=np.ones(len(pix1)))
    mas_desy3 = nu!=0.
    #desy3_map[mas_desy3]=desy3_map[mas_desy3]/nu[mas_desy3]

    stars_map = np.zeros(hp.nside2npix(nside))
    pix1 = hp.ang2pix(nside, star_ra, star_dec, nest=True, lonlat=True)
    unique_pix1, idx1, idx_rep1 = np.unique(pix1, return_index=True, return_inverse=True)
    stars_map[unique_pix1] += np.bincount(idx_rep1, weights=np.ones(len(pix1)))

    weight_map = copy.copy(stars_map)
    weight_map[stars_map!=0.] = desy6_map[stars_map!=0.]/stars_map[stars_map!=0.]
    # assign weights to stars
    # weight_stars = weight_map[pix1]
    hmap = hsp.HealSparseMap(nside_coverage=nside, healpix_map=weight_map)
    weight_stars = hmap.get_values_pix(pix1)

    return desy6_map, stars_map, weight_map, weight_stars

gal_map, star_map, weight_map, weight_stars = _compute_gal_star_density(gal_data['RA'], gal_data['DEC'], star_data['RA'], star_data['DEC'], np.ones(len(gal_data['RA'])), 256, mask=False)
hsp_map = hsp.HealSparseMap(nside_coverage=256, healpix_map=weight_map)
hsp_map.write('/global/cfs/cdirs/des/y6-shear-catalogs/star_weight_map_nside256_gold_masked_v2.fits', clobber=False)