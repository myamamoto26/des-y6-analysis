import healpy as hp
import healsparse as hsp
import skyproj
import fitsio as fio
import numpy as np
gal_data = fio.read('/pscratch/sd/m/myamamot/gold/gold_galaxies_extmash34.fits')
star_data = fio.read('/pscratch/sd/m/myamamot/gold/stars/gold_stars_master.fits')
ra_piff = star_data['RA']
dec_piff = star_data['DEC']

def _compute_gal_star_density(gal_ra, gal_dec, star_ra, star_dec, w, nside, mask=False):
    """
    computes the ratio of galaxy density to stellar density.
    """
    import copy
    import healpy as hp

    if mask:
        mdet_msk_map = hsp.HealSparseMap.read('/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hleda-gaiafull-des-stars-hsmap16384-nomdet-v3.fits')
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

gal_map, star_map, weight_map, weight_stars = _compute_gal_star_density(gal_data['RA'], gal_data['DEC'], ra_piff, dec_piff, np.ones(len(gal_data['RA'])), 256, mask=True)
hsp_map = hsp.HealSparseMap(nside_coverage=256, healpix_map=weight_map)
hsp_map.write('/global/cfs/cdirs/des/y6-shear-catalogs/star_weight_map_nside256_gold_masked.fits', clobber=False)