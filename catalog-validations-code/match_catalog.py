import fitsio as fio
import numpy as np
import h5py as h5
from astropy.coordinates import SkyCoord
from astropy import units as uu
import glob
from tqdm import tqdm
from des_y6utils import mdet
import healsparse

run = 'master' #'master'
match_hleda = False
selections = ['mask', 'sg', 's2nsize', 'flux', 'junk']
selection = 'mask+sg+s2nsize+flux+junk'

def match_hyperleda(d, msk_):

    dh = fio.read('/global/cfs/cdirs/des/y6-shear-catalogs/hyperleda_B16_18.fits.gz', lower=True)
    ra_y6 = np.array(d['ra'])[msk_]
    dec_y6 = np.array(d['dec'])[msk_]
    cat1 = SkyCoord(ra=ra_y6*uu.degree, dec=dec_y6*uu.degree)
    cat2 = SkyCoord(ra=dh['ra']*uu.degree, dec=dh['dec']*uu.degree)
    idx, d2d, d3d = cat1.match_to_catalog_sky(cat2, nthneighbor=1)
    np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/goldmatch/mdety6_hleda_matched_idx.npy', idx)
    np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/goldmatch/mdety6_hleda_matched_d2d.npy', d2d.deg)

def apply_hyperleda():

    def get_hyperleda_radius(bmag):
        slope = -0.00824
        offset = 0.147
        return offset + slope * bmag

    HYPERLEDA_RADIUS_FAC = 1.5
    HYPERLEDA_MINRAD_ARCSEC = 0.0
    minrad_degrees = HYPERLEDA_MINRAD_ARCSEC / 3600
    NSIDE_COVERAGE = 32
    NSIDE = 16384
    HYPERLEDA_VAL = 2**9

    from esutil.numpy_util import between
    data = fio.read('/global/cfs/cdirs/des/y6-shear-catalogs/hyperleda_B16_18.fits.gz', lower=True)
    circles = []
    for objdata in tqdm(data):
        bmag = objdata['bt']
        ra = objdata['ra']
        dec = objdata['dec']

        # keep a superset of the DES area
        if between(dec, -75, 10) and (
            between(ra, 0, 120)
            or between(ra, 295, 360)
        ):

            radius_degrees = get_hyperleda_radius(bmag) * HYPERLEDA_RADIUS_FAC
            if radius_degrees < minrad_degrees:
                radius_degrees = minrad_degrees

            if radius_degrees > 0:

                circle = healsparse.geom.Circle(
                    ra=ra,
                    dec=dec,
                    radius=radius_degrees,
                    value=HYPERLEDA_VAL,
                )
                circles.append(circle)
    
    hm = healsparse.HealSparseMap.make_empty(
        nside_coverage=NSIDE_COVERAGE,
        nside_sparse=NSIDE,
        dtype=np.int16,
        sentinel=0,
    )
    healsparse.realize_geom(circles, hm)
    
    return hm
    

def apply_mask(d, ra_col, dec_col, hleda=False):

    hmap = healsparse.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hleda-gaiafull-des-stars-hsmap16384-mdet-v4.fits")
    footprint_map = healsparse.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6a2_decasu_griz_nexpgt2_and_fp2_footprint.hsp")
    gold_mask_map = healsparse.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6a2_foreground_mask_healsparse_nside16384.fits.gz")

    ra = np.array(d[ra_col])
    dec = np.array(d[dec_col])
    in_gold_footprint = footprint_map.get_values_pos(ra, dec, valid_mask=True)
    in_gold_mask = gold_mask_map.get_values_pos(ra, dec, valid_mask=True)
    msk = (in_gold_footprint & ~in_gold_mask)
    
    in_footprint = hmap.get_values_pos(ra, dec, valid_mask=True)
    msk &= in_footprint
    
    return msk

if run == 'master':
    f = h5.File('/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5b/metadetect_desdmv5a_cutsv5_patchesv5b.h5', 'r')
    d_y6 = f.get('/mdet/noshear')
    f_gold = fio.FITS('/pscratch/sd/m/myamamot/gold/all/gold_all_master.fits')
    d_gold = f_gold[-1].read(columns=['RA', 'DEC'])
    msk_gold = apply_mask(d_gold, 'RA', 'DEC')
    d_gold = d_gold[msk_gold]
    # f_y3 = h5.File('/global/cfs/cdirs/des/www/y3_cats/Y3_mastercat___UNBLIND___final_v1.1_12_22_20.h5', 'r')
    # m = np.array(f_y3.get('/index/select'))
    # d_y3 = f_y3.get('/catalog/metacal/unsheared')

    msk_mdet = apply_mask(d_y6, 'ra', 'dec')
    ra_y6 = np.array(d_y6['ra'])[msk_mdet]; dec_y6 = np.array(d_y6['dec'])[msk_mdet]
    if match_hleda:
        match_hyperleda(d_y6, msk_mdet)
    print(len(ra_y6))
    cat1 = SkyCoord(ra=ra_y6*uu.degree, dec=dec_y6*uu.degree)
    cat2 = SkyCoord(ra=np.array(d_gold['RA'])*uu.degree, dec=np.array(d_gold['DEC'])*uu.degree)
    idx, d2d, d3d = cat1.match_to_catalog_sky(cat2, nthneighbor=1)
    np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/goldmatch/gold_allcuts_shear_allcuts/mdety6_goldy6_matched_idx.npy', idx)
    np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/goldmatch/gold_allcuts_shear_allcuts/mdety6_goldy6_matched_d2d.npy', d2d.deg)
    # np.save('/pscratch/sd/m/myamamot/gold/mdet_matched_d3d.npy', d3d)
elif run == 'patch':

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank, size)

    f_gold = fio.FITS('/pscratch/sd/m/myamamot/gold/all/gold_all_master.fits')
    d_gold = f_gold[-1].read(columns=['RA', 'DEC'])
    mdet_f = sorted(glob.glob('/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5b/jackknife_patches_blinded/patch-*.fits'))
    import healsparse
    hmap = healsparse.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hleda-gaiafull-des-stars-hsmap16384-mdet-v4.fits")
    footprint_map = healsparse.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6a2_decasu_griz_nexpgt2_and_fp2_footprint.hsp")
    gold_mask_map = healsparse.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6a2_foreground_mask_healsparse_nside16384.fits.gz")
    
    # Mask out gold objects 
    in_mdet_footprint = hmap.get_values_pos(d_gold["RA"], d_gold["DEC"], valid_mask=True)
    in_gold_footprint = footprint_map.get_values_pos(d_gold['RA'], d_gold['DEC'], valid_mask=True)
    in_gold_mask = gold_mask_map.get_values_pos(d_gold['RA'], d_gold['DEC'], valid_mask=True)
    msk_mask = (in_mdet_footprint & in_gold_footprint & ~in_gold_mask)
    d_gold = d_gold[msk_mask]
    num_tot = 0
    for n,f in enumerate(mdet_f):
        if n % size != rank:
            continue
        if n % 25 == 0:
            print('made it to ', n)

        i = f.split('/')[-1][6:10]
        d_y6 = fio.read(f)
        num_tot += len(d_y6)

        # basic
        msk = (d_y6["mdet_step"] == 'noshear')
        in_gold_footprint = footprint_map.get_values_pos(d_y6["ra"], d_y6["dec"], valid_mask=True)
        in_gold_mask = gold_mask_map.get_values_pos(d_y6["ra"], d_y6["dec"], valid_mask=True)
        msk_mask = (in_gold_footprint & ~in_gold_mask)
        msk &= msk_mask

        # add selections
        if 'mask' in selections:
            if rank == 0:
                print('masking out footprint')
            in_footprint = hmap.get_values_pos(d_y6["ra"], d_y6["dec"], valid_mask=True)
            msk &= in_footprint

        if 'sg' in selections:
            if rank == 0:
                print('doing s/g sep')
            # s/g sep
            n_terr = 0
            msk &= (d_y6["gauss_T_ratio"] >= np.maximum(
                    0.5,
                    (n_terr*d_y6["gauss_T_err"]/d_y6["gauss_psf_T"])))

        if 's2nsize' in selections:
            if rank == 0:
                print('cutting s2n and size')
            # S/N and size
            min_s2n = 10
            max_s2n = np.inf
            max_size = np.inf # change to 20 for the final version
            msk &= ((d_y6["gauss_s2n"] > min_s2n)
                    & (d_y6["gauss_s2n"] < max_s2n)
                    & ((d_y6["gauss_T_ratio"] * d_y6["gauss_psf_T"]) < max_size)) 

        if 'flux' in selections:
            mag_g = mdet._compute_asinh_mags(d_y6["pgauss_band_flux_g"], 0)
            mag_r = mdet._compute_asinh_mags(d_y6["pgauss_band_flux_r"], 1)
            mag_i = mdet._compute_asinh_mags(d_y6["pgauss_band_flux_i"], 2)
            mag_z = mdet._compute_asinh_mags(d_y6["pgauss_band_flux_z"], 3)
            gmr = mag_g - mag_r
            rmi = mag_r - mag_i
            imz = mag_i - mag_z
            if rank == 0:
                print('flux selection')
            # flux/color; add extinction correction for the final version
            msk &= ((np.abs(gmr) < 5)
                    & (np.abs(rmi) < 5)
                    & (np.abs(imz) < 5)
                    & np.isfinite(mag_g)
                    & np.isfinite(mag_r)
                    & np.isfinite(mag_i)
                    & np.isfinite(mag_z)
                    & (mag_g < 26.5)
                    & (mag_r < 26.5)
                    & (mag_i < 26.2) # add 24.5 for the final version
                    & (mag_z < 25.6)
                    & (d_y6["mfrac"] < 0.1))

        if 'junk' in selections:
            if rank == 0:
                print('junk removal')
            # junks (pgauss and super-spreader)
            msk &= (d_y6["pgauss_T"] < (1.2 - 3.1*d_y6["pgauss_T_err"])) # change 1.6-3.1*x for the final version
            size_sizeerr = (d_y6['gauss_T_ratio']*d_y6['gauss_psf_T']) * d_y6['gauss_T_err']
            size_s2n = (d_y6['gauss_T_ratio']*d_y6['gauss_psf_T']) / d_y6['gauss_T_err']
            msk_superspreader = ((size_sizeerr > 1) & (size_s2n < 10))
            msk &= ~msk_superspreader

        np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/goldmatch/msk_info/'+selection+'_msk_'+i+'.npy', msk)
        d_y6 = d_y6[msk]

        cat1 = SkyCoord(ra=np.array(d_y6['ra'])*uu.degree, dec=np.array(d_y6['dec'])*uu.degree)
        cat2 = SkyCoord(ra=np.array(d_gold['RA'])*uu.degree, dec=np.array(d_gold['DEC'])*uu.degree)
        idx, d2d, d3d = cat1.match_to_catalog_sky(cat2, nthneighbor=1)
        np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/goldmatch/'+selection+'/mdety6_goldy6_matched_idx_'+i+'.npy', idx)
        np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/goldmatch/'+selection+'/mdety6_goldy6_matched_d2d_'+i+'.npy', d2d.deg)
        
    print('total number of objects before cuts', num_tot)