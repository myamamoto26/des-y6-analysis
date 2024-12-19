import fitsio as fio
import numpy as np
import h5py as h5
from astropy.coordinates import SkyCoord
from astropy import units as uu
import glob
from tqdm import tqdm
from des_y6utils import mdet
import healsparse

run = 'patch'
inpath = "/global/cfs/cdirs/des/y6-shear-catalogs"
hdf5cat = inpath + '/Y6A2_METADETECT_V6_UNBLINDED/metadetect_cutsv6_all.h5'
# gold_cat = "/global/cfs/cdirs/des/myamamot/gold/all/gold_all_master.fits"
gold_cat = "/global/cfs/cdirs/des/myamamot/gold/all/gold_all_master.fits"
match_hleda = False
selections = ['mask', 'sg_s2n']
selection = 'mask+sg_s2n' # 'basic' #'mask+sg+s2nsize+flux+junk'
footprint_map = healsparse.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6a2_decasu_griz_nexpgt2_and_fp2_footprint.hsp")
gold_mask_map = healsparse.HealSparseMap.read("/global/cfs/cdirs/des/y6-shear-catalogs/y6a2_foreground_mask_healsparse_nside16384.fits.gz")
hmap = healsparse.HealSparseMap.read('/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hleda-gaiafull-des-stars-hsmap131k-mdet-extra-masks-v2.hsp')


def read_mdet_h5(datafile, gal_weight_file, keys, mdet_step, patch_id=None, response=False, subtract_mean_shear=False, ):

    def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
        """
        Computes indices of 2D grids. Only used when we use shear weight that is binned by S/N and size ratio. 
        """
        from math import log10
        # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps

        logstepx = log10(xmax/xmin)/xsteps
        logstepy = log10(ymax/ymin)/ysteps

        indexx = (np.log10(x/xmin)/logstepx).astype(int)
        indexy = (np.log10(y/ymin)/logstepy).astype(int)

        indexx = np.maximum(indexx,0)
        indexx = np.minimum(indexx, xsteps-1)
        indexy = np.maximum(indexy,0)
        indexy = np.minimum(indexy, ysteps-1)

        return indexx,indexy
    
    def assign_grid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
        # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps
        
        xbins=np.linspace(xmin, xmax, xsteps+1)
        ybins=np.linspace(ymin, ymax, ysteps+1)

        indexx=np.digitize(x, xbins, right=True)
        indexy=np.digitize(y, ybins, right=True)

        indexx = indexx - 1
        indexy = indexy - 1

        indexx = np.maximum(indexx,0)
        indexx = np.minimum(indexx, xsteps-1)
        indexy = np.maximum(indexy,0)
        indexy = np.minimum(indexy, ysteps-1)

        return indexx,indexy
    

    def _find_shear_weight(dat, mask, wgt_dict, snmin, snmax, sizemin, sizemax, steps, mdet_mom):

        """
        Assigns shear weights to the objects based on the grids. 
        """
        
        if wgt_dict is None:
            weights = np.ones(len(dat))
            return weights

        shear_wgt = wgt_dict['weight']
        smoothing = True
        if smoothing:
            from scipy.ndimage import gaussian_filter
            smooth_response = gaussian_filter(wgt_dict['response'], sigma=2.0)
            shear_wgt = (smooth_response/wgt_dict['meanes'])**2
        indexx, indexy = assign_loggrid(np.array(dat[mdet_mom+'_s2n'])[mask], np.array(dat[mdet_mom+'_T_ratio'])[mask], snmin, snmax, steps, sizemin, sizemax, steps)
        weights = np.array([shear_wgt[x, y] for x, y in zip(indexx, indexy)])
        
        return weights

    def _get_shear_weights(dat, mask, gal_weight_file, shape_err=False):
        if shape_err:
            return 1/(0.22**2 + 0.5*(np.array(dat['gauss_g_cov_1_1'])[mask] + np.array(dat['gauss_g_cov_2_2'])[mask]))
        else:
            with open(gal_weight_file, 'rb') as handle:
                wgt_dict = pickle.load(handle)
                snmin = wgt_dict['xedges'][0]
                snmax = wgt_dict['xedges'][-1]
                sizemin = wgt_dict['yedges'][0]
                sizemax = wgt_dict['yedges'][-1]
                steps = len(wgt_dict['xedges'])-1
            shear_wgt = _find_shear_weight(dat, mask, wgt_dict, snmin, snmax, sizemin, sizemax, steps, 'gauss')
            return shear_wgt

    def _wmean(q,w):
        return np.sum(q*w)/np.sum(w)
    
    import h5py as h5
    f = h5.File(datafile, 'r')
    d = f.get('/mdet/'+mdet_step)
    if patch_id is None:
        nrows = len(np.array( d['ra'] ))
        mask = np.ones(nrows)!=0
    else:
        mask = (np.array( d['patch_num'] ) == patch_id)
        nrows = len(np.array( d['ra'] )[mask])
    formats = []
    for key in keys:
        formats.append('f4')
    data = np.recarray(shape=(nrows,), formats=formats, names=keys)
    mags = {'g':0, 'r':1, 'i':2, 'z':3}
    for key in keys:  
        if key == 'w':
            data['w'] = _get_shear_weights(d, mask, gal_weight_file)
        elif key in ('g1', 'g2'):
            data[key] = np.array(d['gauss_'+key[0]+'_'+key[1]])[mask]
        elif key == 'gauss_T':
            data[key] = np.array(d['gauss_psf_T'])[mask] * np.array(d['gauss_T_ratio'])[mask]
        elif key == 'gmi':
            mag_g = mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_g"])[mask], 0)
            mag_i = mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_i"])[mask], 2)
            data[key] = mag_g - mag_i
        elif key in ['mag_g', 'mag_r', 'mag_i', 'mag_z']:
            mag = mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_"+key[-1]])[mask], mags[key[-1]])
            data[key] = mag
        else:
            data[key] = np.array(d[key])[mask]
    print('made recarray with hdf5 file')
    
    # response correction
    if response:
        d_2p = f.get('/mdet/2p')
        d_1p = f.get('/mdet/1p')
        d_2m = f.get('/mdet/2m')
        d_1m = f.get('/mdet/1m')
        # compute response with weights
        g1p = _wmean(np.array(d_1p["gauss_g_1"]), _get_shear_weights(d_1p, gal_weight_file))                                     
        g1m = _wmean(np.array(d_1m["gauss_g_1"]), _get_shear_weights(d_1m, gal_weight_file))
        R11 = (g1p - g1m) / 0.02

        g2p = _wmean(np.array(d_2p["gauss_g_2"]), _get_shear_weights(d_2p, gal_weight_file))
        g2m = _wmean(np.array(d_2m["gauss_g_2"]), _get_shear_weights(d_2m, gal_weight_file))
        R22 = (g2p - g2m) / 0.02

        R = (R11 + R22)/2.
        data['g1'] /= R
        data['g2'] /= R

        mean_g1 = _wmean(data['g1'], data['w'])
        mean_g2 = _wmean(data['g2'], data['w'])
        std_g1 = np.var(data['g1'])
        std_g2 = np.var(data['g2'])
        mean_shear = [mean_g1, mean_g2, std_g1, std_g2]
        # mean shear subtraction
        if subtract_mean_shear:
            print('subtracting mean shear')
            print('mean g1 g2 =(%1.8f,%1.8f)'%(mean_g1, mean_g2))          
            data['g1'] -= mean_g1
            data['g2'] -= mean_g2

    return data


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
    

def apply_gold_mask(d, ra_col, dec_col, hleda=False):

    ra = np.array(d[ra_col])
    dec = np.array(d[dec_col])
    in_gold_footprint = footprint_map.get_values_pos(ra, dec, valid_mask=True)
    in_gold_mask = gold_mask_map.get_values_pos(ra, dec, valid_mask=True)
    msk = (in_gold_footprint & ~in_gold_mask)
    
    return msk


def apply_mdet_mask(d, ra_col, dec_col, hleda=False):

    ra = np.array(d[ra_col])
    dec = np.array(d[dec_col])
    
    in_footprint = hmap.get_values_pos(ra, dec, valid_mask=True)
    msk = in_footprint
    
    return msk

if run == 'master':
    f = h5.File(hdf5cat, 'r')
    d_y6 = f.get('/mdet/noshear')
    f_gold = fio.FITS(gold_cat)
    d_gold = f_gold[-1].read(columns=['RA', 'DEC'])
    msk_gold = apply_gold_mask(d_gold, 'RA', 'DEC')
    msk_mdet = apply_mdet_mask(d_gold, 'RA', 'DEC')
    d_gold = d_gold[msk_gold & msk_mdet]
    # f_y3 = h5.File('/global/cfs/cdirs/des/www/y3_cats/Y3_mastercat___UNBLIND___final_v1.1_12_22_20.h5', 'r')
    # m = np.array(f_y3.get('/index/select'))
    # d_y3 = f_y3.get('/catalog/metacal/unsheared')

    msk_gold = apply_gold_mask(d_y6, 'ra', 'dec')
    ra_y6 = np.array(d_y6['ra']) #[msk_gold]
    dec_y6 = np.array(d_y6['dec']) #[msk_gold]
    if match_hleda:
        match_hyperleda(d_y6, msk_mdet)
    print(len(ra_y6))
    cat1 = SkyCoord(ra=ra_y6*uu.degree, dec=dec_y6*uu.degree)
    cat2 = SkyCoord(ra=np.array(d_gold['RA'])*uu.degree, dec=np.array(d_gold['DEC'])*uu.degree)
    idx, d2d, d3d = cat1.match_to_catalog_sky(cat2, nthneighbor=1)
    np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v6_UNBLINDED/goldmatch/gold_allcuts_shear_allcuts/mdety6_goldy6_nogoldmask_matched_idx.npy', idx)
    np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v6_UNBLINDED/goldmatch/gold_allcuts_shear_allcuts/mdety6_goldy6_nogoldmask_matched_d2d.npy', d2d.deg)
    # np.save('/pscratch/sd/m/myamamot/gold/mdet_matched_d3d.npy', d3d)
elif run == 'patch':

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank, size)

    f_gold = fio.FITS(gold_cat)
    d_gold = f_gold[-1].read(columns=['RA', 'DEC'])
    mdet_f = sorted(glob.glob('/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V6_UNBLINDED/metadetect_cutsv6_patch*.h5'))

    
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

        i = f.split('/')[-1][23:26] #[6:10]
        keys = ['ra', 'dec', 'gauss_s2n', 'pgauss_band_flux_g', 'pgauss_band_flux_r', 'pgauss_band_flux_i', 'pgauss_band_flux_z', \
               'mfrac', 'gauss_T_ratio', 'gauss_psf_T', 'pgauss_T', 'gauss_T', 'pgauss_T_err', 'gauss_T_err']
        wgt_file = None
        d_y6 = read_mdet_h5(f, wgt_file, keys, 'noshear', response=False, subtract_mean_shear=False)
        # d_y6 = mdet.add_extinction_correction_columns(d_y6)
        num_tot += len(d_y6)

        # basic
        nrows = len(np.array( d_y6['ra'] ))
        msk = np.ones(nrows)!=0
        # in_gold_footprint = footprint_map.get_values_pos(d_y6["ra"], d_y6["dec"], valid_mask=True)
        # in_gold_mask = gold_mask_map.get_values_pos(d_y6["ra"], d_y6["dec"], valid_mask=True)
        # msk_mask = (in_gold_footprint & ~in_gold_mask)
        # msk &= msk_mask

        # add selections
        if 'mask' in selections:
            if rank == 0:
                print('masking out footprint')
            in_footprint = hmap.get_values_pos(d_y6["ra"], d_y6["dec"], valid_mask=True)
            msk &= in_footprint

        if 'sg_s2n' in selections:
            if rank == 0:
                print('doing s/g sep')
            # s/g sep
            n_terr = 0
            msk &= (d_y6["gauss_T_ratio"] >= np.maximum(
                    0.5,
                    (n_terr*d_y6["gauss_T_err"]/d_y6["gauss_psf_T"])))
            min_s2n = 10
            max_s2n = np.inf
            msk &= ((d_y6["gauss_s2n"] > min_s2n) & (d_y6["gauss_s2n"] < max_s2n))

        if 'size' in selections:
            if rank == 0:
                print('cutting s2n and size')
            # size
            max_size = 20.0
            msk &= ((d_y6["gauss_T_ratio"] * d_y6["gauss_psf_T"]) < max_size)

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
                    & (mag_i < 24.7)
                    & (mag_z < 25.6)
                    & (d_y6["mfrac"] < 0.1))

        if 'junk' in selections:
            if rank == 0:
                print('junk removal')
            # junks (pgauss and super-spreader)
            msk &= (d_y6["pgauss_T"] < (1.6 - 3.1*d_y6["pgauss_T_err"]))
            size_sizeerr = (d_y6['gauss_T_ratio']*d_y6['gauss_psf_T']) * d_y6['gauss_T_err']
            size_s2n = (d_y6['gauss_T_ratio']*d_y6['gauss_psf_T']) / d_y6['gauss_T_err']
            msk_superspreader = ((size_sizeerr > 1) & (size_s2n < 10))
            msk &= ~msk_superspreader

        np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v6_UNBLINDED/goldmatch/msk_info/'+selection+'_msk_'+i+'.npy', msk)
        d_y6 = d_y6[msk]

        cat1 = SkyCoord(ra=np.array(d_y6['ra'])*uu.degree, dec=np.array(d_y6['dec'])*uu.degree)
        cat2 = SkyCoord(ra=np.array(d_gold['RA'])*uu.degree, dec=np.array(d_gold['DEC'])*uu.degree)
        idx, d2d, d3d = cat1.match_to_catalog_sky(cat2, nthneighbor=1)
        np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v6_UNBLINDED/goldmatch/'+selection+'/mdety6_goldy6_matched_idx_'+i+'.npy', idx)
        np.save('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v6_UNBLINDED/goldmatch/'+selection+'/mdety6_goldy6_matched_d2d_'+i+'.npy', d2d.deg)
        
    print('total number of objects before cuts', num_tot)