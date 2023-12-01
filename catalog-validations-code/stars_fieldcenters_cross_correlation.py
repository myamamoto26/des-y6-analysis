

####################################################################
#  y6 shear catalog tests (This does not include PSF diagnostics)  #
####################################################################

import os, sys
from tqdm import tqdm
import numpy as np
import fitsio as fio
from astropy.io import fits
import matplotlib as mpl
from des_y6utils import mdet
import glob
import treecorr
import healsparse
# sys.path.append('./download-query-concatenation-code')
# from query_examples import query_field_centers, query_coadd_info

def flux2mag(flux, zero_pt=30):
    return zero_pt - 2.5 * np.log10(flux)

def get_shear_weights(d, weight_type, wgt_filepath, mask=None):
    """
    Get shear weights. The shear weight could be computed from
    objects S/N and size ratio, shape error or uniformly.
    """
    import numpy as np
    import pickle

    def _assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
        from math import log10
        # return x and y indices of data (x,y) on a log-spaced grid that runs from
        # [xy]min to [xy]max in [xy]steps

        logstepx = log10(xmax / xmin) / xsteps
        logstepy = log10(ymax / ymin) / ysteps

        indexx = (np.log10(x / xmin) / logstepx).astype(int)
        indexy = (np.log10(y / ymin) / logstepy).astype(int)

        indexx = np.maximum(indexx, 0)
        indexx = np.minimum(indexx, xsteps - 1)
        indexy = np.maximum(indexy, 0)
        indexy = np.minimum(indexy, ysteps - 1)

        return indexx, indexy

    def _find_shear_weight(d, wgt_dict, snmin, snmax, sizemin, sizemax, steps, mask=None):

        if wgt_dict is None:
            weights = np.ones(len(d))
            return weights

        shear_wgt = wgt_dict['weight']
        smoothing = True
        if smoothing:
            from scipy.ndimage import gaussian_filter
            smooth_response = gaussian_filter(wgt_dict['response'], sigma=2.0)
            shear_wgt = (smooth_response/wgt_dict['meanes'])**2
        if mask is None:
            indexx, indexy = _assign_loggrid(
                np.array(d['gauss_s2n']), np.array(d['gauss_T_ratio']), snmin, snmax, steps, sizemin, sizemax, steps)
        else:
            indexx, indexy = _assign_loggrid(
                np.array(d['gauss_s2n'])[mask], np.array(d['gauss_T_ratio'])[mask], snmin, snmax, steps, sizemin, sizemax, steps)
        weights = np.array([shear_wgt[x, y] for x, y in zip(indexx, indexy)])

        return weights

    # import pdb ; pdb.set_trace()
    if weight_type == 's2n_sizer':
        # pickle file that defines w(S/N, size)
        with open(wgt_filepath, 'rb') as handle:
            wgt_dict = pickle.load(handle)
        # TO-DO: make snmin, snmax, sizemin, sizemax available in config file.
        shear_wgt = _find_shear_weight(d, wgt_dict, 10, 1000, 0.5, 5.0, 20, mask=mask)
    elif weight_type == 'shape_err':
        if mask is None:
            shear_wgt = 1 / (0.22**2 + 0.5 * (np.array(d['gauss_g_cov_1_1']) + np.array(d['gauss_g_cov_2_2'])))
        else:
            shear_wgt = 1 / (0.22**2 + 0.5 * (np.array(d['gauss_g_cov_1_1'])[mask] + np.array(d['gauss_g_cov_2_2'])[mask]))
    elif weight_type == 'uniform':
        shear_wgt = np.ones(len(d['ra']))

    shear_wgt[np.isnan(shear_wgt)] = 0.

    return shear_wgt

def read_mdet_h5(datafile, keys, wgt_type, wgt_filepath, response=False, subtract_mean_shear=False, mask=None, color_split=False):

    def _wmean(q,w):
        return np.sum(q*w)/np.sum(w)
    def _make_cut(d, mask_map):
        hmap = healsparse.HealSparseMap.read(mask_map)
        in_footprint = hmap.get_values_pos(np.array(d['ra']), np.array(d['dec']), valid_mask=True)
        return in_footprint
    def _make_color_cut(d, cmin, cmax):
        gmi =  mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_g"]), 0) - mdet._compute_asinh_mags(np.array(d["pgauss_band_flux_i"]), 2)
        msk = ((gmi > cmin) & (gmi < cmax))
        return msk
    
    import h5py as h5
    f = h5.File(datafile, 'r')
    d = f.get('/mdet/noshear')
    nrows = len(np.array( d['ra'] ))
    formats = []
    for key in keys:
        formats.append('f4')
    data = np.recarray(shape=(nrows,), formats=formats, names=keys)
    for key in keys:  
        if key == 'w':
            data['w'] = get_shear_weights(d, wgt_type, wgt_filepath)
        elif key in ('g1', 'g2'):
            data[key] = np.array(d['gauss_'+key[0]+'_'+key[1]])
        else:
            data[key] = np.array(d[key])
    print('made recarray with hdf5 file')
    if mask is not None:
        print('additional mask included')
        msk_footprint = _make_cut(data, mask)
        data = data[msk_footprint]
    if color_split:
        print('splitting color samples')
        # color splits: blue-[-2.00, 0.76], mid-[0.76, 1.49], red-[1.49, 4.00]
        gmi =  mdet._compute_asinh_mags(data["pgauss_band_flux_g"], 0) - mdet._compute_asinh_mags(data["pgauss_band_flux_i"], 2)
        dmin = 0.76
        dmax = 1.49
        msk_color = ((gmi > dmin) & (gmi <= dmax))
        data = data[msk_color]
    
    # response correction
    if response:
        d_2p = f.get('/mdet/2p')
        d_1p = f.get('/mdet/1p')
        d_2m = f.get('/mdet/2m')
        d_1m = f.get('/mdet/1m')
        if mask is None:
            # compute response with weights
            if color_split:
                print('color split included')
                msk_1p = _make_color_cut(d_1p, dmin, dmax)
                g1p = _wmean(np.array(d_1p["gauss_g_1"])[msk_1p], get_shear_weights(d_1p, wgt_type, wgt_filepath, mask=msk_1p))
                msk_1m = _make_color_cut(d_1m, dmin, dmax)
                g1m = _wmean(np.array(d_1m["gauss_g_1"])[msk_1m], get_shear_weights(d_1m, wgt_type, wgt_filepath, mask=msk_1m))
                msk_2p = _make_color_cut(d_2p, dmin, dmax)
                g2p = _wmean(np.array(d_2p["gauss_g_2"])[msk_2p], get_shear_weights(d_2p, wgt_type, wgt_filepath, mask=msk_2p))
                msk_2m = _make_color_cut(d_2m, dmin, dmax)
                g2m = _wmean(np.array(d_2m["gauss_g_2"])[msk_2m], get_shear_weights(d_2m, wgt_type, wgt_filepath, mask=msk_2m))
            else:
                g1p = _wmean(np.array(d_1p["gauss_g_1"]), get_shear_weights(d_1p, wgt_type, wgt_filepath))
                g1m = _wmean(np.array(d_1m["gauss_g_1"]), get_shear_weights(d_1m, wgt_type, wgt_filepath))
                g2p = _wmean(np.array(d_2p["gauss_g_2"]), get_shear_weights(d_2p, wgt_type, wgt_filepath))
                g2m = _wmean(np.array(d_2m["gauss_g_2"]), get_shear_weights(d_2m, wgt_type, wgt_filepath))
        else:
            print('additional mask included')
            msk_1p = _make_cut(d_1p, mask)
            g1p = _wmean(np.array(d_1p["gauss_g_1"])[msk_1p], get_shear_weights(d_1p, wgt_type, wgt_filepath, mask=msk_1p))
            msk_1m = _make_cut(d_1m, mask)
            g1m = _wmean(np.array(d_1m["gauss_g_1"])[msk_1m], get_shear_weights(d_1m, wgt_type, wgt_filepath, mask=msk_1m))
            msk_2p = _make_cut(d_2p, mask)
            g2p = _wmean(np.array(d_2p["gauss_g_2"])[msk_2p], get_shear_weights(d_2p, wgt_type, wgt_filepath, mask=msk_2p))
            msk_2m = _make_cut(d_2m, mask)
            g2m = _wmean(np.array(d_2m["gauss_g_2"])[msk_2m], get_shear_weights(d_2m, wgt_type, wgt_filepath, mask=msk_2m))

        R11 = (g1p - g1m) / 0.02
        R22 = (g2p - g2m) / 0.02
        R = (R11 + R22)/2.
        print(R)
        data['g1'] /= R
        data['g2'] /= R

    # mean shear subtraction
    if subtract_mean_shear:
        if color_split:
            data['g1'] -= 0.00020496
            data['g2'] -= -0.00003086
        else:
            g1 = _wmean(data["g1"],data["w"])
            g2 = _wmean(data["g2"],data["w"])
            print('mean g1 g2 =(%1.8f,%1.8f)'%(g1, g2))          
            data['g1'] -= g1
            data['g2'] -= g2

    return data


def stellar_location_contamination(mdet_response_filepath, mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts):

    """
    Computes NK correlation of star positions and e1,e2 of galaxies.
    """

    f_response = open(mdet_response_filepath, 'r')
    R11, R22 = f_response.read().split('\n')
    R = (float(R11) + float(R22))/2

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 1.0,
        max_sep = 200,
        nbins = 20,

        var_method = 'jackknife', 
        output_dots = False,
    )

    cat1_file = fits.open('/project/projectdirs/des/schutt20/catalogs/y6a2_piff_v3_allres_v3_collated.fits')
    f_pc = '/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed8888.fits'
    d_piff=cat1_file[1].data
    msk = ((d_piff.field("BAND") == "r")) # | (d_piff.field("BAND") == "i") | (d_piff.field("BAND") == "z"))
    ra_piff = d_piff.field('RA')[msk]
    dec_piff = d_piff.field('DEC')[msk]
    flux_piff = d_piff.field('FLUX')[msk]
    
    mask_bright = (flux2mag(flux_piff) < 16.5)
    mask_faint = (flux2mag(flux_piff) > 16.5)
    cat1_bright = treecorr.Catalog(ra=ra_piff[mask_bright], dec=dec_piff[mask_bright], ra_units='deg', dec_units='deg', patch_centers=f_pc)
    cat1_faint = treecorr.Catalog(ra=ra_piff[mask_faint], dec=dec_piff[mask_faint], ra_units='deg', dec_units='deg', patch_centers=f_pc)

    # random point subtraction. 
    cat2_files = sorted(glob.glob(mdet_input_filepath))
    nk_bright_g1 = treecorr.NKCorrelation(bin_config, verbose=2)
    nk_bright_g2 = treecorr.NKCorrelation(bin_config, verbose=2)
    nk_faint_g1 = treecorr.NKCorrelation(bin_config, verbose=2)
    nk_faint_g2 = treecorr.NKCorrelation(bin_config, verbose=2)
    nk_bright_T = treecorr.NKCorrelation(bin_config, verbose=2)
    nk_faint_T = treecorr.NKCorrelation(bin_config, verbose=2)
    for i,cat2_f in tqdm(enumerate(cat2_files)):
        d_mdet = fio.read(cat2_f)
        msk = mdet.make_mdet_cuts(d_mdet, mdet_cuts) 
        msk_noshear = (d_mdet['mdet_step']=='noshear')

        d_mdet = d_mdet[msk & msk_noshear]
        g1 = d_mdet[mdet_mom+'_g_1']/R
        g2 = d_mdet[mdet_mom+'_g_2']/R
        T = d_mdet[mdet_mom+'_T_ratio'] * d_mdet[mdet_mom+'_psf_T']
        # cat2_g1 = treecorr.Catalog(ra=d_mdet['ra'], dec=d_mdet['dec'], ra_units='deg', dec_units='deg', k=g1, patch=i, npatch=200)
        # cat2_g2 = treecorr.Catalog(ra=d_mdet['ra'], dec=d_mdet['dec'], ra_units='deg', dec_units='deg', k=g2, patch=i, npatch=200)
        cat2_T = treecorr.Catalog(ra=d_mdet['ra'], dec=d_mdet['dec'], ra_units='deg', dec_units='deg', k=T, patch=i, npatch=200)

        # nk_bright_g1.process(cat1_bright, cat2_g1, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
        # nk_bright_g2.process(cat1_bright, cat2_g2, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
        # nk_faint_g1.process(cat1_faint, cat2_g1, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
        # nk_faint_g2.process(cat1_faint, cat2_g2, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
        nk_bright_T.process(cat1_bright, cat2_T, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
        nk_faint_T.process(cat1_faint, cat2_T, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
        # cat2_g1.unload()
        # cat2_g2.unload()
        cat2_T.unload()

    # nk_bright_g1.write(os.path.join(out_path, mdet_mom+'_stars_pos_g1_correlation_final_output_bright.fits'))
    # nk_bright_g2.write(os.path.join(out_path, mdet_mom+'_stars_pos_g2_correlation_final_output_bright.fits'))
    # nk_faint_g1.write(os.path.join(out_path, mdet_mom+'_stars_pos_g1_correlation_final_output_faint.fits'))
    # nk_faint_g2.write(os.path.join(out_path, mdet_mom+'_stars_pos_g2_correlation_final_output_faint.fits'))
    nk_bright_T.write(os.path.join(out_path, mdet_mom+'_stars_pos_size_correlation_final_output_bright.fits'))
    nk_faint_T.write(os.path.join(out_path, mdet_mom+'_stars_pos_size_correlation_final_output_faint.fits'))


def shear_stellar_contamination(mdet_response_filepath, mdet_input_filepath, piff_input, mdet_mom, out_path, random_point_map, mdet_cuts, weight_scheme, star_weight_map, simulations=False, parallel=False):

    """
    Copmutes the correlation function between bright/faint stars and shear. 

    Parameters
    ----------
    mdet_response_filepath: the file path for the shear response over the catalogs
    Example) /global/cscratch1/sd/myamamot/metadetect/shear_response_v2.txt

    mdet_input_filepath: the input file path for the metadetection catalogs
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v2/*_metadetect-v5_mdetcat_part0000.fits

    mdet_mom: which of the shape measurement to use (wmom, pgauss etc.)

    out_path: the output file path
    Example)  /global/cscratch1/sd/myamamot/metadetect/systematics
    """

    import treecorr
    import glob

    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
    print(rank)

    def flux2mag(flux, zero_pt=30):
        return zero_pt - 2.5 * np.log10(flux)

    f_response = open(mdet_response_filepath, 'r')
    R11, R22 = f_response.read().split('\n')
    R = (float(R11) + float(R22))/2

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 1.0,
        max_sep = 200,
        nbins = 20,

        var_method = 'bootstrap', 
        output_dots = False,
    )
    # gal_data = fio.read('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5/metadetection_v5_flat_shape_err.fits')
    star_weight_map = healsparse.HealSparseMap.read(star_weight_map)
    f_pc = '/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed8888.fits'
    d_piff = fio.read(piff_input)
    ra_piff = d_piff['RA']
    dec_piff = d_piff['DEC']
    flux_piff = d_piff['FLUX']
    weight_stars = star_weight_map.get_values_pos(ra_piff, dec_piff, lonlat=True)
    
    mask_bright = (flux2mag(flux_piff) < 16.5)
    mask_faint = (flux2mag(flux_piff) > 16.5)
    cat1_bright = treecorr.Catalog(ra=ra_piff[mask_bright], dec=dec_piff[mask_bright], w=weight_stars[mask_bright], ra_units='deg', dec_units='deg', patch_centers=f_pc)
    cat1_faint = treecorr.Catalog(ra=ra_piff[mask_faint], dec=dec_piff[mask_faint], w=weight_stars[mask_faint], ra_units='deg', dec_units='deg', patch_centers=f_pc)

    # random point subtraction. 
    rdata = fio.read(random_point_map)
    random_weghts = star_weight_map.get_values_pos(rdata['ra'], rdata['dec'], lonlat=True)
    cat1r = treecorr.Catalog(ra=rdata['ra'], dec=rdata['dec'], ra_units='deg', dec_units='deg', w=random_weghts, patch_centers=f_pc)

    cat2_files = sorted(glob.glob(mdet_input_filepath))
    ng_bright = treecorr.NGCorrelation(bin_config, verbose=2)
    ng_faint = treecorr.NGCorrelation(bin_config, verbose=2)
    ng_rand = treecorr.NGCorrelation(bin_config, verbose=2)
    if simulations:
        import pickle
        print('running correlations for sims...')
        for seed in tqdm(range(1,201)):
            if seed % size != rank:
                continue
            else:
                if not (os.path.exists(os.path.join(out_path, 'sims/bright_stars_cross_correlation_seed_'+str(seed)+'.fits')) & os.path.exists(os.path.join(out_path, 'sims/faint_stars_cross_correlation_seed_'+str(seed)+'.fits'))):
                    with open('/pscratch/sd/m/myamamot/sample_variance/v5_catalog/seed__fid_'+str(seed)+'.pkl', 'rb') as f:
                        d_sim = pickle.load(f)['sources'][0]
                    cat2 = treecorr.Catalog(ra=d_sim['ra'], dec=d_sim['dec'], ra_units='deg', dec_units='deg', g1=d_sim['e1'], g2=d_sim['e2'], patch_centers=f_pc)
                    ng_bright.process(cat1_bright, cat2, low_mem=True)
                    ng_faint.process(cat1_faint, cat2, low_mem=True)
                    ng_rand.process(cat1r, cat2, low_mem=True)
                    ng_bright.write(os.path.join(out_path, 'sims/bright_stars_cross_correlation_seed_'+str(seed)+'.fits'), rg=ng_rand)
                    ng_faint.write(os.path.join(out_path, 'sims/faint_stars_cross_correlation_seed_'+str(seed)+'.fits'), rg=ng_rand)
    else:
        for i,cat2_f in tqdm(enumerate(cat2_files)):
            d_mdet = fio.read(cat2_f)
            msk = mdet.make_mdet_cuts(d_mdet, mdet_cuts) 
            msk_noshear = (d_mdet['mdet_step']=='noshear')

            d_mdet = d_mdet[msk & msk_noshear]
            g1 = d_mdet[mdet_mom+'_g_1']/R
            g2 = d_mdet[mdet_mom+'_g_2']/R
            if weight_scheme == 'shape_err':
                w = 1/(0.17**2 + 0.5*(d_mdet[mdet_mom+'_g_cov_1_1'] + d_mdet[mdet_mom+'_g_cov_2_2']))
            cat2 = treecorr.Catalog(ra=d_mdet['ra'], dec=d_mdet['dec'], ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch=i, w=w, npatch=200)

            ng_bright.process(cat1_bright, cat2, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
            ng_faint.process(cat1_faint, cat2, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
            ng_rand.process(cat1r, cat2, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
            cat2.unload()

        ng_bright.write(os.path.join(out_path, mdet_mom+'_weighted_stars_shear_cross_correlation_final_output_bootstrap_bright.fits'), rg=ng_rand)
        ng_bright.calculateXi(rg=ng_rand)
        ng_bright_cov = ng_bright.cov
        np.save(os.path.join(out_path, mdet_mom+'_weighted_stars_shear_cross_correlation_final_output_bootstrap_bright_cov.npy'), ng_bright_cov)
        
        ng_faint.write(os.path.join(out_path, mdet_mom+'_weighted_stars_shear_cross_correlation_final_output_bootstrap_faint.fits'), rg=ng_rand)
        ng_faint.calculateXi(rg=ng_rand)
        ng_faint_cov = ng_faint.cov
        np.save(os.path.join(out_path, mdet_mom+'_weighted_stars_shear_cross_correlation_final_output_bootstrap_faint_cov.npy'), ng_faint_cov)


def shear_stellar_contamination_hdf5(mdet_input_filepath, piff_input, mdet_mom, out_path, random_point_map, mdet_cuts, weight_scheme, wgt_filepath, star_weight_map, var_method='bootstrap', parallel=False):

    import treecorr
    import glob
    import h5py as h5

    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        comm = None
        rank = 0
    print(rank)

    def flux2mag(flux, zero_pt=30):
        return zero_pt - 2.5 * np.log10(flux)
    
    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 1.0,
        max_sep = 200,
        nbins = 20,

        var_method = var_method, 
        output_dots = False,
    )

    star_weight_map = healsparse.HealSparseMap.read(star_weight_map)
    f_pc = '/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed8888.fits'
    d_piff = fio.read(piff_input)
    ra_piff = d_piff['RA']
    dec_piff = d_piff['DEC']
    flux_piff = d_piff['FLUX']
    weight_stars = star_weight_map.get_values_pos(ra_piff, dec_piff, lonlat=True)
    
    mask_bright = (flux2mag(flux_piff) < 16.5)
    mask_faint = (flux2mag(flux_piff) > 16.5)
    cat1_bright = treecorr.Catalog(ra=ra_piff[mask_bright], dec=dec_piff[mask_bright], w=weight_stars[mask_bright], ra_units='deg', dec_units='deg', patch_centers=f_pc)
    cat1_faint = treecorr.Catalog(ra=ra_piff[mask_faint], dec=dec_piff[mask_faint], w=weight_stars[mask_faint], ra_units='deg', dec_units='deg', patch_centers=f_pc)

    # random point subtraction. 
    cat1r_ = fio.read(random_point_map)
    random_weights = star_weight_map.get_values_pos(cat1r_['ra'], cat1r_['dec'], lonlat=True)
    cat1r = treecorr.Catalog(ra=cat1r_['ra'], dec=cat1r_['dec'], ra_units='deg', dec_units='deg', patch_centers=f_pc) # w=random_weights,
    
    # mdet file
    keys = ['ra', 'dec', 'g1', 'g2', 'w']
    gal_data = read_mdet_h5(mdet_input_filepath, keys, weight_scheme, wgt_filepath, response=True, subtract_mean_shear=True, mask='/global/cfs/cdirs/des/y6-shear-catalogs/y6a2_foreground_mask_v1.3.hs')
    cat2 = treecorr.Catalog(ra=gal_data['ra'], dec=gal_data['dec'], ra_units='deg', dec_units='deg', g1=gal_data['g1'], g2=gal_data['g2'], w=gal_data['w'], patch_centers=f_pc)
    ng_bright = treecorr.NGCorrelation(bin_config, verbose=2)
    ng_faint = treecorr.NGCorrelation(bin_config, verbose=2)
    ng_rand = treecorr.NGCorrelation(bin_config, verbose=2)

    ng_bright.process(cat1_bright, cat2, low_mem=True, comm=comm)
    ng_faint.process(cat1_faint, cat2, low_mem=True, comm=comm)
    ng_rand.process(cat1r, cat2, low_mem=True, comm=comm)

    if rank==0:
        ng_bright.write(os.path.join(out_path, mdet_mom+'_weighted_stars_shear_cross_correlation_final_output_'+var_method+'_bright_hdf5.fits'), rg=ng_rand)
        ng_bright.calculateXi(rg=ng_rand)
        ng_bright_cov = ng_bright.cov
        np.save(os.path.join(out_path, mdet_mom+'_weighted_stars_shear_cross_correlation_final_output_'+var_method+'_bright_cov_hdf5.npy'), ng_bright_cov)
        
        ng_faint.write(os.path.join(out_path, mdet_mom+'_weighted_stars_shear_cross_correlation_final_output_'+var_method+'_faint_hdf5.fits'), rg=ng_rand)
        ng_faint.calculateXi(rg=ng_rand)
        ng_faint_cov = ng_faint.cov
        np.save(os.path.join(out_path, mdet_mom+'_weighted_stars_shear_cross_correlation_final_output_'+var_method+'_faint_cov_hdf5.npy'), ng_faint_cov)


# Figure 14; Tangential shear around field center
def tangential_shear_field_center(fs, mdet_response_filepath, mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts, weight_scheme, simulations=False, parallel=False):

    """
    Creates a fits file that contains the exposure number and field centers (RA, DEC) from desoper. 
    

    Parameters
    ----------
    fs: the file that contains all the filenames for metadetection catalog

    mdet_response_filepath: the file path for the shear response over the catalogs
    Example) /global/cscratch1/sd/myamamot/metadetect/shear_response_v2.txt

    mdet_input_filepath: the input file path for the metadetection catalogs
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v2/*_metadetect-v5_mdetcat_part0000.fits

    mdet_mom: which of the shape measurement to use (wmom, pgauss etc.)

    out_path: the output file path
    Example)  /global/cscratch1/sd/myamamot/metadetect/systematics

    random_point_map: Random point map created from the healsparse mask map. y6shear_figures.ipynb has the code. 
    Example) /global/homes/m/myamamot/DES/des-y6-analysis/y6-combined-hsmap_random.fits

    coadd_tag: The file that is tagged in desoper database. 
    Example) Y6A2_PIZACUTTER
    """

    from matplotlib import pyplot as plt
    import tqdm
    from tqdm import tqdm
    from compute_shear_response import compute_response_over_catalogs
    import treecorr
    import glob

    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
    print(rank)
             
    def find_exposure_numbers(mdet_fs):

        """
        Finds the list of exposure numbers (ID) from pizza-cutter files.
        """

        use_bands = ['r', 'i', 'z']
        mdet_filenames = [fname.split('/')[-1] for fname in mdet_fs]
        tilenames = [d.split('_')[0] for d in mdet_filenames]

        if not os.path.exists('/global/cfs/cdirs/des/myamamot/pizza-slice/pizza-cutter-coadds-info.fits'):
            out_fname = '/global/cfs/cdirs/des/myamamot/pizza-slice/pizza-cutter-coadds-info.fits'
            query_coadd_info(out_fname, 'Y6A2_PIZZACUTTER_V3')
        coadd_info = fio.read('/global/cfs/cdirs/des/myamamot/pizza-slice/pizza-cutter-coadds-info.fits')
        coadd_files = {t: [] for t in tilenames}
        coadd_paths = {t: [] for t in tilenames}
        for coadd in coadd_info:
            tname = coadd['FILENAME'].split('_')[0]
            fname = coadd['FILENAME'] + coadd['COMPRESSION']
            if fname.split('_')[2] in use_bands:
                if tname in list(coadd_files.keys()):
                    coadd_files[tname].append(fname)
                    coadd_paths[tname].append(os.path.join('/global/cfs/cdirs/des/myamamot/pizza-slice/data', fname))
            else:
                continue

        exp_num = []
        ccd_num = []
        existing_coadd_filepaths = glob.glob('/global/cfs/cdirs/des/myamamot/pizza-slice/data/*.fits.fz', recursive=True)
        for t in tqdm(tilenames):
            for pizza_f in coadd_paths[t]:
                if pizza_f not in existing_coadd_filepaths:
                    print(pizza_f)
                    continue

                coadd = fio.FITS(pizza_f)
                try:
                    epochs = coadd['epochs_info'].read()
                    image_info = coadd['image_info'].read()
                except OSError:
                    print('Corrupt file.?', pizza_f)
                    raise OSError
                    
                image_id = np.unique(epochs[(epochs['flags']==0)]['image_id'])
                image_id = image_id[image_id != 0]
                for iid in image_id:
                    msk_im = np.where(image_info['image_id'] == iid)
                    # ccdnum = _get_ccd_num(image_info['image_path'][msk_im][0])
                    # ccd_num.append(ccdnum)
                    expnum = _get_exp_num(image_info['image_path'][msk_im][0])
                    exp_num.append(expnum)
        exp_num = np.unique(np.array(exp_num))
        total_exp_num = len(exp_num)
        print('total exposure number', total_exp_num)

        # ccd_exp_num = list(set([(ccd, exp) for ccd, exp in zip(ccd_num, exp_num)]))
        # with open('/global/cscratch1/sd/myamamot/pizza-slice/theo_ccd_exp_num.txt', 'w') as f:
        #     for l in ccd_exp_num:
        #         f.write(str(l[0])+', '+str(l[1]))
        #         f.write('\n')
        with open('/pscratch/sd/m/myamamot/pizza-slice/ccd_exp_num.txt', 'w') as f:
            for l in exp_num:
                f.write(str(l))
                f.write('\n')

        return None
    
    def _get_exp_num(image_path):
        return int(image_path.split('/')[1].split('_')[0][3:])

    def _get_ccd_num(image_path):
        return int(image_path.split('/')[1].split('_')[2][1:])
        
    # Compute the shear response over all the tiles. 
    mdet_filenames = [fname.split('/')[-1] for fname in fs]
    tilenames = [d.split('_')[0] for d in mdet_filenames]
    # f_response = open(mdet_response_filepath, 'r')
    # R11, R22 = f_response.read().split('\n')
    # R = (np.float64(R11) + np.float64(R22))/2
    f_pc = '/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed8888.fits'

    # Create ccdnum and expnum text file if it has not been created yet, and query from DESDM table. Should only be done once. 
    if not os.path.exists('/pscratch/sd/m/myamamot/pizza-slice/ccd_exp_num.txt'):
        find_exposure_numbers(fs)
        query_field_centers('/pscratch/sd/m/myamamot/pizza-slice/ccd_exp_num.txt', 300)
        print('done making ccd num file')
    
    expnum_field_centers = fio.read('/pscratch/sd/m/myamamot/pizza-slice/exposure_field_centers.fits')
    print('number of field centers', len(expnum_field_centers))

    bin_config = dict(
                sep_units = 'arcmin',
                bin_slop = 0.01,

                min_sep = 0.5,
                max_sep = 150,
                nbins = 20,

                var_method = 'bootstrap',
                output_dots = False,
                )

    cat1_file = '/pscratch/sd/m/myamamot/pizza-slice/exposure_field_centers.fits'
    cat1 = treecorr.Catalog(cat1_file, ra_col='RA_CENT', dec_col='DEC_CENT', ra_units='deg', dec_units='deg', patch_centers=f_pc)
    # random point subtraction. 
    cat1r_file = random_point_map
    cat1r = treecorr.Catalog(cat1r_file, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg', patch_centers=f_pc)
    ng_rand = treecorr.NGCorrelation(bin_config, verbose=2)
    ng = treecorr.NGCorrelation(bin_config, verbose=2)
    if simulations:
        import pickle
        print('running correlations for sims...')
        for seed in tqdm(range(1,801)):
            if seed % size != rank:
                continue
            else:
                if os.path.exists(os.path.join(out_path, 'sims/field_centers_cross_correlation_seed_'+str(seed)+'.fits')):
                    print('seed ', seed, ' skipping')
                    continue
                print('seed ', seed, ' running')
                with open('/pscratch/sd/m/myamamot/sample_variance/v5_catalog_cosmogrid/seed__fid_cosmogrid_'+str(seed)+'.pkl', 'rb') as f:
                    d_sim = pickle.load(f)['sources'][0]
                cat2 = treecorr.Catalog(ra=d_sim['ra'], dec=d_sim['dec'], ra_units='deg', dec_units='deg', g1=d_sim['e1'], g2=d_sim['e2'], patch_centers=f_pc)
                ng.process(cat1, cat2, low_mem=True)
                ng_rand.process(cat1r, cat2, low_mem=True)
                ng.write(os.path.join(out_path, 'sims/field_centers_cross_correlation_seed_'+str(seed)+'.fits'), rg=ng_rand)
                print('seed ', seed, ' done')
    else:
        cat2_files = sorted(glob.glob(mdet_input_filepath))
        for i,cat2_f in tqdm(enumerate(cat2_files)):
            d_mdet = fio.read(cat2_f)
            msk = mdet.make_mdet_cuts(d_mdet, mdet_cuts) 
            msk_noshear = (d_mdet['mdet_step']=='noshear')
            d_mdet = d_mdet[msk & msk_noshear]

            g1 = d_mdet[mdet_mom+'_g_1']/R
            g2 = d_mdet[mdet_mom+'_g_2']/R
            if weight_scheme == 'shape_err':
                w = 1/(0.17**2 + 0.5*(d_mdet[mdet_mom+'_g_cov_1_1'] + d_mdet[mdet_mom+'_g_cov_2_2']))
            cat2 = treecorr.Catalog(ra=d_mdet['ra'], dec=d_mdet['dec'], ra_units='deg', dec_units='deg', g1=g1, g2=g2, w=w, patch=i, npatch=200)
            ng.process(cat1, cat2, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
            ng_rand.process(cat1r, cat2, initialize=(i==0), finalize=(i==len(cat2_files)-1), low_mem=True)
            cat2.unload()

        ng.write(os.path.join(out_path, mdet_mom+'_field_centers_cross_correlation_final_output_bootstrap.fits'), rg=ng_rand)
        ng.calculateXi(rg=ng_rand)
        ng_cov = ng.cov
        np.save(os.path.join(out_path, mdet_mom+'_field_centers_cross_correlation_final_output_bootstrap_cov.npy'), ng_cov)


def _measure_tangential_shear_hdf5(mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts, weight_scheme, wgt_path, var_method, color_split, parallel=False):

    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        comm = None
        rank = 0
    print(rank)

    f_pc = '/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed8888.fits'
    expnum_field_centers = fio.read('/pscratch/sd/m/myamamot/pizza-slice/exposure_field_centers.fits')
    print('number of field centers', len(expnum_field_centers))

    bin_config = dict(
                sep_units = 'arcmin',
                bin_slop = 0.01,

                min_sep = 0.5,
                max_sep = 150,
                nbins = 20,

                var_method = var_method,
                output_dots = False,
                )
    cat1_file = '/pscratch/sd/m/myamamot/pizza-slice/exposure_field_centers.fits'
    cat1 = treecorr.Catalog(cat1_file, ra_col='RA_CENT', dec_col='DEC_CENT', ra_units='deg', dec_units='deg', patch_centers=f_pc)
    # random point subtraction. 
    cat1r_file = random_point_map
    cat1r = treecorr.Catalog(cat1r_file, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg', patch_centers=f_pc)
    ng_rand = treecorr.NGCorrelation(bin_config, verbose=2)
    ng = treecorr.NGCorrelation(bin_config, verbose=2)

    keys = ['ra', 'dec', 'g1', 'g2', 'w', 'pgauss_band_flux_g', 'pgauss_band_flux_i']
    gal_data = read_mdet_h5(mdet_input_filepath, keys, weight_scheme, wgt_path, response=True, subtract_mean_shear=True, color_split=color_split)

    cat2 = treecorr.Catalog(ra=gal_data['ra'], dec=gal_data['dec'], ra_units='deg', dec_units='deg', g1=gal_data['g1'], g2=gal_data['g2'], w=gal_data['w'], patch_centers=f_pc)
    ng.process(cat1, cat2, low_mem=True, comm=comm)
    ng_rand.process(cat1r, cat2, low_mem=True, comm=comm)

    if rank==0:
        ng.write(os.path.join(out_path, mdet_mom+'_field_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.5_max150_hdf5_norand.fits'))
        ng_rand.write(os.path.join(out_path, mdet_mom+'_field_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.5_max150_hdf5_randonly.fits'))
        ng.write(os.path.join(out_path, mdet_mom+'_field_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.5_max150_hdf5.fits'), rg=ng_rand)
        ng.calculateXi(rg=ng_rand)
        ng_cov = ng.cov
        np.save(os.path.join(out_path, mdet_mom+'_field_centers_cross_correlation_final_output_'+var_method+'_bins0.01_min0.5_max150_cov_hdf5.npy'), ng_cov)


def mean_shear_tomoz(gold_f, fs):

    def flux2mag(flux, zero_pt=30):
        return zero_pt - 2.5 * np.log10(flux)
    def _compute_g1g2(res, bind):
        g1 = res['noshear'][0][bind] / res['num_noshear'][0][bind]
        g1p = res['1p'][0][bind] / res['num_1p'][0][bind]
        g1m = res['1m'][0][bind] / res['num_1m'][0][bind]
        R11 = (g1p - g1m) / 2 / 0.01

        g2 = res['noshear'][1][bind] / res['num_noshear'][1][bind]
        g2p = res['2p'][1][bind] / res['num_2p'][1][bind]
        g2m = res['2m'][1][bind] / res['num_2m'][1][bind]
        R22 = (g2p - g2m) / 2 / 0.01
        
        return g1/R11, g2/R22

    import smatch
    import pickle 
    import time 
    gold = fio.read(gold_f)

    nside = 4096
    maxmatch = 1
    radius = 0.263/3600 # degrees

    with open('/global/cscratch1/sd/myamamot/metadetect/mdet_bin_psfe1.pickle', 'rb') as f:
        psf1bin = pickle.load(f)
    with open('/global/cscratch1/sd/myamamot/metadetect/mdet_bin_psfe2.pickle', 'rb') as f2:
        psf2bin = pickle.load(f2)

    # f_response = open('/global/cscratch1/sd/myamamot/metadetect/shear_response_v2.txt', 'r')
    # R11, R22 = f_response.read().split('\n')

    tomobin = {'bin1': [0,0.36], 'bin2': [0.36,0.63], 'bin3': [0.63,0.87], 'bin4': [0.87,2.0], 'all': [0.0,2.0]}
    tomobin_color = {'gi_color': {
                    'bin1': {'mag': 0.0, 'num': 0.0}, 
                    'bin2': {'mag': 0.0, 'num': 0.0}, 
                    'bin3': {'mag': 0.0, 'num': 0.0}, 
                    'bin4': {'mag': 0.0, 'num': 0.0}, 
                    'all': {'mag': 0.0, 'num': 0.0}
                    },}
    tomobin_shear = {'raw_sum': {
                     'bin1': {'noshear': np.zeros((2,15)), '1p': np.zeros((2,15)),  '1m': np.zeros((2,15)),  '2p': np.zeros((2,15)),  '2m': np.zeros((2,15)),  'num_noshear': np.zeros((2,15)), 'num_1p': np.zeros((2,15)), 'num_1m': np.zeros((2,15)), 'num_2p': np.zeros((2,15)), 'num_2m': np.zeros((2,15))}, 
                     'bin2': {'noshear': np.zeros((2,15)), '1p': np.zeros((2,15)),  '1m': np.zeros((2,15)),  '2p': np.zeros((2,15)),  '2m': np.zeros((2,15)),  'num_noshear': np.zeros((2,15)), 'num_1p': np.zeros((2,15)), 'num_1m': np.zeros((2,15)), 'num_2p': np.zeros((2,15)), 'num_2m': np.zeros((2,15))}, 
                     'bin3': {'noshear': np.zeros((2,15)), '1p': np.zeros((2,15)),  '1m': np.zeros((2,15)),  '2p': np.zeros((2,15)),  '2m': np.zeros((2,15)),  'num_noshear': np.zeros((2,15)), 'num_1p': np.zeros((2,15)), 'num_1m': np.zeros((2,15)), 'num_2p': np.zeros((2,15)), 'num_2m': np.zeros((2,15))}, 
                     'bin4': {'noshear': np.zeros((2,15)), '1p': np.zeros((2,15)),  '1m': np.zeros((2,15)),  '2p': np.zeros((2,15)),  '2m': np.zeros((2,15)),  'num_noshear': np.zeros((2,15)), 'num_1p': np.zeros((2,15)), 'num_1m': np.zeros((2,15)), 'num_2p': np.zeros((2,15)), 'num_2m': np.zeros((2,15))}, 
                     'all': {'noshear': np.zeros((2,15)), '1p': np.zeros((2,15)),  '1m': np.zeros((2,15)),  '2p': np.zeros((2,15)),  '2m': np.zeros((2,15)),  'num_noshear': np.zeros((2,15)), 'num_1p': np.zeros((2,15)), 'num_1m': np.zeros((2,15)), 'num_2p': np.zeros((2,15)), 'num_2m': np.zeros((2,15))}, 
                    },
                    'mean_tile':  {
                    'bin1': {'g1': np.zeros((15, len(fs))), 'g2': np.zeros((15, len(fs)))}, 
                    'bin2': {'g1': np.zeros((15, len(fs))), 'g2': np.zeros((15, len(fs)))}, 
                    'bin3': {'g1': np.zeros((15, len(fs))), 'g2': np.zeros((15, len(fs)))}, 
                    'bin4': {'g1': np.zeros((15, len(fs))), 'g2': np.zeros((15, len(fs)))}, 
                    'all': {'g1': np.zeros((15, len(fs))), 'g2': np.zeros((15, len(fs)))}
                    }, 
                    }
    for i, fname in tqdm(enumerate(fs)):
        fp = os.path.join(work_mdet_cuts, fname)
        if os.path.exists(fp):
            d = fio.read(fp)
        else:
            continue

        gold_msked = gold[((gold['RA'] > np.min(d['ra'])) & (gold['RA'] < np.max(d['ra'])) & (gold['DEC'] > np.min(d['dec'])) & (gold['DEC'] < np.max(d['dec'])))]
        matches = smatch.match(d['ra'], d['dec'], radius, gold_msked['RA'], gold_msked['DEC'], nside=nside, maxmatch=maxmatch)
        zs = gold_msked[matches['i2']]['DNF_Z']
        d_match = d[matches['i1']]
        for b in ['bin1', 'bin2', 'bin3', 'bin4', 'all']:
            msk_bin = ((zs > tomobin[b][0]) & (zs < tomobin[b][1]))
            psfe1 = d_match[msk_bin]['psfrec_g_1']
            psfe2 = d_match[msk_bin]['psfrec_g_2']
            d_bin = d_match[msk_bin]

            # save magnitude here.
            gi_color = flux2mag(d_bin['mdet_g_flux']) - flux2mag(d_bin['mdet_i_flux'])
            tomobin_color['gi_color'][b]['mag'] += np.sum(gi_color)
            tomobin_color['gi_color'][b]['num'] += len(gi_color)
            
            for j, pbin in enumerate(zip(psf1bin['low'], psf1bin['high'])):
                msk_psf = ((psfe1 > pbin[0]) & (psfe1 < pbin[1]))
                d_psfbin = d_bin[msk_psf]
                for step in ['noshear', '1p', '1m', '2p', '2m']:
                    msk_step = (d_psfbin['mdet_step'] == step)
                    np.add.at(tomobin_shear['raw_sum'][b][step], (0, j), np.sum(d_psfbin[msk_step]['mdet_g_1']))
                    np.add.at(tomobin_shear['raw_sum'][b][step], (1, j), np.sum(d_psfbin[msk_step]['mdet_g_2']))
                    np.add.at(tomobin_shear['raw_sum'][b]['num_'+step], (0, j), len(d_psfbin[msk_step]['mdet_g_1']))
                    np.add.at(tomobin_shear['raw_sum'][b]['num_'+step], (1, j), len(d_psfbin[msk_step]['mdet_g_2']))
                g1, g2 = _compute_g1g2(tomobin_shear['raw_sum'][b], j)
                tomobin_shear['mean_tile'][b]['g1'][j, i] = g1
                tomobin_shear['mean_tile'][b]['g2'][j, i] = g2
                
            for j, pbin in enumerate(zip(psf2bin['low'], psf2bin['high'])):
                msk_psf = ((psfe2 > pbin[0]) & (psfe2 < pbin[1]))
                d_psfbin = d_bin[msk_psf]
                for step in ['noshear', '1p', '1m', '2p', '2m']:
                    msk_step = (d_psfbin['mdet_step'] == step)
                    np.add.at(tomobin_shear['raw_sum'][b][step], (0, j), np.sum(d_psfbin[msk_step]['mdet_g_1']))
                    np.add.at(tomobin_shear['raw_sum'][b][step], (1, j), np.sum(d_psfbin[msk_step]['mdet_g_2']))
                    np.add.at(tomobin_shear['raw_sum'][b]['num_'+step], (0, j), len(d_psfbin[msk_step]['mdet_g_1']))
                    np.add.at(tomobin_shear['raw_sum'][b]['num_'+step], (1, j), len(d_psfbin[msk_step]['mdet_g_2']))
                g1, g2 = _compute_g1g2(tomobin_shear['raw_sum'][b], j)
                tomobin_shear['mean_tile'][b]['g1'][j, i] = g1
                tomobin_shear['mean_tile'][b]['g2'][j, i] = g2

    with open('/global/cscratch1/sd/myamamot/metadetect/mean_shear_tomobin_binresponse_e1e2.pickle', 'wb') as ft:
        pickle.dump(tomobin_shear, ft, protocol=pickle.HIGHEST_PROTOCOL)

    for b in ['bin1', 'bin2', 'bin3', 'bin4', 'all']:
        mean_gi_color = tomobin_color['gi_color'][b]['mag'] / tomobin_color['gi_color'][b]['num']
        print(mean_gi_color)

def main(argv):

    # gold_f = '/global/project/projectdirs/des/myamamot/y6_gold_dnf_z.fits'
    f = open('/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5b/tiles_blinded/mdet_files.txt', 'r')
    fs = f.read().split('\n')[:-1]
    
    mdet_response_filepath = sys.argv[1]
    mdet_input_filepath = sys.argv[2]
    piff_input = sys.argv[3]
    mdet_mom = sys.argv[4]
    out_path = sys.argv[5]
    random_point_map = sys.argv[6]
    mdet_cuts = int(sys.argv[7])
    test_name = sys.argv[8]
    weight_name = sys.argv[9]
    weight_path = sys.argv[10]
    var_method = sys.argv[11]
    star_weight_map = sys.argv[12]
    parallel = eval(sys.argv[13])
    sims = eval(sys.argv[14])
    color_split = eval(sys.argv[15])
    
    if test_name == 'nk':
        stellar_location_contamination(mdet_response_filepath, mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts)
    elif test_name == 'ng_stars':
        shear_stellar_contamination(mdet_response_filepath, mdet_input_filepath, piff_input, mdet_mom, out_path, random_point_map, mdet_cuts, weight_name, star_weight_map, simulations=sims, parallel=parallel)
    elif test_name == 'ng_stars_h5':
        shear_stellar_contamination_hdf5(mdet_input_filepath, piff_input, mdet_mom, out_path, random_point_map, mdet_cuts, weight_name, weight_path, star_weight_map, var_method, parallel=parallel)
    elif test_name == 'ng_fields':
        tangential_shear_field_center(fs, mdet_response_filepath, mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts, weight_name, simulations=sims, parallel=parallel)
    elif test_name == 'ng_fields_h5':
        _measure_tangential_shear_hdf5(mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts, weight_name, weight_path, var_method, color_split, parallel=parallel)
    # mean_shear_tomoz(gold_f, fs)

if __name__ == "__main__":
    main(sys.argv)