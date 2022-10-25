

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

def shear_stellar_contamination(mdet_response_filepath, mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts):

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

        var_method = 'jackknife', 
        output_dots = False,
    )

    cat1_file = fits.open('/project/projectdirs/des/schutt20/catalogs/y6a2_piff_v3_allres_v3_collated.fits')
    f_pc = '/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed9999.fits'
    d_piff=cat1_file[1].data
    msk = ((d_piff.field("BAND") == "r")) # | (d_piff.field("BAND") == "i") | (d_piff.field("BAND") == "z"))
    ra_piff = d_piff.field('RA')[msk]
    dec_piff = d_piff.field('DEC')[msk]
    flux_piff = d_piff.field('FLUX')[msk]
    
    mask_bright = (flux2mag(flux_piff) < 16.5)
    mask_faint = (flux2mag(flux_piff) > 16.5)
    cat1_bright = treecorr.Catalog(ra=ra_piff[mask_bright], dec=dec_piff[mask_bright], ra_units='deg', dec_units='deg', patch_centers=f_pc)
    cat1_faint = treecorr.Catalog(ra=ra_piff[mask_faint], dec=dec_piff[mask_faint], ra_units='deg', dec_units='deg', patch_centers=f_pc)

    cat2_files = glob.glob(mdet_input_filepath)
    cat2_bright = []
    cat2_faint = []
    for cat2_file in tqdm(cat2_files):
        d_mdet = fio.read(cat2_file)
        msk = mdet.make_mdet_cuts(d_mdet, mdet_cuts) 
        msk_noshear = (d_mdet['mdet_step']=='noshear')

        d_mdet = d_mdet[msk & msk_noshear]
        g1 = d_mdet[mdet_mom+'_g_1']/R
        g2 = d_mdet[mdet_mom+'_g_2']/R
        cat_bright = treecorr.Catalog(ra=d_mdet['ra'], dec=d_mdet['dec'], ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch_centers=f_pc)
        cat_faint = treecorr.Catalog(ra=d_mdet['ra'], dec=d_mdet['dec'], ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch_centers=f_pc)
        cat2_bright.append(cat_bright)
        cat2_faint.append(cat_faint)

    # random point subtraction. 
    cat1r_file = random_point_map
    cat1r = treecorr.Catalog(cat1r_file, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg', patch_centers=f_pc)
    cat2_both = [cat2_bright, cat2_faint]
    fname = ['bright', 'faint']
    for ii,cat1 in enumerate([cat1_bright, cat1_faint]):
        ng = treecorr.NGCorrelation(bin_config, verbose=2)
        ng_rand = treecorr.NGCorrelation(bin_config, verbose=2)
        for i,cat2 in tqdm(enumerate(cat2_both[ii])):
            ng.process(cat1, cat2, initialize=(i==0), finalize=(i==len(cat2_both[ii])-1))
            ng_rand.process(cat1r, cat2, initialize=(i==0), finalize=(i==len(cat2_both[ii])-1))
            cat2.unload()
        # np.save(os.path.join(out_path, mdet_mom+'_stars_shear_cross_correlation_cov_'+fname[ii]+'.npy'), ng.cov)
        # ng.write(os.path.join(out_path, mdet_mom+'_stars_shear_cross_correlation_output_'+fname[ii]+'.fits'))

        ng.write(os.path.join(out_path, mdet_mom+'_stars_shear_cross_correlation_final_output_'+fname[ii]+'.fits'), rg=ng_rand)

# Figure 14; Tangential shear around field center
def tangential_shear_field_center(fs, mdet_response_filepath, mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts):

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
    sys.path.append('./download-query-concatenation-code')
    from query_examples import query_field_centers, query_coadd_info
    import treecorr
    import glob
             
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
                    expnum = _get_exp_num(image_info['image_path'][msk_im][0])
                    exp_num.append(expnum)
        exp_num = np.unique(np.array(exp_num))
        total_exp_num = len(exp_num)
        print('total exposure number', total_exp_num)

        with open('/global/cscratch1/sd/myamamot/pizza-slice/ccd_exp_num.txt', 'w') as f:
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
    f_response = open(mdet_response_filepath, 'r')
    R11, R22 = f_response.read().split('\n')
    R = (np.float64(R11) + np.float64(R22))/2
    f_pc = '/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed9999.fits'

    # Create ccdnum and expnum text file if it has not been created yet, and query from DESDM table. Should only be done once. 
    if not os.path.exists('/global/cscratch1/sd/myamamot/pizza-slice/ccd_exp_num.txt'):
        find_exposure_numbers(fs)
        query_field_centers('/global/cscratch1/sd/myamamot/pizza-slice/ccd_exp_num.txt', 300)
        print('done making ccd num file')
    
    expnum_field_centers = fio.read('/global/cscratch1/sd/myamamot/pizza-slice/exposure_field_centers.fits')
    print('number of field centers', len(expnum_field_centers))

    bin_config = dict(
                sep_units = 'arcmin',
                bin_slop = 0.1,

                min_sep = 1.0,
                max_sep = 250,
                nbins = 20,

                var_method = 'jackknife',
                output_dots = False,
                )

    cat1_file = '/global/cscratch1/sd/myamamot/pizza-slice/exposure_field_centers.fits'
    cat1 = treecorr.Catalog(cat1_file, ra_col='RA_CENT', dec_col='DEC_CENT', ra_units='deg', dec_units='deg', patch_centers=f_pc)
    # random point subtraction. 
    cat1r_file = random_point_map
    cat1r = treecorr.Catalog(cat1r_file, ra_col='ra', dec_col='dec', ra_units='deg', dec_units='deg', patch_centers=f_pc)
    ng_rand = treecorr.NGCorrelation(bin_config, verbose=2)
    cat2_files = glob.glob(mdet_input_filepath)
    ng = treecorr.NGCorrelation(bin_config, verbose=2)
    for i,cat2_f in tqdm(enumerate(cat2_files)):
        d_mdet = fio.read(cat2_f)
        msk = mdet.make_mdet_cuts(d_mdet, mdet_cuts) 
        msk_noshear = (d_mdet['mdet_step']=='noshear')
        d_mdet = d_mdet[msk & msk_noshear]

        g1 = d_mdet[mdet_mom+'_g_1']/R
        g2 = d_mdet[mdet_mom+'_g_2']/R
        cat2 = treecorr.Catalog(ra=d_mdet['ra'], dec=d_mdet['dec'], ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch_centers=f_pc)
    
        ng.process(cat1, cat2, initialize=(i==0), finalize=(i==len(cat2_files)-1))
        ng_rand.process(cat1r, cat2, initialize=(i==0), finalize=(i==len(cat2_files)-1))
        cat2.unload()
    
    nn_rand = treecorr.NNCorrelation(bin_config, verbose=2)
    nn_rand.process(cat1r)
    ng_final = ng.calculateXi(nn_rand, dr=ng_rand)
    ng_cov = ng_final.cov
    
    np.save(os.path.join(out_path, mdet_mom+'_field_centers_cross_correlation_final_output_cov.npy'), ng_cov)
    ng.write(os.path.join(out_path, mdet_mom+'_field_centers_cross_correlation_final_output.fits'), rg=ng_rand)

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
    f = open('/global/cfs/cdirs/des/y6-shear-catalogs/metadetection/mdet_files.txt', 'r')
    fs = f.read().split('\n')[:-1]
    
    mdet_response_filepath = sys.argv[1]
    mdet_input_filepath = sys.argv[2]
    mdet_mom = sys.argv[3]
    out_path = sys.argv[4]
    random_point_map = sys.argv[5]
    mdet_cuts = int(sys.argv[6])
    
    # shear_stellar_contamination(mdet_response_filepath, mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts)
    tangential_shear_field_center(fs, mdet_response_filepath, mdet_input_filepath, mdet_mom, out_path, random_point_map, mdet_cuts)
    # mean_shear_tomoz(gold_f, fs)

if __name__ == "__main__":
    main(sys.argv)