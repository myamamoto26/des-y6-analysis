
import fitsio as fio
import numpy as np
import os, sys
from tqdm import tqdm
from matplotlib import pyplot as plt
import emcee
import pickle
from rho_stats import measure_rho, measure_tao, write_stats, write_stats_tao, plot_overall_rho, plot_overall_tao

PATH = "/data/des70.a/data/masaya/"

def get_coaddtile_geom(query, out_fname):
    ## Query Gaia stars and PSF model from DESDM database. 
    """Get the coadd tile geom and return as dict.
    You can use the info returned here to query the unique tile area via
        if crossra0 == 'Y':
            uramin = uramin - 360.0
            msk = ra > 180.0
            ra[msk] -= 360
        in_coadd = (
            (ra > uramin)
            & (ra <= uramax)
            & (dec > udecmin)
            & (dec <= udecmax)
        )
    Parameters
    ----------
    tilename : str
        The name of the tile (e.g., "DES0146-3623").
    Returns
    -------
    ctg : dict
        A dictionary with key/values
            crossra0 : str
                Either Y or N.
            udecmin, udecmax, uramin, uramax : float
                The ra/dec ranges of the unique tile region.
    """
    os.system('export LD_LIBRARY_PATH=/home/s1/masaya/oml4rclient_install_dir/instantclient_21_1')
    import easyaccess as ea
    conn = ea.connect(section='desoper')
    curs = conn.cursor()
    # q = """select * from %s where """ % tablename
    # for i in query:
    #     q += '('+i+') or '
    # q = q[:-4]
    fnames = tuple(query)
    q = """
    select 
        sit.ccdnum, 
        sit.dec, 
        sit.expnum, 
        sit.flux, 
        sit.model_e1, 
        sit.model_e2, 
        sit.model_T, 
        sit.ra, 
        sit.snr, 
        sit.star_e1, 
        sit.star_e2, 
        sit.star_T, 
        sit.X, 
        sit.Y
    from 
        PIFF_STAR_QA sit 
    where 
        sit.filename in %s
    """ % (fnames,)
    conn.query_and_save(q, out_fname)
    # c = curs.fetchall()
    # if len(c) == 0:
    #     raise RuntimeError(
    #         "No coadd tile geom information can be found for tile '%s'!" % tilename
    #     )
    # if tablename == 'PIFF_MODEL_QA':
    #     ccdnum, chisq, dof, expnum, fname, flag, frac_cen_out, frac_width_out, fwhm_cen, model_e1_mean, model_e1_std, model_e2_mean, model_e2_std, model_nfit, model_T_mean, model_T_std, nremoved, nstar, star_e1_mean, star_e1_std, star_e2_mean, star_e2_std, star_nfit, star_T_mean, star_T_std = c[0]
    #     return dict(
    #         ccdnum = ccdnum, 
    #         chisq = chisq, 
    #         dof = dof, 
    #         expnum = expnum, 
    #         fname = fname, 
    #         flag = flag, 
    #         frac_cen_out = frac_cen_out, 
    #         frac_width_out = frac_width_out, 
    #         fwhm_cen = fwhm_cen, 
    #         model_e1_mean = model_e1_mean, 
    #         model_e1_std = model_e1_std, 
    #         model_e2_mean = model_e2_mean, 
    #         model_e2_std = model_e2_std, 
    #         model_nfit = model_nfit, 
    #         model_T_mean = model_T_mean, 
    #         model_T_std = model_T_std, 
    #         nremoved = nremoved, 
    #         nstar = nstar, 
    #         star_e1_mean = star_e1_mean, 
    #         star_e1_std = star_e1_std, 
    #         star_e2_mean = star_e2_mean, 
    #         star_e2_std = star_e2_std, 
    #         star_nfit = star_nfit, 
    #         star_T_mean = star_T_mean, 
    #         star_T_std = star_T_std
    #     )
    # elif tablename == 'PIFF_STAR_QA':
    #     ccdnum, dec, expnum, fname, flux, hpix_16384, hpix_64, hpix_65536, model_e1, model_e2, model_flag, model_T, ra, snr, star_e1, star_e2, star_flag, star_num, star_T, X, Y = c[0]
    #     return dict(
    #         ccdnum = ccdnum, 
    #         dec = dec, 
    #         expnum = expnum, 
    #         fname = fname, 
    #         flux = flux, 
    #         hpix_16384 = hpix_16384, 
    #         hpix_64 = hpix_64, 
    #         hpix_65536 = hpix_65536, 
    #         model_e1 = model_e1, 
    #         model_e2 = model_e2, 
    #         model_flag = model_flag, 
    #         model_T = model_T, 
    #         ra = ra, 
    #         snr = snr, 
    #         star_e1 = star_e1, 
    #         star_e2 = star_e2, 
    #         star_flag = star_flag, 
    #         star_num = star_num, 
    #         star_T = star_T, 
    #         X = X, 
    #         Y = Y
    #     )

def get_ccdnum_expnum(t, good_piffs, columns, band, read_expnum=True):

    # from pizza coadd file, get ccd number and exposure number for this tile and get PSF model. 
    mdet_cat = fio.read(os.path.join(PATH, 'metadetect/'+t+'_metadetect-v3_mdetcat_part0000.fits'))
    try:
        coadd = fio.FITS(os.path.join(PATH, 'pizza-slice/'+band+'_band/'+t+'_r5227p01_'+band+'_pizza-cutter-slices.fits.fz'))
    except:
        coadd = fio.FITS(os.path.join(PATH, 'pizza-slice/'+band+'_band/'+t+'_r5227p03_'+band+'_pizza-cutter-slices.fits.fz'))
    r_epochs = coadd['epochs_info'].read()
    r_image_info = coadd['image_info'].read()

    gal_data = []
    se_data = []
    ccdexpnum = []
    slice = mdet_cat['slice_id']
    for obj in range(len(mdet_cat)):
        # which pizza-slice does this object belong? 
        slice_id = slice[obj]
        # get single exposures for this pizza-slice. 
        single_epochs = r_epochs[r_epochs['id']==slice_id]
        file_id = single_epochs[single_epochs['flags']==0]['file_id']

        object = np.zeros(1, dtype=columns)
        object['ind'] = obj
        object['ra'] = mdet_cat['ra'][obj]
        object['dec'] = mdet_cat['dec'][obj]
        object['gal_e1'] = mdet_cat['mdet_g'][obj][0]
        object['gal_e2'] = mdet_cat['mdet_g'][obj][1]
        object['gal_T'] = mdet_cat['mdet_T'][obj]
        if not read_expnum:
            # accumulate CCD numbers and exposure numbers to query from database. 
            object_ccd_exp = np.zeros(len(file_id), dtype=[('slice_id', int), ('ccdnum', int), ('expnum', int)])
            for start, f in enumerate(file_id):
                cn = int(r_image_info['image_path'][f][-28:-26])
                en = int(r_image_info['image_path'][f][5:13].lstrip("0")) # change indexing for different bands. 
                ccdexpnum.append((cn, en))

                object_ccd_exp['slice_id'][start] = slice_id
                object_ccd_exp['ccdnum'][start] = cn
                object_ccd_exp['expnum'][start] = en
            se_data.append(object_ccd_exp)
        
        gal_data.append(object)

    gal_data = np.concatenate(gal_data, axis=0)
    if read_expnum:
        with open('/data/des70.a/data/masaya/metadetect/mdet_'+band+'_ccdnum_expnum_v1.pickle', 'rb') as handle:
            tile_expnum = pickle.load(handle)
            se_data = tile_expnum[t]
            ccdexpnum = [(se_data['ccdnum'][i], se_data['expnum'][i]) for i in range(len(se_data))]
    else:
        se_data = np.concatenate(se_data, axis=0)

    unique_ccdexpnum = list(set(ccdexpnum))
    query_fname = []
    for q in range(len(unique_ccdexpnum)):
        mask = ((good_piffs['CCDNUM']==unique_ccdexpnum[q][0]) & (good_piffs['EXPNUM']==unique_ccdexpnum[q][1]))
        fname = good_piffs[mask]['FILENAME']
        if len(fname) == 0:
            continue
        elif fname[0] not in query_fname:
            query_fname.append(fname[0])

    return query_fname, gal_data, se_data

def _make_cuts(d_model, d_star, default=True):

    if default:
        msk = ((d_star['STAR_FLAG'] == 0) & 
               (d_star['MODEL_FLAG'] == 0))    
    elif default == 'default2':
        msk_model = ((d_model['NSTAR'] - d_model['NREMOVED'] > 25) & 
                     (d_model['STAR_T_STD'] < 0.3*d_model['STAR_T_MEAN']) & 
                     (d_model['FWHM_CEN'] < 3.6) & 
                     (d_model['CHISQ'] < 1.5*d_model['DOF']))
        d_model_cut = d_model[msk_model]
        msk = ((d_star['STAR_FLAG'] == 0) & 
               (d_star['MODEL_FLAG'] == 0) & 
               (d_star['CCDNUM'] in d_model_cut['CCDNUM']) & 
               (d_star['EXPNUM'] in d_model_cut['EXPNUM']))
    return d_star[msk]

def find_piff_stars(piff_model_out, se_d):

    piff_star = fio.read(piff_model_out)

    model_stars_ = {}
    piff_star = piff_star[ np.argsort(piff_star,order=['EXPNUM','CCDNUM']) ]
    nstars = 0
    for exp in np.unique(se_d['expnum']):
        start = np.searchsorted(piff_star['EXPNUM'], exp)
        end = np.searchsorted(piff_star['EXPNUM'], exp+1)
        piff_star_ = piff_star[start:end]
        model_stars_[exp] = {}

        for ccd in np.unique(se_d['ccdnum']):
            start_ = np.searchsorted(piff_star_['CCDNUM'], ccd)
            end_ = np.searchsorted(piff_star_['CCDNUM'], ccd+1)
            piff_star__ = piff_star_[start_:end_]

            nstars += len(piff_star__['RA'])
            if len(piff_star__['RA'])==0:
                continue
            model_stars_[exp][ccd] = {'n':[], 'ra':[], 'dec':[], 'piff_e1':[], 'piff_e2':[], 'piff_T':[], 'obs_e1':[], 'obs_e2':[], 'obs_T':[], 'flux':[]}
            model_stars_[exp][ccd]['n'].append(len(piff_star__['RA']))
            model_stars_[exp][ccd]['piff_e1'].append(piff_star__['MODEL_E1'])
            model_stars_[exp][ccd]['piff_e2'].append(piff_star__['MODEL_E2'])
            model_stars_[exp][ccd]['piff_T'].append(piff_star__['MODEL_T'])
            model_stars_[exp][ccd]['obs_e1'].append(piff_star__['STAR_E1'])
            model_stars_[exp][ccd]['obs_e2'].append(piff_star__['STAR_E2'])
            model_stars_[exp][ccd]['obs_T'].append(piff_star__['STAR_T'])
            model_stars_[exp][ccd]['flux'].append(piff_star__['FLUX'])
            model_stars_[exp][ccd]['ra'].append(piff_star__['RA'])
            model_stars_[exp][ccd]['dec'].append(piff_star__['DEC'])

    return nstars, model_stars_

def make_stars_catalog(nstars, piff_stars_dict, columns):

    piff_stars_ = np.zeros(nstars, dtype=columns)
    start = 0
    for i,exp in enumerate(piff_stars_dict):
        for j,ccd in enumerate(piff_stars_dict[exp]):
            end = len(piff_stars_dict[exp][ccd]['ra'][0])
            piff_stars_['ra'][start:start+end] = piff_stars_dict[exp][ccd]['ra'][0]
            piff_stars_['dec'][start:start+end] = piff_stars_dict[exp][ccd]['dec'][0]
            piff_stars_['piff_e1'][start:start+end] = piff_stars_dict[exp][ccd]['piff_e1'][0]
            piff_stars_['piff_e2'][start:start+end] = piff_stars_dict[exp][ccd]['piff_e2'][0]
            piff_stars_['piff_T'][start:start+end] = piff_stars_dict[exp][ccd]['piff_T'][0]
            piff_stars_['obs_e1'][start:start+end] = piff_stars_dict[exp][ccd]['obs_e1'][0]
            piff_stars_['obs_e2'][start:start+end] = piff_stars_dict[exp][ccd]['obs_e2'][0]
            piff_stars_['obs_T'][start:start+end] = piff_stars_dict[exp][ccd]['obs_T'][0]
            piff_stars_['flux'][start:start+end] = piff_stars_dict[exp][ccd]['flux'][0]
            start += end
    return piff_stars_

# def run_emcee():



def main(argv):

    f = open('/home/s1/masaya/des-y6-analysis/tiles.txt', 'r')
    tilenames = f.read().split('\n')[:-1]
    bands = ['i', 'z']
    tao = True
    piff_stars = None
    gal_cat = None

    good_piffs_table = fio.read(os.path.join(PATH, 'piff_models/good_piffs_newcuts_query_v1.fits'))
    for band in bands:
        work = os.path.join(PATH, 'piff_models/'+band+'_band')
        if not os.path.exists(os.path.join(work, 'rho_all_newcuts_'+band+'.json')): 
            for tilename in tqdm(tilenames):
                f1 = os.path.join(PATH, 'metadetect/'+tilename+'_metadetect-v3_mdetcat_part0000.fits')
                f2 = os.path.join(PATH, 'pizza-slice/'+band+'_band/'+tilename+'_r5227p01_'+band+'_pizza-cutter-slices.fits.fz')
                if not os.path.exists(f2):
                    f2 = os.path.join(PATH, 'pizza-slice/'+band+'_band/'+tilename+'_r5227p03_'+band+'_pizza-cutter-slices.fits.fz')
                if (not os.path.exists(f1)) or (not os.path.exists(f2)):
                    print('this tilename ', tilename, ' does not have either mdet cat or coadd info.')
                    continue
                if tilename in ['DES0031+0001']:
                    continue

                print('Getting CCD number and exposure number for this coadd tile.')
                gal_d_columns = [('ind', int), ('ra', float), ('dec', float), ('gal_e1', float), ('gal_e2', float), ('gal_T', float)]
                query, gal_d, se_d = get_ccdnum_expnum(tilename, good_piffs_table, gal_d_columns, band)

                print('Query data...')
                piff_model_out = os.path.join(work, tilename+'_newcuts_piff_model.fits')
                if (not os.path.exists(piff_model_out)):
                    get_coaddtile_geom(query, piff_model_out)
                nstars, piff_stars_dict = find_piff_stars(piff_model_out, se_d)

                # need to create a new dict that contains, [ra, dec, obs_e1, obs_e2, obs_T, ccdnum, expnum, model_e1, model_e2, model_T etc...]
                gal_star_columns = [('ind', int), ('ccdnum', int), ('expnum', int), ('ra', float), ('dec', float), ('mag', float), 
                                    ('obs_e1', float), ('obs_e2', float), ('obs_T', float), ('flux', float), ('piff_e1', float), 
                                    ('piff_e2', float), ('piff_T', float)]

                if piff_stars is None:
                    piff_stars = make_stars_catalog(nstars, piff_stars_dict, gal_star_columns)
                    if tao and gal_cat is None:
                        gal_cat = gal_d
                else:
                    piff_stars_ = make_stars_catalog(nstars, piff_stars_dict, gal_star_columns)
                    piff_stars = np.concatenate([piff_stars, piff_stars_], axis=0)
                    if tao:
                        gal_cat = np.concatenate([gal_cat, gal_d], axis=0)

            print('Computing rho-stats...')
            max_sep = 250
            max_mag = 0
            name = 'all_newcuts' #'y3_cuts'
            tag = ''.join(band)
            stats = measure_rho(piff_stars, max_sep, max_mag, subtract_mean=True, do_rho0=True)
            stat_file = os.path.join(work, "rho_%s_%s.json"%(name,tag))
            write_stats(stat_file,*stats)
            plot_overall_rho(work, name)

            if tao:
                print('Computing tao-stats...')
                stats_tao = measure_tao(piff_stars, gal_cat, max_sep, max_mag, subtract_mean=True)
                stat_tao_file = os.path.join(work, "tao_%s_%s.json"%(name,tag))
                write_stats_tao(stat_tao_file,*stats_tao)
                plot_overall_tao(work, name)
        else:
            print('Computing rho-stats...')
            max_sep = 250
            max_mag = 0
            name = 'all_newcuts' #'y3_cuts'
            tag = ''.join(band)
            plot_overall_rho(work, name)

            if tao:
                print('Computing tao-stats...')
                plot_overall_tao(work, name)
    

if __name__ == "__main__":
    main(sys.argv)
