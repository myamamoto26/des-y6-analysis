
## Test DESDM query ##
import os
import healpy as hp
import sys
import fitsio as fio
import numpy as np
from tqdm import tqdm

def get_coaddtile_geom(section, query, out_fname):
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
    conn = ea.connect(section=section)
    curs = conn.cursor()
    conn.query_and_save(query, out_fname)

def main(argv):

    if sys.argv[1] == 'mdet':
        query = """
        select 
            *
        from 
            GRUENDL.METADETECT_OBJECT 
        where 
            flags = 0 
            and mdet_s2n > 10 
            and mfrac < 0.1 
            and mdet_t_ratio > 1.2 
            and mask_flags = 0;
        """
        out_fname = '/data/des70.a/data/masaya/metadetect/mdet_test_all_v2.fits'
        get_coaddtile_geom('desoper', query, out_fname)

    elif sys.argv[1] == 'basic_piff_model':
        query = """
        select
          distinct
          qa.ccdnum,
          qa.expnum,
          qa.filename, 
          qa.fwhm_cen, 
          qa.star_t_std, 
          qa.star_t_mean, 
          qa.nstar, 
          qa.exp_star_t_mean, 
          qa.exp_star_t_std
        from
          PIFF_MODEL_QA qa,
          proctag t,
          miscfile m
        where
          t.tag = 'Y6A2_PIFF_TEST_V2'
          and t.pfw_attempt_id = m.pfw_attempt_id
          and m.filetype = 'piff_model'
          and m.filename = qa.filename
          and qa.flag = 0;
        """

        out_fname = '/data/des70.a/data/masaya/piff_models/basic_piffs_query_test_v2.fits'
        get_coaddtile_geom('desoper', query, out_fname)

    elif sys.argv[1] == 'good_piff_model':
        query_good_piff_model = """
        select
          distinct
          qa.ccdnum,
          qa.expnum,
          qa.filename, 
          qa.fwhm_cen, 
          qa.star_t_std, 
          qa.star_t_mean, 
          qa.nstar, 
          qa.exp_star_t_mean, 
          qa.exp_star_t_std
        from
          PIFF_MODEL_QA qa,
          proctag t,
          miscfile m
        where
          t.tag = 'Y6A1_PIFF'
          and t.pfw_attempt_id = m.pfw_attempt_id
          and m.filetype = 'piff_model'
          and m.filename = qa.filename
          and qa.flag = 0
          and qa.fwhm_cen < 3.6
          and qa.star_t_std < 0.3 * qa.star_t_mean
          and qa.nstar >= 25
          and abs(qa.star_t_mean - qa.exp_star_t_mean) < 4. * qa.exp_star_t_std;
        """

        out_fname = '/data/des70.a/data/masaya/piff_models/good_piffs_newcuts_query_v1.fits'
        get_coaddtile_geom('desoper', query_good_piff_model, out_fname)
    
    elif sys.argv[1] == 'gold':
        # nside = 32
        # npix = hp.nside2npix(nside)
        test_regions = [np.array([17, 33, -41, -29]), np.array([28, 42, -11, 1]), np.array([308, 325, -58, -42]), np.array([63, 80, -55, -41])]
        for i, radec in enumerate(test_regions):
            query_gold = """
            select 
              RA, 
              DEC, 
              HPIX_32, 
              BDF_MAG_G,
              BDF_MAG_R, 
              BDF_MAG_I, 
              BDF_MAG_Z
            from 
              y6_gold_2_0
            where 
              RA >= %s
              and RA < %s
              and DEC >= %s
              and DEC < %s
            """ % (radec[0], radec[1], radec[2], radec[3])
            out_fname = '/data/des70.a/data/masaya/gold/gold_2_0_magnitudes_%s.fits' % i
            get_coaddtile_geom('dessci', query_gold, out_fname)

    elif sys.argv[1] == 'field_centers':

        f = open('/global/cscratch1/sd/myamamot/pizza-slice/ccd_exp_num.txt', 'r')
        en = f.read().split('\n')[:-1]

        split_en = np.array_split(en, int(sys.argv[2]))
        fc_all = ()
        for ii, exp in tqdm(enumerate(split_en)):
            query = """
            select 
                i.expnum,
                count(*) as cnt,
                avg(i.ra_cent) as ra_cent,
                avg(i.dec_cent) as dec_cent 
            from 
                IMAGE i, proctag t
            where 
                t.tag='Y6A1_COADD_INPUT'
                and t.pfw_attempt_id=i.pfw_attempt_id
                and i.filetype='red_immask'
                and i.expnum in %s
            group by i.expnum;
            """ % (tuple(exp), )
            ################################################################
            ## ERROR HAPPENING HERE.                                      ##
            ## ORA-01795: maximum number of expressions in a list is 1000 ##
            ################################################################
            out_fname = '/global/cscratch1/sd/myamamot/pizza-slice/exposure_field_centers_'+str(ii)+'.fits'
            # get_coaddtile_geom('desoper', query, out_fname)

            fc = fio.read(out_fname)
            fc_all.append(fc)

        fc_all = np.concatenate(fc)
        fio.write('/global/cscratch1/sd/myamamot/pizza-slice/exposure_field_centers.fits', fc_all)


    elif sys.argv[1] == 'None':
        good_piffs = fio.read('/data/des70.a/data/masaya/piff_models/good_piffs_newcuts_query_test_v2.fits')
        band_extract = np.array([fname[10:11] for fname in good_piffs['FILENAME']])

        for band in ['r', 'i', 'z']:
            mask_band = (band_extract == band)
            fnames = good_piffs[mask_band]['FILENAME']
            fnames_split = np.array_split(fnames, np.ceil(len(fnames)/1000))
            print(len(fnames_split))
            for i, f in enumerate(fnames_split):
                query = """
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
                    sit.model_flag = 0
                    and sit.star_flag = 0
                    and sit.filename in %s;
                """ % (tuple(f),)
                out_fname = '/data/des70.a/data/masaya/piff_models/newcuts_test_v2/'+band+'_'+str(i)+'_piff_models_test_v2.fits'
                get_coaddtile_geom('desoper', query, out_fname)
            
            master = []
            for i in range(len(fnames_split)):
                master.append(fio.read('/data/des70.a/data/masaya/piff_models/newcuts_test_v2/'+band+'_'+str(i)+'_piff_models_test_v2.fits'))
            master_data = np.concatenate(master, axis=0)
            fio.write('/data/des70.a/data/masaya/piff_models/'+band+'_band/master_'+band+'_piff_models_newcuts_test_v2.fits', master_data)


if __name__ == "__main__":
    main(sys.argv)
