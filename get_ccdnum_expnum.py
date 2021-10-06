
import fitsio as fio
import numpy as np
import os
from tqdm import tqdm
import pickle

PATH = "/data/des70.a/data/masaya/"
f = open('/home/s1/masaya/des-y6-analysis/tiles.txt', 'r')
tilenames = f.read().split('\n')[:-1]
bands = ['r', 'i', 'z']

for band in bands:
    work = os.path.join(PATH, 'piff_models/'+band+'_band')
    band_expnum_ccdnum = {}
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

        # from pizza coadd file, get ccd number and exposure number for this tile and get PSF model. 
        mdet_cat = fio.read(os.path.join(PATH, 'metadetect/'+tilename+'_metadetect-v3_mdetcat_part0000.fits'))
        try:
            coadd = fio.FITS(os.path.join(PATH, 'pizza-slice/'+band+'_band/'+tilename+'_r5227p01_'+band+'_pizza-cutter-slices.fits.fz'))
        except:
            coadd = fio.FITS(os.path.join(PATH, 'pizza-slice/'+band+'_band/'+tilename+'_r5227p03_'+band+'_pizza-cutter-slices.fits.fz'))
        r_epochs = coadd['epochs_info'].read()
        r_image_info = coadd['image_info'].read()

        ccdexpnum = []
        se_data = []
        slice = mdet_cat['slice_id']
        for obj in range(len(mdet_cat)):
            slice_id = slice[obj]
            single_epochs = r_epochs[r_epochs['id']==slice_id]
            file_id = single_epochs[single_epochs['flags']==0]['file_id']

            object_ccd_exp = np.zeros(len(file_id), dtype=[('slice_id', int), ('ccdnum', int), ('expnum', int)])
            for start, f in enumerate(file_id):
                cn = int(r_image_info['image_path'][f][-28:-26])
                en = int(r_image_info['image_path'][f][5:13].lstrip("0")) # change indexing for different bands. 
                ccdexpnum.append((cn, en))

                object_ccd_exp['slice_id'][start] = slice_id
                object_ccd_exp['ccdnum'][start] = cn
                object_ccd_exp['expnum'][start] = en

            se_data.append(object_ccd_exp)

        se_data = np.concatenate(se_data, axis=0)
        band_expnum_ccdnum[tilename] = se_data
    with open('/data/des70.a/data/masaya/metadetect/mdet_'+band+'_ccdnum_expnum_v1.pickle', 'wb') as handle:
        pickle.dump(band_expnum_ccdnum, handle, protocol=pickle.HIGHEST_PROTOCOL) 