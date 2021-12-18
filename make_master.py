
import fitsio as fio
import numpy as np
import sys, os
from tqdm import tqdm
PATH = "/data/des70.a/data/masaya/"

# def make_master_cat(fname, catalog):

#     f = open('/home/s1/masaya/des-y6-analysis/tiles.txt', 'r')
#     tilenames = f.read().split('\n')[:-1]
#     band = 'r'

#     mdet = []
#     coadd = []
#     piff_model = []
#     piff_star = []
#     for tilename in tqdm(tilenames):
#         f_mdet = os.path.join(PATH, 'metadetect/'+tilename+'_metadetect-v3_mdetcat_part0000.fits')
#         f_coadd = os.path.join(PATH, 'pizza-slice/'+band+'_band/'+tilename+'_r5227p01_'+band+'_pizza-cutter-slices.fits.fz')
#         if not os.path.exists(f_coadd):
#             f_coadd = os.path.join(PATH, 'pizza-slice/'+band+'_band/'+tilename+'_r5227p03_'+band+'_pizza-cutter-slices.fits.fz')
#         if (not os.path.exists(f_mdet)) or (not os.path.exists(f_coadd)):
#             print('this tilename ', tilename, ' does not have either mdet cat or coadd info.')
#             continue
#         f_piff_model = os.path.join(PATH, 'piff_models/'+band+'_band/'+tilename+'_model.fits')
#         f_piff_star = os.path.join(PATH, 'piff_models/'+band+'_band/'+tilename+'_star.fits')

#         mdet.append(fio.read(f_mdet))
#         coadd.append(fio.read(f_coadd))
#         piff_model.append(fio.read(f_piff_model))
#         piff_star.append(fio.read(f_piff_star))

#     master_mdet = np.concatenate(mdet, axis=0)
#     master_coadd = np.concatenate(coadd, axis=0)
#     master_piff_model = np.concatenate(piff_model, axis=0)
#     master_piff_star = np.concatenate(piff_star, axis=0)

#     fio.write('metadetect_master_v1.fits', master_mdet)
#     fio.write('coadd_master_v1.fits', master_coadd)
#     fio.write('piff_model_master_v1.fits', master_piff_model)
#     fio.write('piff_star_master_v1.fits', master_piff_star)
    
def make_master_cat():

    out_fname = '/data/des70.a/data/masaya/metadetect/v3/mdet_test_all_v3.fits'
    f = open('/data/des70.a/data/masaya/metadetect/v3/fnames.txt', 'r')
    fs = f.read().split('\n')

    master = []
    for fname in fs:
        d = fname.split('/')[-1]
        mdet = fio.read(d)
        master.append(mdet)
    master_mdet = np.concatenate(master, axis=0)
    fio.write(out_fname, master_mdet)
    
