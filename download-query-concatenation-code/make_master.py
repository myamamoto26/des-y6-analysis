
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

# Combine PIFF tables.
def combine_piff(bands, work_piff, tilenames, all=False):
    
    if not all:
        for band in bands:
            model = []
            for t in tqdm(tilenames):
                if not os.path.exists(os.path.join(work_piff, band+'_band/'+t+'_piff_model.fits')):
                    continue
                model.append(fio.read(os.path.join(work_piff, band+'_band/'+t+'_piff_model.fits')))
            
            fio.write(os.path.join(work_piff, band+'_band/master_'+band+'_piff_models.fits'), np.concatenate(model, axis=0))
    else:
        r = fio.read(os.path.join(work_piff, 'r_band/master_r_piff_models.fits'))
        i = fio.read(os.path.join(work_piff, 'i_band/master_i_piff_models.fits'))
        z = fio.read(os.path.join(work_piff, 'z_band/master_z_piff_models.fits'))
        fio.write(os.path.join(work_piff, 'master_all_piff_models.fits'), np.concatenate([r, i, z], axis=0))

# combine gold catalogs split by healpix ids.
def combine_gold(test_region, work_gold):

    import healpy as hp

    gold_mag = []
    # npix = hp.nside2npix(nside)
    for i in range(test_region):
        for split in range(5):
            if os.path.exists(os.path.join(work_gold, 'gold_2_0_magnitudes_'+str(i)+'_'+str(split).zfill(6)+'.fits')):
                gold_mag.append(fio.read(os.path.join(work_gold, 'gold_2_0_magnitudes_'+str(i)+'_'+str(split).zfill(6)+'.fits')))
    gold_all = np.concatenate(gold_mag, axis=0)
    fio.write(os.path.join(work_gold, 'y6_gold_2_0_magnitudes.fits'), gold_all)
    
def main(argv):

    out_fname = '/data/des70.a/data/masaya/metadetect/v3/mdet_test_all_v3_'+sys.argv[2]+'.fits'
    f = open('/data/des70.a/data/masaya/metadetect/v3/fnames.txt', 'r')
    fs = f.read().split('\n')
    fs_split = np.array_split(fs, int(sys.argv[1]))

    master = []
    for fname in tqdm(fs_split[int(sys.argv[2])]):
        if fname == '':
            continue
        d = fname.split('/')[-1]
        mdet = fio.read(d)
        master.append(mdet)
    master_mdet = np.concatenate(master, axis=0)
    fio.write(out_fname, master_mdet)
    
if __name__ == "__main__":
    main(sys.argv)