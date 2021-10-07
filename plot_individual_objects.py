
import fitsio as fio
import numpy as np
import os, sys
import meds

work = '/data/des70.a/data/masaya/'
tilenames = ['DES2208-5123', 'DES0234-0207', 'DES0326-2041']
mdet_obj = fio.read(os.path.join(work, 'metadetect/v2/mdet_test_all_v2.fits'))

coadd_fnames = fio.read(os.path.join('/data/des70.a/data/masaya/pizza-slice', 'pizza_slices_coadd_v2.fits'))
# coadd = fio.FITS(os.path.join('/data/des70.a/data/masaya/pizza-slice/v2/r_band/', pizza_f))
# r_epochs = coadd['epochs_info'].read()
# r_image_info = coadd['image_info'].read()

for t in tilenames:
    mdet = mdet_obj[mdet_obj['TILENAME']==t]
    coadd_f = coadd_fnames[((coadd_fnames['FILENAME'][:12]==t) & (coadd_fnames['BAND']=='r'))]['FILENAME']
    m = meds.MEDS(os.path.join('/data/des70.a/data/masaya/pizza-slice/v2/r_band/', coadd_f))
    for obj in mdet:
        if obj['MDET_T'] > 1.8:
            slice_id = obj['SLICE_ID']
            coadd_image = m.get_cutout(slice_id, 0)
            fio.write('/data/des70.a/data/masaya/pizza-slice/coadd_image/'+str(slice_id)+'_coadd.fits', coadd_image)