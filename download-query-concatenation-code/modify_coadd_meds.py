
from email import header
import os,sys
import fitsio as fio
import numpy as np
import glob
from astropy.io import fits
from tqdm import tqdm
import subprocess

pizza_info = fio.read('/global/cfs/cdirs/des/myamamot/pizza-slice/pizza-cutter-coadds-info.fits')
band = ['z']
no_files = ["DES2357-5040_r5992p01_r_pizza-cutter-slices.fits", "DES0410-5831_r5934p01_r_pizza-cutter-slices.fits", "DES0105+0126_r5932p01_r_pizza-cutter-slices.fits", "DES0542-3832_r5982p01_r_pizza-cutter-slices.fits", "DES0236-5248_r5933p01_i_pizza-cutter-slices.fits", "DES2127-5123_r5992p01_i_pizza-cutter-slices.fits", "DES2302-6414_r5992p01_z_pizza-cutter-slices.fits"]
for path,fname in tqdm(zip(pizza_info['PATH'], pizza_info['FILENAME'])):
    filepath = os.path.join(path, fname+'.fz')

    if fname.split('_')[2] in band:
        cmd = """rsync -av --ignore-existing --password-file /global/cfs/cdirs/des/myamamot/pizza-slice/rsync_pass.txt masaya.yamamoto@desar2.cosmology.illinois.edu::ALLDESFiles/new_archive/desarchive/%s /global/cfs/cdirs/des/myamamot/pizza-slice/data/%s""" % (filepath, fname+'.fz')

        coadd = os.path.join('/global/cfs/cdirs/des/myamamot/pizza-slice/data', fname+'.fz')
        if fname in no_files :
            print(filepath)
            continue
        if not os.path.exists(coadd):
            try:
                subprocess.run(cmd, shell=True, check=True)
            except:
                print(filepath)

        if len(fits.open(coadd)) == 3:
            print('cannot rewrite', coadd)
            continue

        data1,hdr1 = fits.getdata(coadd, 'image_info', header=True)
        data2,hdr2 = fits.getdata(coadd, 'epochs_info', header=True)
        fits.writeto(coadd, data1, header=hdr1, overwrite=True)
        fits.append(coadd, data2, header=hdr2)