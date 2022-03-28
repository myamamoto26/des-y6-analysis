
from email import header
import os,sys
import fitsio as fio
import numpy as np
import glob
from astropy.io import fits
from tqdm import tqdm

f=glob.glob('/global/project/projectdirs/des/myamamot/pizza-slice/OPS/multiepoch/Y6A2_PIZZACUTTER/**/*.fits.fz', recursive=True)
# f = ['/global/project/projectdirs/des/myamamot/pizza-slice/OPS/multiepoch/Y6A2_PIZZACUTTER/r5763/DES0004-3540/p01/pizza-cutter/DES0004-3540_r5763p01_i_pizza-cutter-slices.fits.fz', 
#     '/global/project/projectdirs/des/myamamot/pizza-slice/OPS/multiepoch/Y6A2_PIZZACUTTER/r5763/DES0022-5914/p01/pizza-cutter/DES0022-5914_r5763p01_g_pizza-cutter-slices.fits.fz', 
#     '/global/project/projectdirs/des/myamamot/pizza-slice/OPS/multiepoch/Y6A2_PIZZACUTTER/r5763/DES0022-5914/p01/pizza-cutter/DES0022-5914_r5763p01_r_pizza-cutter-slices.fits.fz']
for coadd in tqdm(f):
    data1,hdr1 = fits.getdata(coadd, 'image_info', header=True)
    data2,hdr2 = fits.getdata(coadd, 'epochs_info', header=True)
    fits.writeto(coadd, data1, header=hdr1, overwrite=True)
    fits.append(coadd, data2, header=hdr2)