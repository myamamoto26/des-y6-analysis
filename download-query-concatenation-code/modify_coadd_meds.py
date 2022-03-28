
from email import header
import os,sys
import fitsio as fio
import numpy as np
import glob
from astropy.io import fits

# f=glob.glob('/global/project/projectdirs/des/myamamot/pizza-slice/OPS/multiepoch/Y6A2_PIZZACUTTER/**/*.fits.fz', recursive=True)
f=glob.glob('/global/cscratch1/sd/myamamot/pizza-slice/test/*.fits.fz', recursive=True)
for coadd in f:
    data1,hdr1 = fits.getdata(coadd, 'image_info' header=True)
    data2,hdr2 = fits.getdata(coadd, 'epochs_info' header=True)
    fits.writeto(coadd, data1, header=hdr1)
    fits.append(coadd, data2, header=hdr2)