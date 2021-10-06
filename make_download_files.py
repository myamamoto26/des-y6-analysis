import numpy as np

f = open('/home/s1/masaya/des-y6-analysis/tiles.txt', 'r')
tilen = f.read().split('\n')
mdet = False

all_files = []
all_r_files = []
all_i_files = []
all_z_files = []
for t in tilen:
	if mdet:
		fname_mdet = 'https://www.cosmo.bnl.gov/Private/gpfs/workarea/beckermr/des-y6-analysis/2021_05_08_mdet_process_y6a2_test1/outputs/' + t + '_metadetect-v3_mdetcat_part0000.fits'
		all_files.append(fname_mdet)
	else:
		fname_r_pizza = 'https://www.cosmo.bnl.gov/Private/gpfs/workarea/beckermr/DESDATA/ACT/multiepoch/Y6A2_PIZZACUTTER/r5227/'+t+'/p01/pizza-cutter/'+t+'_r5227p01_r_pizza-cutter-slices.fits.fz'
		fname_i_pizza = 'https://www.cosmo.bnl.gov/Private/gpfs/workarea/beckermr/DESDATA/ACT/multiepoch/Y6A2_PIZZACUTTER/r5227/'+t+'/p01/pizza-cutter/'+t+'_r5227p01_i_pizza-cutter-slices.fits.fz'
		fname_z_pizza = 'https://www.cosmo.bnl.gov/Private/gpfs/workarea/beckermr/DESDATA/ACT/multiepoch/Y6A2_PIZZACUTTER/r5227/'+t+'/p01/pizza-cutter/'+t+'_r5227p01_z_pizza-cutter-slices.fits.fz'
		all_r_files.append(fname_r_pizza)
		all_i_files.append(fname_i_pizza)
		all_z_files.append(fname_z_pizza)

if mdet:
	all_files = np.array(all_files)
	np.savetxt('mdet_cat_download.txt', all_files, fmt="%s")
else:
	all_r_files = np.array(all_r_files)
	all_i_files = np.array(all_i_files)
	all_z_files = np.array(all_z_files)
	np.savetxt('pizza_r_slice_download.txt', all_r_files, fmt="%s")
	np.savetxt('pizza_i_slice_download.txt', all_i_files, fmt="%s")
	np.savetxt('pizza_z_slice_download.txt', all_z_files, fmt="%s")