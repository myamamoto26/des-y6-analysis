import fitsio as fio
import numpy as np 
from des_y6utils import mdet
import glob
from tqdm import tqdm
import healsparse
import os

def compute_asinh_flux(mag, band):

    zp = 30.0
    b_array = np.array([3.27e-12, 4.83e-12, 6.0e-12, 9.0e-12])
    bscale = np.array(b_array) * 10.**(zp / 2.5)

    A = (2.5 * np.log10(1.0 / b_array[band]) - mag) * 0.4 * np.log(10)
    flux = 2*bscale[band] * np.sinh(A)

    return flux


# def selections_tiles():
mdet_fs = sorted(glob.glob('/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5b/jackknife_patches_blinded/patch-*.fits'))
hmap = healsparse.HealSparseMap.read('/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hleda-gaiafull-des-stars-hsmap16384-nomdet-v3.fits')
dustmap = healsparse.HealSparseMap.read('/global/cfs/cdirs/des/y6-shear-catalogs/SFD_dust_4096.hsp')

l = []
n_terr = 0
for f in tqdm(mdet_fs):
    patch = f.split('/')[-1]
    if os.path.exists('/global/cfs/cdirs/des/myamamot/metadetect/Y6A2_METADETECT_V5b/'+patch):
        continue
    
    d = fio.read(f)

    msk = ((d["gauss_s2n"] > 10)
        & (d["mfrac"] < 0.1)
        # & (d["mdet_step"] == 'noshear')
        & (d["pgauss_T"] < (1.2 - 3.1*d["pgauss_T_err"]))
        & (
            d["gauss_T_ratio"] >= np.maximum(
                0.5,
                (n_terr*d["gauss_T_err"]/d["gauss_psf_T"]))
        ))
    
    size_sizeerr = (d['gauss_T_ratio']*d['gauss_psf_T']) * d['gauss_T_err']
    size_s2n = (d['gauss_T_ratio']*d['gauss_psf_T']) / d['gauss_T_err']
    msk_superspreader = ((size_sizeerr > 1) & (size_s2n < 10))
    msk &= ~msk_superspreader
    
    in_footprint = hmap.get_values_pos(d["ra"], d["dec"], valid_mask=True)
    msk &= in_footprint
    
    d = d[msk]

    # extinction
    dered = dustmap.get_values_pos(d["ra"], d["dec"])
    mag_g = mdet._compute_asinh_mags(d["pgauss_band_flux_g"], 0)
    dered_mag_g = mag_g - 3.186*dered
    mag_r = mdet._compute_asinh_mags(d["pgauss_band_flux_r"], 1)
    dered_mag_r = mag_r - 2.140*dered
    mag_i = mdet._compute_asinh_mags(d["pgauss_band_flux_i"], 2)
    dered_mag_i = mag_i - 1.196*dered
    mag_z = mdet._compute_asinh_mags(d["pgauss_band_flux_z"], 3)
    dered_mag_z = mag_z - 1.048*dered

    # magnitude cut
    gmr = dered_mag_g - dered_mag_r
    rmi = dered_mag_r - dered_mag_i
    imz = dered_mag_i - dered_mag_z
    mag_msk = ((np.abs(gmr) < 5)
                & (np.abs(rmi) < 5)
                & (np.abs(imz) < 5)
                & np.isfinite(dered_mag_g)
                & np.isfinite(dered_mag_r)
                & np.isfinite(dered_mag_i)
                & np.isfinite(dered_mag_z)
                & (dered_mag_g < 26.5)
                & (dered_mag_r < 26.5)
                & (dered_mag_i < 26.2)
                & (dered_mag_z < 25.6))
    d = d[mag_msk]

    res = np.zeros(len(d), dtype=[('uid', 'i8'), ('ra', 'f8'), ('dec', 'f8'), ('gauss_T_ratio', 'f8'), ('gauss_s2n', 'f8'), ('gauss_g_1', 'f8'), ('gauss_g_2', 'f8'), ('gauss_g_cov_1_1', 'f8'), ('gauss_g_cov_2_2', 'f8'), ('ebv', 'f8'), ('pgauss_band_flux_g', 'f8'), ('pgauss_band_flux_r', 'f8'), ('pgauss_band_flux_i', 'f8'), ('pgauss_band_flux_z', 'f8'), ('nodered_pgauss_band_flux_g', 'f8'), ('nodered_pgauss_band_flux_r', 'f8'), ('nodered_pgauss_band_flux_i', 'f8'), ('nodered_pgauss_band_flux_z', 'f8'), ('pgauss_band_flux_err_g', 'f8'), ('pgauss_band_flux_err_r', 'f8'), ('pgauss_band_flux_err_i', 'f8'), ('pgauss_band_flux_err_z', 'f8'), ('mdet_step', object)])
    res['uid'] = d['uid']
    res['ra'] = d['ra']
    res['dec'] = d['dec']
    res['gauss_T_ratio'] = d['gauss_T_ratio']
    res['gauss_s2n'] = d['gauss_s2n']
    res['gauss_g_1'] = d['gauss_g_1']
    res['gauss_g_2'] = d['gauss_g_2']
    res['gauss_g_cov_1_1'] = d['gauss_g_cov_1_1']
    res['gauss_g_cov_2_2'] = d['gauss_g_cov_2_2']
    res['nodered_pgauss_band_flux_g'] = d['pgauss_band_flux_g']
    res['nodered_pgauss_band_flux_r'] = d['pgauss_band_flux_r']
    res['nodered_pgauss_band_flux_i'] = d['pgauss_band_flux_i']
    res['nodered_pgauss_band_flux_z'] = d['pgauss_band_flux_z']
    res['pgauss_band_flux_err_g'] = d['pgauss_band_flux_err_g']
    res['pgauss_band_flux_err_r'] = d['pgauss_band_flux_err_r']
    res['pgauss_band_flux_err_i'] = d['pgauss_band_flux_err_i']
    res['pgauss_band_flux_err_z'] = d['pgauss_band_flux_err_z']
    res['mdet_step'] = d['mdet_step']
    res['ebv'] = dered[mag_msk]
    res['pgauss_band_flux_g'] = compute_asinh_flux(dered_mag_g[mag_msk], 0)
    res['pgauss_band_flux_r'] = compute_asinh_flux(dered_mag_r[mag_msk], 1)
    res['pgauss_band_flux_i'] = compute_asinh_flux(dered_mag_i[mag_msk], 2)
    res['pgauss_band_flux_z'] = compute_asinh_flux(dered_mag_z[mag_msk], 3)
    
    with fio.FITS('/global/cfs/cdirs/des/myamamot/metadetect/Y6A2_METADETECT_V5b/'+patch, 'rw') as fits:
        fits.write(res)


print('done with individual files')

# Make HDF5 file
import h5py as h5
# from mpi4py import MPI
# from mpi4py.util import pkl5
import pickle
# comm = pkl5.Intracomm(MPI.COMM_WORLD)
# rank = comm.Get_rank()
# size = comm.Get_size()


fs = sorted(glob.glob('/global/cfs/cdirs/des/myamamot/metadetect/Y6A2_METADETECT_V5b/patch-*.fits'))
mdet_steps = np.load('/global/cfs/cdirs/des/myamamot/metadetect/Y6A2_METADETECT_V5b/mdet_steps.npy')
keys = fio.read(fs[0]).dtype.names[:-1] # except mdet_step key

d = {}
dp = []
for patch in tqdm(fs):
    dp_ = fio.read(patch)
    dp.append(dp_)
dp = np.concatenate(dp)
for i in ['noshear', '1p', '1m', '2p', '2m']:
    msk = (mdet_steps == i)
    d[i] = dp[msk]
with open('/global/cfs/cdirs/des/myamamot/metadetect/Y6A2_METADETECT_V5b/patch-data.pickle', 'wb') as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
print('pickle dumped')

dered_h5 = h5.File('/global/cfs/cdirs/des/myamamot/metadetect/Y6A2_METADETECT_V5b/metadetect_desdmv5a_cutsv5_dereddened.h5', 'w') 
with open('/global/cfs/cdirs/des/myamamot/metadetect/Y6A2_METADETECT_V5b/patch-data.pickle', 'rb') as f:
    d = pickle.load(f)
keys = d['noshear'].dtype.names[:-1]
    
for j in ['noshear/','1p/','1m/','2p/','2m/']:
    for k in tqdm(keys):
        dered_h5.create_dataset('mdet/'+j+k, data = d[j[:-1]][k] )
dered_h5.close()