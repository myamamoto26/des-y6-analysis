import os, sys
from tqdm import tqdm
import numpy as np
import fitsio as fio
from astropy.io import fits
from des_y6utils import mdet
import glob
import treecorr
import json
import galsim

def _read_tanshear_signal(fname):
    """
    Reads in a file that contains radius from field center, 
    and tangential shear signal, and returns interpolated function.
    """
    from scipy import interpolate

    dat = fio.read(fname)
    gt = dat['gamT']; gx = dat['gamX']
    r = dat['meanr']
    ft = interpolate.Akima1DInterpolator(r, gt)
    fx = interpolate.Akima1DInterpolator(r, gx)
    
    return r[0], r[-1], ft, fx


def _compute_shear_fp(mint, maxt, ft, fx, x, y, g1, g2, msk):
    """
    Computes the shear signal of a given focal plane coordinates.
    """
    pix_scale = 0.26 # arcsec/pixel
    min_sep = 0.5
    max_sep = 250
    nbins = 30
    s = np.geomspace(min_sep, max_sep, nbins+1)

    dist = np.sqrt(x**2 + y**2) * pix_scale / 60.
    dist[(dist < mint)] = mint
    dist[(dist > maxt)] = maxt
    
    gt = ft(dist)
    gx = fx(dist)
    phi = 0.5 * np.arctan2(gx, gt)
    g = np.sqrt(gt**2 + gx**2)
    e1 = -g * np.cos(2*phi)
    e2 = g * np.sin(2*phi)

    g1[msk] = e1
    g2[msk] = e2

    return g1, g2

def _get_ccd_num(image_path):
        return int(image_path.split('/')[1].split('_')[2][1:])

def assign_shear(coadd_files, coord_file, d, funt, funx, mint, maxt):

    """
    Reads in coadd meds files for a tile, 
    gets focal plane coordinates of all objects in a tile, 
    assign shear to those objects. 
    """

    use_bands = ['r', 'i', 'z']
    g1_list = []; g2_list = []
    res = np.zeros(len(d['ra']), dtype=[('ra', 'f8'), ('dec', 'f8'), ('e1', 'f8'), ('e2', 'f8')])
    for pizza_f in coadd_files:
        if pizza_f.split('_')[2] in use_bands: 
            coadd = fio.FITS(os.path.join('/global/cfs/cdirs/des/myamamot/pizza-slice/data', pizza_f))
        else:
            continue
        try:
            epochs = coadd['epochs_info'].read()
            image_info = coadd['image_info'].read()
        except OSError:
            print('Corrupt file.?', pizza_f)
            raise OSError

        image_id = np.unique(epochs[(epochs['flags']==0)]['image_id'])
        image_id = image_id[image_id != 0]
        for iid in image_id:
            msk_im = np.where(image_info['image_id'] == iid)
            gs_wcs = galsim.FitsWCS(header=json.loads(image_info['wcs'][msk_im][0]))
            position_offset = image_info['position_offset'][msk_im][0]

            msk = ((epochs['flags'] == 0) & (epochs['image_id']==iid) & (epochs['weight'] > 0))
            if not np.any(msk):
                continue
            unique_slices = np.unique(epochs['id'][msk])

            msk_obj = np.where(np.in1d(d['slice_id'], unique_slices))[0]
            gamma1 = np.zeros(len(d['ra'])); gamma2 = np.zeros(len(d['ra']))
            if len(msk_obj) == 0:
                continue

            ccdnum = _get_ccd_num(image_info['image_path'][msk_im][0])
            crpix1 = coord_file[((coord_file['BAND'] == 'r') & (coord_file['CCDNUM'] == ccdnum))]['CRPIX1']
            crpix2 = coord_file[((coord_file['BAND'] == 'r') & (coord_file['CCDNUM'] == ccdnum))]['CRPIX2']
        
            n = len(msk_obj)
            ra_obj = d['ra'][msk_obj]
            dec_obj = d['dec'][msk_obj]

            pos_x, pos_y = gs_wcs.radecToxy(ra_obj, dec_obj, units="degrees")
            pos_x = pos_x - position_offset
            fpx = crpix1 + (pos_x - 2048//2)
            pos_y = pos_y - position_offset
            fpy = crpix2 + (pos_y - 4096//2)

            gamma1, gamma2 = _compute_shear_fp(mint, maxt, funt, funx, fpx, fpy, gamma1, gamma2, msk_obj)

            g1_list.append(gamma1)
            g2_list.append(gamma2)
        print(len(g1_list), np.where(np.in1d(d['slice_id'], unique_slices)))
    if ((len(g1_list)==0) or (len(g2_list)==0)):
        print(pizza_f)
    g1 = np.stack(g1_list); g2 = np.stack(g2_list)
    g1 = np.nanmean(np.where(g1!=0, g1, np.nan), axis=0)
    g2 = np.nanmean(np.where(g2!=0, g2, np.nan), axis=0)
    
    res['ra'] = d['ra']
    res['dec'] = d['dec']
    res['e1'] = g1
    res['e2'] = g2

    return res


def main(argv):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('mpi', rank, size)

    mdet_tilename_filepath = sys.argv[1]
    mdet_input_filepaths = sys.argv[2]
    coord_file = fio.read('/global/cfs/cdirs/des/myamamot/pizza-slice/focal-plane-coords-info.fits')
    pizza_coadd_info = '/global/cfs/cdirs/des/myamamot/pizza-slice/pizza-cutter-coadds-info.fits'
    shear_field_centers_file = sys.argv[3]
    outpath = sys.argv[4]
    make_cat = eval(sys.argv[5])

    if make_cat:
        mdet_f = open(mdet_tilename_filepath, 'r')
        mdet_fs = mdet_f.read().split('\n')[:-1]
        mdet_filenames = [fname.split('/')[-1] for fname in mdet_fs]
        tilenames = [d.split('_')[0] for d in mdet_filenames]
        # tilenames = ['DES0017-4206']
        print('num of tiles', len(tilenames))

        coadd_info = fio.read(pizza_coadd_info)
        coadd_files = {t: [] for t in tilenames}
        for coadd in coadd_info:
            tname = coadd['FILENAME'].split('_')[0]
            fname = coadd['FILENAME']+'.fz'
            if tname in list(coadd_files.keys()):
                coadd_files[tname].append(fname)

        mint, maxt, ft, fx = _read_tanshear_signal(shear_field_centers_file)
        cat = []
        split_tilenames = np.array_split(tilenames, size)
        for t in tqdm(split_tilenames[rank]):
            d = fio.read(os.path.join(mdet_input_filepaths, mdet_filenames[np.where(np.in1d(tilenames, t))[0][0]]))
            msk = mdet.make_mdet_cuts(d, 5)
            d = d[msk & (d['mdet_step'] == 'noshear')]

            res = assign_shear(coadd_files[t], coord_file, d, ft, fx, mint, maxt)
            cat.append(res)
        sys.exit()
        if rank != 0:
            res = comm.bcast(res, root=0)
        comm.Barrier()

        if rank == 0:
            for i in tqdm(range(1,size)):
                tmp_res = comm.recv(source=i)
                cat.append(tmp_res)
            cat = np.concatenate(cat)
            fio.write(outpath+'interpolated_shear_around_field_centers.fits', cat)

    # compute 2PCF with treecorr

if __name__ == "__main__":
    main(sys.argv)