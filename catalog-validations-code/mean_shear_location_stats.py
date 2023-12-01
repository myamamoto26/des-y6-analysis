import fitsio as fio
import numpy as np
import pickle
import glob
from des_y6utils import mdet
from tqdm import tqdm
import os, sys
import galsim
import json

def _accum_shear(res, shear, mdet_step, xind, yind, g, b, fp=False):

    if fp:
        np.add.at(
                res[b][shear], 
                (yind, xind), 
                g,
            )
        np.add.at(
            res[b]["num_" + shear], 
            (yind, xind), 
            np.ones_like(g),
        )
    else:
        msk_s = (mdet_step == shear)
        if np.any(msk_s):
            # see https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html#numpy.ufunc.at
            np.add.at(
                res[b][shear], 
                (yind[msk_s], xind[msk_s]), 
                g[msk_s],
            )
            np.add.at(
                res[b]["num_" + shear], 
                (yind[msk_s], xind[msk_s]), 
                np.ones_like(g[msk_s]),
            )
    
def _accum_std(res_std, shear, mdet_step, bind, g, b, fp=False):

    uid = np.unique(bind)
    for bid in uid:
        if fp:
            msk_s = (bind == bid)
        else:
            msk_s = ((mdet_step == shear) & (bind == bid))
        if np.any(msk_s):
            res_std[b][bid].append(g[msk_s])

    return res_std

def get_stats_in_slice_coordinates():
    # coadd coordinates; we can use pre-defined PSF shape bins. 
    # 6 7 8
    # 3 4 5
    # 0 1 2
    xmin = 1; xmax= 200
    ymin = 1; ymax = 200
    num_bin = 3
    areaids = np.arange(0, num_bin**2)
    side = 200/3. 
    mdet_mom = 'wmom'
    xaxis = 'psfrec_g_1'
    mdet_files = glob.glob('/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V3/*_metadetect.fits')
    with open(os.path.join('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/v2', 'mean_shear_bin_final_v2_weighted.pickle'), 'rb') as handle:
        bin_dict = pickle.load(handle)[xaxis]
        binnum = len(bin_dict['hist'])
    # res =  {id: {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
    #         '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
    #         '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
    #         '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
    #         '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))} for id in areaids}
    res_e1 =  {b: {'noshear': np.zeros((num_bin, num_bin)), 'num_noshear': np.zeros((num_bin, num_bin)), 
                '1p': np.zeros((num_bin, num_bin)), 'num_1p': np.zeros((num_bin, num_bin)), 
                '1m': np.zeros((num_bin, num_bin)), 'num_1m': np.zeros((num_bin, num_bin)),
                '2p': np.zeros((num_bin, num_bin)), 'num_2p': np.zeros((num_bin, num_bin)),
                '2m': np.zeros((num_bin, num_bin)), 'num_2m': np.zeros((num_bin, num_bin))} for b in range(binnum)}
    res_e2 =  {b: {'noshear': np.zeros((num_bin, num_bin)), 'num_noshear': np.zeros((num_bin, num_bin)), 
                '1p': np.zeros((num_bin, num_bin)), 'num_1p': np.zeros((num_bin, num_bin)), 
                '1m': np.zeros((num_bin, num_bin)), 'num_1m': np.zeros((num_bin, num_bin)),
                '2p': np.zeros((num_bin, num_bin)), 'num_2p': np.zeros((num_bin, num_bin)),
                '2m': np.zeros((num_bin, num_bin)), 'num_2m': np.zeros((num_bin, num_bin))} for b in range(binnum)}
    e1_std = {b: {c: [] for c in range(num_bin**2)} for b in range(binnum)}
    e2_std = {b: {c: [] for c in range(num_bin**2)} for b in range(binnum)}
    for i,fname in tqdm(enumerate(mdet_files)):

        d = fio.read(fname)
        msk = mdet.make_mdet_cuts(d, 2)
        d = d[msk]
        mdet_step = d["mdet_step"]
        psf = d[xaxis]

        slice_x = d['slice_x']; slice_y = d['slice_y']
        xind = np.floor(slice_x/side).astype(int); yind = np.floor(slice_y/side).astype(int)
        bind = areaids.reshape(num_bin, num_bin)[yind, xind]

        for b,(low,high) in enumerate(zip(bin_dict['low'], bin_dict['high'])):
            msk_bin = ((psf > low) & (psf < high))
            _accum_shear(res_e1, "noshear", mdet_step[msk_bin], xind[msk_bin], yind[msk_bin], d[mdet_mom+"_g_1"][msk_bin], b)
            _accum_std(e1_std, "noshear", mdet_step[msk_bin], bind[msk_bin], d[mdet_mom+"_g_1"][msk_bin], b)
            _accum_shear(res_e2, "noshear", mdet_step[msk_bin], xind[msk_bin], yind[msk_bin], d[mdet_mom+"_g_2"][msk_bin], b)
            _accum_std(e2_std, "noshear", mdet_step[msk_bin], bind[msk_bin], d[mdet_mom+"_g_2"][msk_bin], b)
            _accum_shear(res_e1, "1p", mdet_step[msk_bin], xind[msk_bin], yind[msk_bin], d[mdet_mom+"_g_1"][msk_bin], b)
            _accum_shear(res_e1, "1m", mdet_step[msk_bin], xind[msk_bin], yind[msk_bin], d[mdet_mom+"_g_1"][msk_bin], b)
            _accum_shear(res_e2, "2p", mdet_step[msk_bin], xind[msk_bin], yind[msk_bin], d[mdet_mom+"_g_2"][msk_bin], b)
            _accum_shear(res_e2, "2m", mdet_step[msk_bin], xind[msk_bin], yind[msk_bin], d[mdet_mom+"_g_2"][msk_bin], b)

    for b in range(binnum):
        for c in range(num_bin**2):
            e1_std[b][c] = np.std(np.concatenate(e1_std[b][c]))
            e2_std[b][c] = np.std(np.concatenate(e2_std[b][c]))

    res = {'e1': res_e1, 'e2': res_e2, 'e1_std': e1_std, 'e2_std': e2_std}
    with open('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/v2/mean_shear_slice_coordinates.pickle', 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)



outpath = '/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/v2/mean_shear_measurement'
mdet_mom = 'wmom'
mdet_f = open('/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V3/tiles_blinded/mdet_files.txt', 'r')
mdet_fs = mdet_f.read().split('\n')[:-1]
mdet_filenames = [fname.split('/')[-1] for fname in mdet_fs]
tilenames = [d.split('_')[0] for d in mdet_filenames]

def find_objects_in_ccd(res, mdet_obj, coadd_files, mdet_mom, coord_file):
    def _get_ccd_num(image_path):
        return int(image_path.split('/')[1].split('_')[2][1:])

    for pizza_f in coadd_files:
        if pizza_f.split('_')[2] == 'r': # just do r-band for now. 
            coadd = fio.FITS(os.path.join('/global/cfs/cdirs/des/myamamot/pizza-slice/data', pizza_f))
        else:
            continue

        epochs = coadd['epochs_info'].read()
        image_info = coadd['image_info'].read()

        # For each single-epoch image to create the coadd tile (10,000x10,000), find the WCS that can convert sky coordinates in coadd slices that used the single-epoch image to the image coordinates in single-epoch image, find objects in the slices
        # image_info['image_id']: single-epoch image ID. 
        # epochs_info['image_id']: single-epoch image ID. 
        # epochs_info['id']: coadd slice ID (200x200). 
        # mdet['slice_id']: coadd slice ID where objects are detected (200x200). 

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

            msk_obj = np.where(np.in1d(mdet_obj['slice_id'], unique_slices))[0]
            if len(msk_obj) == 0:
                continue
            
            ccdnum = _get_ccd_num(image_info['image_path'][msk_im][0])
            crpix1 = coord_file[((coord_file['BAND'] == 'r') & (coord_file['CCDNUM'] == ccdnum))]['CRPIX1']
            crpix2 = coord_file[((coord_file['BAND'] == 'r') & (coord_file['CCDNUM'] == ccdnum))]['CRPIX2']
            for cname in ['noshear', '1p', '1m', '2p', '2m']:
                msk_ = (mdet_obj['mdet_step'][msk_obj] == cname)
                n = len(mdet_obj['mdet_step'][msk_obj][msk_])
                d_loc = np.zeros(n, dtype=[('ra', 'f8'), ('dec', 'f8'), ('ccd_x', 'f8'), ('ccd_y', 'f8'), ('e1', 'f8'), ('e2', 'f8'), ('x', 'f8'), ('y', 'f8'), ('psf_g1', 'f8'), ('psf_g2', 'f8')])
                ra_obj = mdet_obj['ra'][msk_obj][msk_]
                dec_obj = mdet_obj['dec'][msk_obj][msk_]

                pos_x, pos_y = gs_wcs.radecToxy(ra_obj, dec_obj, units="degrees")
                pos_x = pos_x - position_offset
                pos_y = pos_y - position_offset

                d_loc['ra'] = ra_obj
                d_loc['dec'] = dec_obj
                d_loc['x'] = crpix1 + (pos_x - 2048//2)
                d_loc['y'] = crpix2 + (pos_y - 4096//2)
                d_loc['ccd_x'] = pos_x
                d_loc['ccd_y'] = pos_y
                d_loc['e1'] = mdet_obj[mdet_mom+'_g_1'][msk_obj][msk_]
                d_loc['e2'] = mdet_obj[mdet_mom+'_g_2'][msk_obj][msk_]
                d_loc['psf_g1'] = mdet_obj['psfrec_g_1'][msk_obj][msk_]
                d_loc['psf_g2'] = mdet_obj['psfrec_g_2'][msk_obj][msk_]
                
                res[cname].append(d_loc)
    
    return res

def get_focal_plane_coordinates():
    # focal plane coordinates; let's save focal plane coordinates for each mdet object. 
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('mpi', rank, size)

    coord_info = fio.read('/global/cfs/cdirs/des/myamamot/pizza-slice/focal-plane-coords-info.fits')
    coadd_info = fio.read('/global/cfs/cdirs/des/myamamot/pizza-slice/pizza-cutter-coadds-info.fits')
    coadd_files = {t: [] for t in tilenames}
    for coadd in coadd_info:
        tname = coadd['FILENAME'].split('_')[0]
        fname = coadd['FILENAME']+'.fz'
        if tname in list(coadd_files.keys()):
            coadd_files[tname].append(fname)

    split_tilenames = np.array_split(tilenames, size)
    for t in tqdm(split_tilenames[rank]):
        res = {'noshear': [], '1p': [], '1m': [], '2p': [], '2m': []}
        if not os.path.exists(os.path.join(outpath, 'mdet_shear_focal_plane_'+t+'.pickle')):
            try:
                d = fio.read(os.path.join('/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V3/tiles_blinded', mdet_filenames[np.where(np.in1d(tilenames, t))[0][0]]))
                msk = mdet.make_mdet_cuts(d, 2)
                d = d[msk]
            except:
                print(t, " does not exist. skipping")
                continue
            # msk = if additional cuts are necessary. 
            res = find_objects_in_ccd(res, d, coadd_files[t], mdet_mom, coord_info)
            for cname in ['noshear', '1p', '1m', '2p', '2m']:
                if len(res[cname]) != 0:
                    res[cname] = np.concatenate(res[cname])
            with open(os.path.join(outpath, 'mdet_shear_focal_plane_'+t+'.pickle'), 'wb') as raw:
                pickle.dump(res, raw, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Already made this tile.', t)
    comm.Barrier()


fplane_files = glob.glob(os.path.join(outpath, 'mdet_shear_focal_plane_*.pickle'))
# 1. divide focal plane coordinates into different sections.
# xmin = -2048*6; xmax= 2048*6
# ymin = -4096*3; ymax = 4096*3
num_bin = 3
areaids = np.arange(0, num_bin**2)
side = 30000/3.
toff = 30000/2.
xaxis = 'psfrec_g_1'
# 2. For each section and PSF shape bin, accumulate shear (sum for response and raw for std).
with open(os.path.join('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/v2', 'mean_shear_bin_final_v2_weighted.pickle'), 'rb') as handle:
    bin_dict = pickle.load(handle)[xaxis]
    binnum = len(bin_dict['hist'])
res_e1 =  {b: {'noshear': np.zeros((num_bin, num_bin)), 'num_noshear': np.zeros((num_bin, num_bin)), 
                '1p': np.zeros((num_bin, num_bin)), 'num_1p': np.zeros((num_bin, num_bin)), 
                '1m': np.zeros((num_bin, num_bin)), 'num_1m': np.zeros((num_bin, num_bin)),
                '2p': np.zeros((num_bin, num_bin)), 'num_2p': np.zeros((num_bin, num_bin)),
                '2m': np.zeros((num_bin, num_bin)), 'num_2m': np.zeros((num_bin, num_bin))} for b in range(binnum)}
res_e2 =  {b: {'noshear': np.zeros((num_bin, num_bin)), 'num_noshear': np.zeros((num_bin, num_bin)), 
            '1p': np.zeros((num_bin, num_bin)), 'num_1p': np.zeros((num_bin, num_bin)), 
            '1m': np.zeros((num_bin, num_bin)), 'num_1m': np.zeros((num_bin, num_bin)),
            '2p': np.zeros((num_bin, num_bin)), 'num_2p': np.zeros((num_bin, num_bin)),
            '2m': np.zeros((num_bin, num_bin)), 'num_2m': np.zeros((num_bin, num_bin))} for b in range(binnum)}
e1_std = {b: {c: [] for c in range(num_bin**2)} for b in range(binnum)}
e2_std = {b: {c: [] for c in range(num_bin**2)} for b in range(binnum)}
for fp in tqdm(fplane_files):
    with open(fp, 'rb') as handle:
        d = pickle.load(handle)
        handle.close()
    if len(d['noshear']) == 0:
        print(fp)
        continue
    for cname in ['noshear', '1p', '1m', '2p', '2m']:
        x = d[cname]['x'] - 1125 + toff; y = d[cname]['y'] -2048 + toff # 1125 and 2048 are subtracted to get the center of focal plane back to (0,0)
        psf = d[cname]['psf_g1']
        e1 = d[cname]['e1']
        e2 = d[cname]['e2']
        xind = np.floor(x/side).astype(int); yind = np.floor(y/side).astype(int)
        bind = areaids.reshape(num_bin, num_bin)[yind, xind]

        for b,(low,high) in enumerate(zip(bin_dict['low'], bin_dict['high'])):
            msk_bin = ((psf > low) & (psf < high))
            _accum_shear(res_e1, cname, cname, xind[msk_bin], yind[msk_bin], e1[msk_bin], b, fp=True)
            _accum_shear(res_e2, cname, cname, xind[msk_bin], yind[msk_bin], e2[msk_bin], b, fp=True)
            # _accum_shear(res_e1, "1p", "1p", xind[msk_bin], yind[msk_bin], e1[msk_bin], b, fp=True)
            # _accum_shear(res_e1, "1m", "1m", xind[msk_bin], yind[msk_bin], e1[msk_bin], b, fp=True)
            # _accum_shear(res_e2, "2p", "2p", xind[msk_bin], yind[msk_bin], e2[msk_bin], b, fp=True)
            # _accum_shear(res_e2, "2m", "2m", xind[msk_bin], yind[msk_bin], e2[msk_bin], b, fp=True)
            if cname == 'noshear':
                 _accum_std(e1_std, "noshear", "noshear", bind[msk_bin], e1[msk_bin], b)
                 _accum_std(e2_std, "noshear", "noshear", bind[msk_bin], e2[msk_bin], b)
# 3. For each PSF shape bin, compute response and mean shear.
for b in range(binnum):
    for c in range(num_bin**2):
        e1_std[b][c] = np.std(np.concatenate(e1_std[b][c]))
        e2_std[b][c] = np.std(np.concatenate(e2_std[b][c]))

res = {'e1': res_e1, 'e2': res_e2, 'e1_std': e1_std, 'e2_std': e2_std}
with open('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/v2/mean_shear_focal_plane_coordinates.pickle', 'wb') as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)









# get_stats_in_slice_coordinates()
# get_focal_plane_coordinates()