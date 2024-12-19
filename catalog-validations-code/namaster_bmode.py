import numpy as np
import glob
import pymaster as nmt
import os
import pickle
import healpy as hp
from tqdm import tqdm

def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
    """
    Computes indices of 2D grids. Only used when we use shear weight that is binned by S/N and size ratio. 
    """
    from math import log10
    # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps

    logstepx = log10(xmax/xmin)/xsteps
    logstepy = log10(ymax/ymin)/ysteps

    indexx = (np.log10(x/xmin)/logstepx).astype(int)
    indexy = (np.log10(y/ymin)/logstepy).astype(int)

    indexx = np.maximum(indexx,0)
    indexx = np.minimum(indexx, xsteps-1)
    indexy = np.maximum(indexy,0)
    indexy = np.minimum(indexy, ysteps-1)

    return indexx,indexy

def _find_shear_weight(d, mask, wgt_dict, snmin, snmax, sizemin, sizemax, steps, mdet_mom):

    """
    Assigns shear weights to the objects based on the grids. 
    """
    
    if wgt_dict is None:
        weights = np.ones(len(d))
        return weights

    shear_wgt = wgt_dict['weight']
    smoothing = True
    if smoothing:
        from scipy.ndimage import gaussian_filter
        smooth_response = gaussian_filter(wgt_dict['response'], sigma=2.0)
        shear_wgt = (smooth_response/wgt_dict['meanes'])**2
    indexx, indexy = assign_loggrid(np.array(d[mdet_mom+'_s2n'])[mask], np.array(d[mdet_mom+'_T_ratio'])[mask], snmin, snmax, steps, sizemin, sizemax, steps)
    weights = np.array([shear_wgt[x, y] for x, y in zip(indexx, indexy)])
    
    return weights


def _get_shear_weights(dat, mask, weight_file):
    shape_err = False
    if shape_err:
        return 1/(0.22**2 + 0.5*(np.array(dat['gauss_g_cov_1_1'])[mask] + np.array(dat['gauss_g_cov_2_2'])[mask]))
    else:
        with open(os.path.join(weight_file), 'rb') as handle:
            wgt_dict = pickle.load(handle)
            snmin = wgt_dict['xedges'][0]
            snmax = wgt_dict['xedges'][-1]
            sizemin = wgt_dict['yedges'][0]
            sizemax = wgt_dict['yedges'][-1]
            steps = len(wgt_dict['xedges'])-1
        shear_wgt = _find_shear_weight(dat, mask, wgt_dict, snmin, snmax, sizemin, sizemax, steps, 'gauss')
        return shear_wgt
    

def _wmean(q,w):
    return np.sum(q*w)/np.sum(w)

def read_mdet_h5(datafile, keys, weight_file, patch_id=None, response=False, subtract_mean_shear=False):

    import h5py as h5
    f = h5.File(datafile, 'r')
    d = f.get('/mdet/noshear')
    # mask_mfrac = False; mfrac = 0.01
    nrows = len(np.array( d['ra'] ))
    mask = np.ones(nrows)!=0
    formats = []
    for key in keys:
        formats.append('f4')
    data = np.recarray(shape=(nrows,), formats=formats, names=keys)
    for key in keys:  
        if key == 'R':
            continue
        elif key == 'w':
            data['w'] = _get_shear_weights(d, mask, weight_file)
        elif key in ('g1', 'g2'):
            data[key] = np.array(d['gauss_'+key[0]+'_'+key[1]])[mask]
        elif key in ('g1_cov', 'g2_cov'):
            data[key] = np.array(d['gauss_'+key[0]+'_cov_'+key[1]+'_'+key[1]])[mask]
        else:
            data[key] = np.array(d[key])[mask]
    print('made recarray with hdf5 file')
    
    # response correction
    if response:
        d_2p = f.get('/mdet/2p'); mask_2p = np.ones(len(np.array( d_2p['ra'] )))!=0
        d_1p = f.get('/mdet/1p'); mask_1p = np.ones(len(np.array( d_1p['ra'] )))!=0
        d_2m = f.get('/mdet/2m'); mask_2m = np.ones(len(np.array( d_2m['ra'] )))!=0
        d_1m = f.get('/mdet/1m'); mask_1m = np.ones(len(np.array( d_1m['ra'] )))!=0
        # compute response with weights
        g1p = _wmean(np.array(d_1p["gauss_g_1"])[mask_1p], _get_shear_weights(d_1p, mask_1p, weight_file))                                     
        g1m = _wmean(np.array(d_1m["gauss_g_1"])[mask_1m], _get_shear_weights(d_1m, mask_1m, weight_file))
        R11 = (g1p - g1m) / 0.02

        g2p = _wmean(np.array(d_2p["gauss_g_2"])[mask_2p], _get_shear_weights(d_2p, mask_2p, weight_file))
        g2m = _wmean(np.array(d_2m["gauss_g_2"])[mask_2m], _get_shear_weights(d_2m, mask_2m, weight_file))
        R22 = (g2p - g2m) / 0.02

        R = (R11 + R22)/2.
        data['g1'] /= R
        data['g2'] /= R

    mean_g1 = _wmean(data['g1'], data['w'])
    mean_g2 = _wmean(data['g2'], data['w'])
    std_g1 = np.var(data['g1'])
    std_g2 = np.var(data['g2'])
    mean_shear = [mean_g1, mean_g2, std_g1, std_g2]
    # mean shear subtraction
    if subtract_mean_shear:
        print('subtracting mean shear')
        print('mean g1 g2 =(%1.8f,%1.8f)'%(mean_g1, mean_g2))          
        data['g1'] -= mean_g1
        data['g2'] -= mean_g2

    return data, mean_shear

def read_mdet_h5_tomobin_master(datafile, keys, patch_id=None, response=False, subtract_mean_shear=False):
    
    tomobins = ['/tomo_bin_0', '/tomo_bin_1', '/tomo_bin_2', '/tomo_bin_3']
    data_all = {}
    for t in tomobins:
    
        import h5py as h5
        f = h5.File(datafile, 'r')
        # d = f.get('noshear'+t)
        d = f.get('noshear'+t)
        if patch_id is None:
            nrows = len(np.array( d['ra'] ))
            mask = np.ones(nrows)!=0
        else:
            mask = (np.array( d['patch_num'] ) == patch_id)
            nrows = len(np.array( d['ra'] )[mask])
        formats = []
        for key in keys:
            formats.append('f4')
        data = np.recarray(shape=(nrows,), formats=formats, names=keys)

        data_all[t[-5:]] = data
        for key in keys:  
            if key == 'R':
                continue
            elif key in ('g1', 'g2'):
                data[key] = np.array(d['gauss_'+key[0]+'_'+key[1]])[mask]
            elif key in ('g1_cov', 'g2_cov'):
                data[key] = np.array(d['gauss_'+key[0]+'_cov_'+key[1]+'_'+key[1]])[mask]
            else:
                data[key] = np.array(d[key])[mask]
        print('made recarray with hdf5 file')

        # response correction
        if response:
            d_2p = f.get('2p'+t); nrows = len(np.array( d_2p['ra'] )); mask_2p = np.ones(nrows)!=0
            d_1p = f.get('1p'+t); nrows = len(np.array( d_1p['ra'] )); mask_1p = np.ones(nrows)!=0
            d_2m = f.get('2m'+t); nrows = len(np.array( d_2m['ra'] )); mask_2m = np.ones(nrows)!=0
            d_1m = f.get('1m'+t); nrows = len(np.array( d_1m['ra'] )); mask_1m = np.ones(nrows)!=0
            # compute response with weights
            g1p = _wmean(np.array(d_1p["gauss_g_1"]), np.array(d_1p['w']))                                     
            g1m = _wmean(np.array(d_1m["gauss_g_1"]), np.array(d_1m['w']))
            R11 = (g1p - g1m) / 0.02

            g2p = _wmean(np.array(d_2p["gauss_g_2"]), np.array(d_2p['w']))
            g2m = _wmean(np.array(d_2m["gauss_g_2"]), np.array(d_2m['w']))
            R22 = (g2p - g2m) / 0.02

            R = (R11 + R22)/2.
            print('weighted shear response is ', R)
            data['g1'] /= R
            data['g2'] /= R

            mean_g1 = _wmean(data['g1'], data['w'])
            mean_g2 = _wmean(data['g2'], data['w'])
            std_g1 = np.var(data['g1'])
            std_g2 = np.var(data['g2'])
            mean_shear = [mean_g1, mean_g2, std_g1, std_g2]
            # mean shear subtraction
            if subtract_mean_shear:
                print('subtracting mean shear')
                print('mean g1 g2 =(%1.8f,%1.8f)'%(mean_g1, mean_g2))          
                data['g1'] -= mean_g1
                data['g2'] -= mean_g2
                
    return data_all, mean_shear

def apply_random_rotation(e1_in, e2_in):
    np.random.seed() # CRITICAL in multiple processes !
    rot_angle = np.random.rand(len(e1_in))*2*np.pi
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = + e1_in * cos + e2_in * sin
    e2_out = - e1_in * sin + e2_in * cos
    return e1_out, e2_out

def make_maps(nside, d, col, mean_shear=True):

    n_source = np.zeros(hp.nside2npix(nside)); n_source_w = np.zeros(hp.nside2npix(nside))
    pix = hp.ang2pix(nside, d['ra'], d['dec'], lonlat=True, nest=True)
    unique_pix1, idx1, idx_rep1 = np.unique(pix, return_index=True, return_inverse=True)
    n_source[unique_pix1] += np.bincount(idx_rep1, weights=np.ones(len(d['ra'])))
    n_source_w[unique_pix1] += np.bincount(idx_rep1, weights=d['w'])

    maps = np.zeros(hp.nside2npix(nside)); maps_w = np.zeros(hp.nside2npix(nside))
    de = d[col]
    if mean_shear:
        emean = np.average(de, weights=d['w'])
        maps[unique_pix1] += np.bincount(idx_rep1, weights=de-emean)
        maps_w[unique_pix1] += np.bincount(idx_rep1, weights=(de-emean)*d['w'])
    else:
        maps[unique_pix1] += np.bincount(idx_rep1, weights=de)
        maps_w[unique_pix1] += np.bincount(idx_rep1, weights=de*d['w'])
    maps[n_source!=0] /= n_source[n_source!=0]
    maps_w[n_source_w!=0] /= n_source_w[n_source_w!=0]

    mask = np.zeros_like(maps_w)
    mask[maps_w!=0] = 1.

    return maps, maps_w, mask

nside = 1024
b = nmt.NmtBin.from_nside_linear(nside, 100)
leff = b.get_effective_ells()
outpath = '/pscratch/sd/m/myamamot/des-y6-analysis/cosmicshear/v6_UNBLINDED/Bmode/namaster/'

# C_ell computation --------------------------------------------------------------------
sims = True; parallel = False; nontomo = True
if parallel:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)

if nontomo and not sims:
    print('running nontomo')
    keys = ['ra', 'dec', 'g1', 'g2', 'w']
    input_file = '/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V6_UNBLINDED/metadetect_cutsv6_all.h5'
    w_file = '/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v6_UNBLINDED/inverse_variance_weight_v6.pickle'
    gal_data, mean_shear = read_mdet_h5(input_file, keys, w_file, response=True, subtract_mean_shear=True)

    bmode = np.zeros(len(leff), dtype=[('b', 'f8'), ('bw', 'f8'), ('berr', 'f8'), ('b_noise', 'f8')])
    e1,e1_w,mask = make_maps(nside, gal_data, 'g1')
    e2,e2_w,mask = make_maps(nside, gal_data, 'g2')

    # f_np = nmt.NmtField(mask, [e1,-e2])
    f_np_w = nmt.NmtField(mask, [e1_w,-e2_w])
    print('finished making e map')

    # w_np = nmt.NmtWorkspace()
    # w_np.compute_coupling_matrix(f_np, f_np, b)

    # cl_coupled = nmt.compute_coupled_cell(f_np, f_np)
    # cl_decoupled = w_np.decouple_cell(cl_coupled)

    w_np = nmt.NmtWorkspace()
    w_np.compute_coupling_matrix(f_np_w, f_np_w, b)

    cl_coupled_w = nmt.compute_coupled_cell(f_np_w, f_np_w)
    cl_decoupled_w = w_np.decouple_cell(cl_coupled_w)

    ncl = []
    for i in tqdm(range(800)):
        e1_noise, e2_noise = apply_random_rotation(e1_w, -e2_w)
        f_np_n = nmt.NmtField(mask, [e1_noise,e2_noise])

        w_np = nmt.NmtWorkspace()
        w_np.compute_coupling_matrix(f_np_n, f_np_n, b)

        Ncl_coupled = nmt.compute_coupled_cell(f_np_n, f_np_n)
        Ncl_decoupled = w_np.decouple_cell(Ncl_coupled)
        ncl.append(Ncl_decoupled[3])

    # bmode['b'] = cl_decoupled[3]
    bmode['bw'] = cl_decoupled_w[3]
    bmode['b_noise'] = np.mean(ncl, axis=0)
    bmode['berr'] = np.std(ncl, axis=0)
    np.save(outpath+'bmode_namaster_nontomo_unblinded_g2flip.npy', bmode)

# Get galaxy info
keys = ['ra', 'dec', 'g1', 'g2', 'w']
input_file = '/global/cfs/cdirs/des/y6-shear-catalogs/master/metadetect_2024-07-29.hdf5'
gal_data, mean_shear = read_mdet_h5_tomobin_master(input_file, keys, response=True, subtract_mean_shear=True)

emap_dict = {}
mask_dict = {}
zbin_pair = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
if nontomo: 
    zbin = [0]
else:
    zbin = [0,1,2,3]
if sims:
    print('computing cell for sims')
    for bi in zbin:
        # if bi == 0:
        #     continue
        print('bin %s' % bi)
        if nontomo:
            sim_path = '/pscratch/sd/m/myamamot/sample_variance/v6_UNBLINDED/'
        else:
            sim_path = '/global/cfs/cdirs/des/myamamot/sample_variance/v6_UNBLINDED_cosmogrid_bin%s/' % bi
        
        for seed in tqdm(range(800)):

            # if seed % size != rank:
            #     continue

            outpath_sims = os.path.join(outpath, 'sims')
            if os.path.exists(os.path.join(outpath_sims, 'bmode_bin%s_' % bi + str(seed+1)+'.npy')):
                continue

            with open(os.path.join(sim_path, 'seed__fid_cosmogrid_'+str(seed+1)+'.pkl'), 'rb') as f:
                d_sim = pickle.load(f)['sources'][0]

            bmode = np.zeros(len(leff), dtype=[('b', 'f8'), ('bw', 'f8'), ('b_noise', 'f8')])
            e1,e1_w,mask = make_maps(nside, d_sim, 'e1')
            e2,e2_w,mask = make_maps(nside, d_sim, 'e2')
            # f_np = nmt.NmtField(mask, [e1,e2])
            f_np_w = nmt.NmtField(mask, [e1_w,e2_w])

            # w_np = nmt.NmtWorkspace()
            # w_np.compute_coupling_matrix(f_np, f_np, b)

            # cl_coupled = nmt.compute_coupled_cell(f_np, f_np)
            # cl_decoupled = w_np.decouple_cell(cl_coupled)
            # bmode['b'] = cl_decoupled[3]

            w_np = nmt.NmtWorkspace()
            w_np.compute_coupling_matrix(f_np_w, f_np_w, b)

            cl_coupled_w = nmt.compute_coupled_cell(f_np_w, f_np_w)
            cl_decoupled_w = w_np.decouple_cell(cl_coupled_w)
            bmode['bw'] = cl_decoupled_w[3]
            
            np.save(outpath_sims+'/bmode_bin%s_' % bi +str(seed+1)+'.npy', bmode)

else:
    for bi in zbin:
        d = gal_data['bin_%s' % bi]
        e1_map,e1_w_map,mask = make_maps(nside, d, 'g1')
        e2_map,e2_w_map,mask = make_maps(nside, d, 'g2')
        emap_dict[str(bi)] = [e1_map, e2_map, e1_w_map, e2_w_map]

        mask = np.zeros(nside**2*12)
        mask[e1_map!=0] = 1.
        mask_dict[str(bi)] = mask
        print('finished making e map')

    print('computing cell for data')
    bmode_dict = {}
    for bi in zbin:
        if bi != 3:
            continue
        print('bin %s' % bi)
        bmode = np.zeros(len(leff), dtype=[('b', 'f8'), ('bw', 'f8'), ('berr', 'f8'), ('b_noise', 'f8')])
        e1 = emap_dict[str(bi)][0]
        e2 = emap_dict[str(bi)][1]
        f_np = nmt.NmtField(mask, [e1,e2])

        e1_w = emap_dict[str(bi)][2]
        e2_w = emap_dict[str(bi)][3]
        f_np_w = nmt.NmtField(mask, [e1_w,e2_w])

        w_np = nmt.NmtWorkspace()
        w_np.compute_coupling_matrix(f_np, f_np, b)

        cl_coupled = nmt.compute_coupled_cell(f_np, f_np)
        cl_decoupled = w_np.decouple_cell(cl_coupled)

        w_np = nmt.NmtWorkspace()
        w_np.compute_coupling_matrix(f_np_w, f_np_w, b)

        cl_coupled_w = nmt.compute_coupled_cell(f_np_w, f_np_w)
        cl_decoupled_w = w_np.decouple_cell(cl_coupled_w)

        ncl_ = []
        for i in tqdm(range(800)):
            e1_noise, e2_noise = apply_random_rotation(e1_w, e2_w)
            f_np = nmt.NmtField(mask, [e1_noise,e2_noise])

            w_np = nmt.NmtWorkspace()
            w_np.compute_coupling_matrix(f_np, f_np, b)

            Ncl_coupled = nmt.compute_coupled_cell(f_np, f_np)
            Ncl_decoupled = w_np.decouple_cell(Ncl_coupled)
            ncl_.append(Ncl_decoupled[3])
        
        bmode['b'] = cl_decoupled[3]
        bmode['bw'] = cl_decoupled_w[3]
        bmode['b_noise'] = np.mean(ncl_, axis=0)
        bmode['berr'] = np.std(ncl_, axis=0)
        bmode_dict[str(bi)] = bmode
        np.save(outpath+'bmode_namaster_bin%s.npy' % bi, bmode)

"""
import healpy as hp
config = dict()
config['nside'] = 512
# apodize mask
def doit(fname,outfname,ell_eff,weight_map):

    with open(fname, 'rb') as f:
        cat = pickle.load(f)['sources'][0]

    for rel in range(4):
        path = outfname+'_rel%s.npy' % str(rel)
        if not os.path.exists(path):
            cls_ = dict()

            cls_e1e2 = dict()
            for bin1 in range(1,5):
                for bin2 in range (bin1,5):
                    binx = '{0}_{1}'.format(bin1,bin2)
                    #print (np.sum(mask))
                    #print (cat[rel][bin1]['kE'])

                    e1 = np.zeros(hp.nside2npix(config['nside']))
                    e2 = np.zeros(hp.nside2npix(config['nside']))
                    e1n = np.zeros(hp.nside2npix(config['nside']))
                    e2n = np.zeros(hp.nside2npix(config['nside']))
                    e1[cat[rel][bin1]['pix']] = cat[rel][bin1]['e1']
                    e2[cat[rel][bin1]['pix']] = cat[rel][bin1]['e2']
                    e1n[cat[rel][bin1]['pix']] = cat[rel][bin1]['e1n']
                    e2n[cat[rel][bin1]['pix']] = cat[rel][bin1]['e2n']
                    mask = np.in1d(np.arange(len(e1)),cat[rel][bin1]['pix'])
                    be1 = np.zeros(hp.nside2npix(config['nside']))
                    be2 = np.zeros(hp.nside2npix(config['nside']))
                    be1n = np.zeros(hp.nside2npix(config['nside']))
                    be2n = np.zeros(hp.nside2npix(config['nside']))
                    be1[cat[rel][bin2]['pix']] = cat[rel][bin2]['e1']
                    be2[cat[rel][bin2]['pix']] = cat[rel][bin2]['e2']
                    be1n[cat[rel][bin2]['pix']] = cat[rel][bin2]['e1n']
                    be2n[cat[rel][bin2]['pix']] = cat[rel][bin2]['e2n']      
                    
                    
                   # f_0a = nmt.NmtField(weight_map[rel][bin1], [e1,e2])
                   # f_0b = nmt.NmtField(weight_map[rel][bin2], [be1,be2])
                   # f_2a = nmt.NmtField(weight_map[rel][bin1], [e1n,e2n])
                   # f_2b = nmt.NmtField(weight_map[rel][bin2], [be1n,be2n])
                    
                    m_1 = np.zeros(len(weight_map[rel][bin1]))
                    m_1[weight_map[rel][bin1]!=0] =1.
                    m_2 = np.zeros(len(weight_map[rel][bin1]))
                    m_2[weight_map[rel][bin2]!=0] =1.

                    f_0a = nmt.NmtField(m_1, [e1,e2])
                    f_0b = nmt.NmtField(m_2, [be1,be2])
                    f_2a = nmt.NmtField(m_1, [e1n,e2n])
                    f_2b = nmt.NmtField(m_2, [be1n,be2n])

            
            
                    cl_22 = nmt.compute_full_master(f_0a, f_0b, b)
                    cl_22n = nmt.compute_full_master(f_2a, f_2b, b)

      
                    cls_[binx] = [cl_22,cl_22n,ell_eff]
            np.save(root+name+'_rel{0}'.format(rel),cls_)

if __name__ == '__main__':
    
    outpath = '/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/bmode'
    sim_path = '/pscratch/sd/m/myamamot/sample_variance/v5_catalog_cosmogrid'
    files_ = glob.glob(os.path.join(sim_path,'seed__fid_cosmogrid_*.pkl'))

    weight_map = np.load('/global/cfs/cdirs/des/mass_maps/Maps_final/weight_maps.npy',allow_pickle=True).item()


    nside = 512 # this is the nside of the maps
    _nside_h = 512 # this one is only used to determine the binning, keep it fixed
    lmin = 8
    lmax = 3*_nside_h
    b_lmax = 3*nside-1
    n_ell_bins = 28

    b = harm.utils.make_nmtbin_powspaced(_nside_h, lmin, lmax, n_ell_bins, power=0.5, verbose=True, b_lmax=b_lmax, f_ell='pixwin') 
    ell_eff = b.get_effective_ells()

    if not sims:
         doit(run_count, ell_eff, weight_map)
    else:
        files = []
        
        import os
        count = 0
        nn = 0
        for file in files_:
            
                seed = file.split('/')[-1].split('_')[-1].split('.')[0]
                xx = os.path.join(outpath, 'namaster_cll_%s_rel%s.npy' % (str(seed), 1))
                if os.path.exists(xx):
                    count +=1
                else:
                    files.append(file)
            
        print(count,len(files))
        from mpi4py import MPI 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        run_count = 0
        for i,file in enumerate(files):
            if i % size != rank:
                continue
            outf = os.path.join(outpath, 'namaster_cll_%s' % str(seed))
            doit(file, outf, ell_eff, weight_map)
#srun --nodes=4 --tasks-per-node=64 python compute_cl_full.py
"""