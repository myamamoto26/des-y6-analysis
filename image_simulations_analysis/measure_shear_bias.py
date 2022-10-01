import fitsio as fio 
import numpy as np 
import os
import glob
from tqdm import tqdm

N_PATCH = 20
DPATCH = 10_000 / N_PATCH

# accum shear
def _accum_shear_per_tile(res, mdet_step, g1, g2):

    """
    Returns the dictionary of the accumulated shear (sum of individual shear).

    Parameters
    ----------
    res: A dictionary in which accumulated sums of shear are stored
    mdet_step: An array of metadetection steps (noshear, 1p, 1m, 2p, 2m) for each object in metadetection catalog
    g1: An array of the measured shapes (e1) for each object in metadetection catalog
    g2: An array of the measured shapes (e2) for each object in metadetection catalog

    """

    for step in ['noshear', '1p', '1m', '2p', '2m']:
        msk_s = (mdet_step == step)
        
        np.add.at(
            res[step], 
            (0, 0), 
            np.sum(g1[msk_s]),
        )
        np.add.at(
            res[step], 
            (0, 1), 
            np.sum(g2[msk_s]),
        )
        np.add.at(
            res["num_" + step], 
            (0, 0), 
            len(g1[msk_s]),
        )
        np.add.at(
            res["num_" + step], 
            (0, 1), 
            len(g2[msk_s]),
        )
    return res

# measure response
def _compute_shear_response(res):

    g1 = res['noshear'][0][0] / res['num_noshear'][0][0]
    g1p = res['1p'][0][0] / res['num_1p'][0][0]
    g1m = res['1m'][0][0] / res['num_1m'][0][0]
    R11 = (g1p - g1m) / 2 / 0.01

    g2 = res['noshear'][0][1] / res['num_noshear'][0][1]
    g2p = res['2p'][0][1] / res['num_2p'][0][1]
    g2m = res['2m'][0][1] / res['num_2m'][0][1]
    R22 = (g2p - g2m) / 2 / 0.01

    return g1, g2, R11, R22

def _msk_it(d, mdet_mom, s2n_cut=None, size_cut=None, shear=''):
    return (
        (d[mdet_mom+"_flags"] == 0) & 
        (d["mask_flags"] == 0) & 
        (d[mdet_mom+"_s2n"] > 10) & 
        (d["mfrac"] < 0.1) &
        (d["shear_bands"] == "012") )


def _measure_m_c(res_g1p, res_g1m, swap=False):

    g1_p, g2_p, R11_p, R22_p = _compute_shear_response(res_g1p)
    g1_m, g2_m, R11_m, R22_m = _compute_shear_response(res_g1m) 

    # print(R11_p, R22_p, R11_m, R22_m)
    if swap: # m2, c1
        m = (g2_p - g2_m)/(R22_p + R22_m)/0.02 - 1.0
        c = (g1_p/R11_p + g1_m/R11_m)/2.0
    else: # m1, c2
        m = (g1_p - g1_m)/(R11_p + R11_m)/0.02 - 1.0
        c = (g2_p/R22_p + g2_m/R22_m)/2.0

    return m, c

def _process_tile(tilenames, mdet_mom):

    d_p_list = []
    d_m_list = []

    # accumulate bin_edges for all the tiles & subsets. 
    bin_edges = [np.array([0]), np.array([0])]
    tile_edge = [0, 0]
    jk_sample = 0
    # mdet_stats_p = np.zeros(nobj_p, dtype=[('g1', float), ('g2', float), ('mdet_step', float)])
    # mdet_stats_m = np.zeros(nobj_m dtype=[('g1', float), ('g2', float), ('mdet_step', float)])
    for tilename in tqdm(tilenames[:-1]):
        for ii in range(2):
            if ((ii%2 == 0) & (os.path.exists(f"""/global/cfs/cdirs/des/y6-image-sims/eastlake/g1002/{tilename}_metadetect-v7_mdetcat_part0000.fits"""))):
                d = fio.read(f"""/global/cfs/cdirs/des/y6-image-sims/eastlake/g1002/{tilename}_metadetect-v7_mdetcat_part0000.fits""")
                jk_sample += DPATCH
            elif ((ii%2 == 1) & (os.path.exists(f"""/global/cfs/cdirs/des/y6-image-sims/eastlake/g1n002/{tilename}_metadetect-v7_mdetcat_part0000.fits"""))):
                d = fio.read(f"""/global/cfs/cdirs/des/y6-image-sims/eastlake/g1n002/{tilename}_metadetect-v7_mdetcat_part0000.fits""")
            else:
                continue

            d = d[_msk_it(d, mdet_mom)]
            xind = np.floor(d["x"] / DPATCH).astype(int)
            yind = np.floor(d["y"] / DPATCH).astype(int)
            patch_inds = xind + N_PATCH * yind

            s = np.argsort(patch_inds)
            d = d[s]
            patch_inds = patch_inds[s]

            d_flat = np.zeros(len(d), dtype=[('g1', float), ('g2', float), ('mdet_step', object)])
            d_flat['g1'] = d[mdet_mom+'_g_1']
            d_flat['g2'] = d[mdet_mom+'_g_2']
            d_flat['mdet_step'] = d['mdet_step']
            if ii % 2 == 0:
                d_p_list.append(d_flat)
            elif ii % 2 == 1:
                d_m_list.append(d_flat)

            i = 0
            curr = patch_inds[i]
            while i < patch_inds.shape[0]:
                if patch_inds[i] != curr:
                    bin_edges[ii] = np.append(bin_edges[ii], i+tile_edge[ii])
                    # bin_edges.append(i)
                    curr = patch_inds[i]
                i += 1
            bin_edges[ii] = np.append(bin_edges[ii], i+tile_edge[ii])
            # bin_edges.append(i)

            tile_edge[ii] += len(patch_inds)
    d_p_list = np.concatenate(d_p_list)
    d_m_list = np.concatenate(d_m_list)

    # assert len(bin_edges[0])-1 == jk_sample
    return d_p_list, d_m_list, bin_edges

def _measure_m_c_jk(d_p, d_m, bin_edges):

    m_jk = []
    c_jk = []

    # compute shear bias for each bin.
    assert len(bin_edges[0]) == len(bin_edges[1])
    for i in tqdm(range(len(bin_edges[0])-1)):
        for ii in range(2):
            start = bin_edges[ii][i]
            end = bin_edges[ii][i+1]
            msk = np.arange(start, end)

            if ii%2 == 0:
                obj_id = np.arange(len(d_p))
                d_jk_p = d_p[np.in1d(obj_id, msk, invert=True)]
                res_jk_p = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
                '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
                '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
                '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
                '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
                res_jk_p = _accum_shear_per_tile(res_jk_p, d_jk_p['mdet_step'], d_jk_p['g1'], d_jk_p['g2'])
            elif ii%2 == 1:
                obj_id = np.arange(len(d_m))
                d_jk_m = d_m[np.in1d(obj_id, msk, invert=True)]
                res_jk_m = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
                '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
                '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
                '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
                '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
                res_jk_m = _accum_shear_per_tile(res_jk_m, d_jk_m['mdet_step'], d_jk_m['g1'], d_jk_m['g2'])

        m_sample, c_sample = _measure_m_c(res_jk_p, res_jk_m)
        m_jk.append(m_sample)
        c_jk.append(c_sample)

    return m_jk, c_jk


# read in simulated metadetection catalogs
mdet_mom = 'wmom'
f_tile=open('/global/cfs/cdirs/des/y6-image-sims/eastlake/mdet_files.txt', 'r')
tilenames = f_tile.read().split('\n')

binnum = 1
res_g1p = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
        '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
        '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
        '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
        '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
res_g1m = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
        '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
        '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
        '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
        '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}

nobj_p = 0
nobj_m = 0
for tilename in tilenames[:-1]:
    
    if os.path.exists(f"""/global/cfs/cdirs/des/y6-image-sims/eastlake/g1002/{tilename}_metadetect-v7_mdetcat_part0000.fits"""):
        print('processing ', tilename)
        d_p = fio.read(f"""/global/cfs/cdirs/des/y6-image-sims/eastlake/g1002/{tilename}_metadetect-v7_mdetcat_part0000.fits""")
        d_m = fio.read(f"""/global/cfs/cdirs/des/y6-image-sims/eastlake/g1n002/{tilename}_metadetect-v7_mdetcat_part0000.fits""")
    else: 
        print('missing ', tilename)

    d_p = d_p[_msk_it(d_p, mdet_mom)]
    nobj_p += len(d_p)
    d_m = d_m[_msk_it(d_m, mdet_mom)]
    nobj_m += len(d_m)

    # accumulate raw shears for the mean over all the tiles and patches. 
    res_g1p = _accum_shear_per_tile(res_g1p, d_p['mdet_step'], d_p[mdet_mom+'_g_1'], d_p[mdet_mom+'_g_2'])
    res_g1m = _accum_shear_per_tile(res_g1m, d_m['mdet_step'], d_m[mdet_mom+'_g_1'], d_m[mdet_mom+'_g_2'])

# measure mc
m, c = _measure_m_c(res_g1p, res_g1m)

# process tiles into patches for jackknife covariance
d_p_list, d_m_list, bin_edges = _process_tile(tilenames, mdet_mom)
m_jk, c_jk = _measure_m_c_jk(d_p_list, d_m_list, bin_edges)

N = len(bin_edges[0])
m_err = np.sqrt((N-1)/N)*np.sqrt(np.sum((m_jk - np.mean(m_jk))**2))
c_err = np.sqrt((N-1)/N)*np.sqrt(np.sum((c_jk - np.mean(c_jk))**2))

print("m: %f +/- %f" % (m, m_err))
print("c: %f +/- %f" % (c, c_err))