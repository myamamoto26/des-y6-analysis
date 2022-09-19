import fitsio as fio 
import numpy as np 
import os
import glob

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

    return g1/R11, g2/R22, R11, R22

def _msk_it(d, mdet_mom, s2n_cut=None, size_cut=None, shear=''):
    return (
        (d[mdet_mom+"_flags"] == 0) & 
        (d["mask_flags"] == 0) & 
        (d[mdet_mom+"_s2n"] > 10) & 
        (d["mfrac"] < 0.1))


def _measure_m_c(res_g1p, res_g1m):

    g1_p, g2_p, R11_p, R22_p = _compute_shear_response(res_g1p)
    g1_m, g2_m, R11_m, R22_m = _compute_shear_response(res_g1m) 

    m = (g1_p - g1_m)/0.04/2 - 1.0
    c = (g2_p + g2_m)/2.0
    print(g1_p, g1_m)
    return m, c

# def _measure_m_c_boot(seed, d_p, d_m):

#     rng = np.random.RandomState(seed=seed)
#     inds = rng.choice(d_p.shape[0], size=dp.shape[0], replace=True)
#     return _measure_m_c(dp[inds], dm[inds], swap=swap)

# read in simulated metadetection catalogs
mdet_mom = 'pgauss'
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
gobs_raw = {'g1p': {'g1': [], 'g2': []}, 'g1m': {'g1': [], 'g2': []}, 'g2p': {'g1': [], 'g2': []}, 'g2m': {'g1': [], 'g2': []}}
for tilename in tilenames[:-1]:
    print('processing ', tilename)
    d_p = fio.read(f"""/global/cfs/cdirs/des/y6-image-sims/eastlake/g1002/{tilename}_metadetect-v7_mdetcat_part0000.fits""")
    d_m = fio.read(f"""/global/cfs/cdirs/des/y6-image-sims/eastlake/g1n002/{tilename}_metadetect-v7_mdetcat_part0000.fits""")

    d_p = d_p[_msk_it(d_p, mdet_mom)]
    d_m = d_m[_msk_it(d_m, mdet_mom)]

    res_g1p = _accum_shear_per_tile(res_g1p, d_p['mdet_step'], d_p[mdet_mom+'_g_1'], d_p[mdet_mom+'_g_2'])
    res_g1m = _accum_shear_per_tile(res_g1m, d_m['mdet_step'], d_m[mdet_mom+'_g_1'], d_m[mdet_mom+'_g_2'])

    # gobs_raw['g1p']['g1'].append(d_p[mdet_mom+'_g_1'])
    # gobs_raw['g1p']['g2'].append(d_p[mdet_mom+'_g_2'])
    # gobs_raw['g1m']['g1'].append(d_m[mdet_mom+'_g_1'])
    # gobs_raw['g1m']['g2'].append(d_m[mdet_mom+'_g_2'])

# raw shears
# gobs_raw['g1p']['g1'] = np.concatenate(gobs_raw['g1p']['g1'])
# gobs_raw['g1p']['g2'] = np.concatenate(gobs_raw['g1p']['g2'])
# gobs_raw['g1m']['g1'] = np.concatenate(gobs_raw['g1m']['g1'])
# gobs_raw['g1m']['g2'] = np.concatenate(gobs_raw['g1m']['g2'])

# measure mc
m, c = _measure_m_c(res_g1p, res_g1m)

# measure mc boot
m_err = 0.00
c_err = 0.00

print("m: %f +/- %f" % (m, m_err))
print("c: %f +/- %f" % (c, c_err))