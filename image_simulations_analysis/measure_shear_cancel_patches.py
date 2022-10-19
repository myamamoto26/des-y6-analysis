from __future__ import print_function
import sys
import os
import numpy as np
import fitsio
import tqdm
from joblib import Parallel, delayed

SHEARS = ['1p', '1m', '2p', '2m']
N_PATCH = 20
DPATCH = 10_000 / N_PATCH
MASK_FRAC_CUT = 1.0


def _msk_it(d, s2n_cut=None, size_cut=None, shear=''):
    return (
        # (d['flags'] == 0) &
        # (np.sum(d['psf_flags'], axis=1) == 0) &
        # (d['obj_flags'] == 0) &
        # (d['gauss_flags'] == 0) &
        (d['mcal_flags'] == 0) &
        ((d['mcal_s2n' + shear]) > s2n_cut) &
        ((d['mcal_T_ratio' + shear]) > size_cut)
    )


def _measure_sums(d, s2n_cut=None, size_cut=None):
    msks = {}
    for shear in SHEARS:
        msks[shear] = _msk_it(
            d, s2n_cut=s2n_cut, size_cut=size_cut, shear='_' + shear)
    msks['noshear'] = _msk_it(
        d, s2n_cut=s2n_cut, size_cut=size_cut, shear='_noshear')

    g1p = np.mean(d['mcal_g_1p'][msks['1p'], 0])
    g1m = np.mean(d['mcal_g_1m'][msks['1m'], 0])
    g2p = np.mean(d['mcal_g_2p'][msks['2p'], 1])
    g2m = np.mean(d['mcal_g_2m'][msks['2m'], 1])
    g1 = np.mean(d['mcal_g_noshear'][msks['noshear'], 0])
    g2 = np.mean(d['mcal_g_noshear'][msks['noshear'], 1])
    R11 = (g1p - g1m) / 2 / 0.01
    R22 = (g2p - g2m) / 2 / 0.01

    return (
        g1, np.sum(msks['noshear']),
        g2, np.sum(msks['noshear']),
        R11,
        R22,
    )


def _process_tile(fname_plus, fname_minus, fname_tp):
    s2n_cut = 10.0 
    size_cut = 0.5

    print("    reading data", flush=True, file=sys.stderr)
    dp = fitsio.read(fname_plus)
    dm = fitsio.read(fname_minus)
    tp = fitsio.read(fname_tp)

    #assert np.array_equal(dp["ra"], dm["ra"])
    #assert np.array_equal(dp["ra"], tp["ra"])
    #assert np.array_equal(dp["dec"], dm["dec"])
    #assert np.array_equal(dp["dec"], tp["dec"])

    # make the patches
    dd = [dp, dm]
    data_plus = []
    data_minus = []
    ite = 0
    for d in dd: 
        print("    computing patches", flush=True, file=sys.stderr)
        xind = np.floor(d["x"] / DPATCH).astype(int)
        yind = np.floor(d["y"] / DPATCH).astype(int)
        patch_inds = xind + N_PATCH * yind

        s = np.argsort(patch_inds)
        d = d[s]
        #dp = dp[s]
        #dm = dm[s]
        #tp = tp[s]
        patch_inds = patch_inds[s]

        bin_edges = [0]
        i = 0
        curr = patch_inds[i]
        while i < patch_inds.shape[0]:
            if patch_inds[i] != curr:
                bin_edges.append(i)
                curr = patch_inds[i]
            i += 1
        bin_edges.append(i)

        for i in range(len(bin_edges)-1):
            start = bin_edges[i]
            end = bin_edges[i+1]
            _d = _measure_sums(d[start:end], s2n_cut=s2n_cut, size_cut=size_cut)
            #_dm = _measure_sums(dm[start:end], s2n_cut=s2n_cut, size_cut=size_cut)
            if np.all(np.isfinite(_d)):
                if ite == 0:
                    data_plus.append(_d)
                    #data_minus.append(_dm)
                elif ite == 1:
                    data_minus.append(_d)
        ite+=1

    return data_plus, data_minus


def _measure_m_c(dp, dm, swap=False):
    np1 = np.sum(dp['ng1'])
    nm1 = np.sum(dm['ng1'])
    np2 = np.sum(dp['ng2'])
    nm2 = np.sum(dm['ng2'])

    R11 = (
        np.sum(dp['R11'] * dp['ng1']) / np1
        + np.sum(dm['R11'] * dm['ng1']) / nm1
    ) / 2.0

    R22 = (
        np.sum(dp['R22'] * dp['ng2']) / np2
        + np.sum(dm['R22'] * dm['ng2']) / nm2
    ) / 2.0

    if swap:
        gm = (
            np.sum(dp['g2'] * dp['ng2']) / np2 -
            np.sum(dm['g2'] * dm['ng2']) / nm2
        ) / 2.0 / R22
        m = (gm / 0.02 - 1.0)

        c = (
            np.sum(dp['g1'] * dp['ng1']) / np1 +
            np.sum(dm['g1'] * dm['ng1']) / nm1
        ) / 2.0 / R22
    else:
        gm = (
            np.sum(dp['g1'] * dp['ng1']) / np1 -
            np.sum(dm['g1'] * dm['ng1']) / nm1
        ) / 2.0 / R11
        m = (gm / 0.02 - 1.0)

        c = (
            np.sum(dp['g2'] * dp['ng2']) / np2 +
            np.sum(dm['g2'] * dm['ng2']) / nm2
        ) / 2.0 / R22

    return m, c


def _measure_m_c_boot(seed, dp, dm, swap=False):
    rng = np.random.RandomState(seed=seed)
    inds = rng.choice(dp.shape[0], size=dp.shape[0], replace=True)
    return _measure_m_c(dp[inds], dm[inds], swap=swap)


# read files and measure shear in patches
fnames = sys.argv[1:]
swap = False
neg = False
fnames_to_read = []
for fname in fnames:
    if fname == "--swap":
        swap = True
    else:
        fnames_to_read.append(fname)

jobs = []
for fname_plus in fnames_to_read:
    tilename = os.path.basename(fname_plus)[:len("DES0007-5957")]

    if swap:
        fname_minus = fname_plus.replace("g10.00_g20.02", "g10.00_g2-0.02")
    else:
        fname_minus = fname_plus.replace("g10.02_g20.00", "g1-0.02_g20.00")
    #fname_tp = os.path.join(
    #    "/global/cscratch1/sd/myamamot/",
    #    "true_positions/%s-truepositions.fits" % tilename)
    fname_tp = "/global/cscratch1/sd/myamamot/imsim/2022_10_03_additive_bias_in_grid/g1002/true_positions/DES0433-2332-truepositions.fits"
    if not (
        os.path.exists(fname_plus)
        and os.path.exists(fname_minus)
        and os.path.exists(fname_tp)
    ):
        continue

    print("processing tile:", tilename, flush=True, file=sys.stderr)
    print("    plus:", fname_plus, flush=True, file=sys.stderr)
    print("    minus:", fname_minus, flush=True, file=sys.stderr)
    print("    true pos:", fname_tp, flush=True, file=sys.stderr)

    jobs.append(delayed(_process_tile)(fname_plus, fname_minus, fname_tp))

dt = [
    ('g1', 'f8'),
    ('ng1', 'f8'),
    ('g2', 'f8'),
    ('ng2', 'f8'),
    ('R11', 'f8'),
    ('R22', 'f8'),
]

res = Parallel(n_jobs=-1, verbose=0)(jobs)
dp = []
dm = []
for r in res:
    rp, rm = r
    dp.extend(rp)
    dm.extend(rm)
dp = np.array(dp, dtype=dt)
dm = np.array(dm, dtype=dt)

# combine the patches w/ bootstrapping to produce estimates of m and c

n_boot = 100
marr = []
carr = []
rng = np.random.RandomState(seed=10)
seeds = rng.randint(1, 2**31, size=n_boot)

jobs = [
    delayed(_measure_m_c_boot)(seed, dp, dm, swap=swap)
    for seed in seeds
]
res = Parallel(n_jobs=-1, verbose=0)(jobs)

for r in tqdm.tqdm(res):
    m, c = r
    marr.append(m)
    carr.append(c)

m, c = _measure_m_c(dp, dm, swap=swap)
m_err = np.std(marr)
c_err = np.std(carr)

print("m: %f +/- %f" % (m, m_err))
print("c: %f +/- %f" % (c, c_err))
