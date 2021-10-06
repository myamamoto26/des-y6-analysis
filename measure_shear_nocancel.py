from __future__ import print_function
import sys
import numpy as np
import fitsio
import tqdm
from joblib import Parallel, delayed

SHEARS = ['1p', '1m', '2p', '2m']


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


def _measure_R(d, s2n_cut=None, size_cut=None):
    msks = {}
    for shear in SHEARS:
        msks[shear] = _msk_it(
            d, s2n_cut=s2n_cut, size_cut=size_cut, shear='_' + shear)
    msks['noshear'] = _msk_it(
        d, s2n_cut=s2n_cut, size_cut=size_cut, shear='_noshear')

    g1_1p = np.mean(d['mcal_g_1p'][msks['1p'], 0])
    g1_1m = np.mean(d['mcal_g_1m'][msks['1m'], 0])
    g2_2p = np.mean(d['mcal_g_2p'][msks['2p'], 1])
    g2_2m = np.mean(d['mcal_g_2m'][msks['2m'], 1])
    R11 = (g1_1p - g1_1m) / 2 / 0.01
    R22 = (g2_2p - g2_2m) / 2 / 0.01

    g1 = np.mean(d['mcal_g_noshear'][msks['noshear'], 0])
    g2 = np.mean(d['mcal_g_noshear'][msks['noshear'], 1])

    return g1, g2, R11, R22


def _measure_R_seed(d, seed, s2n_cut=None, size_cut=None):
    n_obj = d.shape[0]
    rng = np.random.RandomState(seed=seed)
    inds = rng.choice(n_obj, size=n_obj, replace=True)
    g1, g2, R11, R22 = _measure_R(d[inds], s2n_cut=s2n_cut, size_cut=size_cut)
    return (g1, g2, R11, R22)


fnames = sys.argv[1:]
swap = False
neg = False
fnames_to_read = []
for fname in fnames:
    if fname == "--swap":
        swap = True
    elif fname == '--neg':
        neg = True
    else:
        fnames_to_read.append(fname)

d = []
for fname in fnames_to_read:
    print("reading file:", fname, flush=True, file=sys.stderr)
    _d = fitsio.read(fname)
    d.extend(list(_d))
d = np.array(d, dtype=_d.dtype)
n_obj = len(d)

s2n_cut = 20.0 
size_cut = 0.5

true_shear = 0.02
if neg:
    true_shear *= (-1.0)

n_boot = 100

marr = []
carr = []
rng = np.random.RandomState(seed=10)
seeds = rng.randint(1, 2**31, size=n_boot)

jobs = [
    delayed(_measure_R_seed)(d, seed, s2n_cut=s2n_cut, size_cut=size_cut)
    for seed in seeds
]
res = Parallel(n_jobs=-1, verbose=0)(jobs)

for r in tqdm.tqdm(res):
    g1, g2, R11, R22 = r
    if swap:
        marr.append(g2/R22 / true_shear - 1.0)
        carr.append(g1/R11)
    else:
        marr.append(g1/R11 / true_shear - 1.0)
        carr.append(g2/R22)

g1, g2, R11, R22 = _measure_R(d, s2n_cut=s2n_cut, size_cut=size_cut)
if swap:
    m = g2/R22 / true_shear - 1.0
    c = g1/R11
else:
    m = g1/R11 / true_shear - 1.0
    c = g2/R22
m_err = np.std(marr)
c_err = np.std(carr)

print("m: %f +/- %f" % (m, m_err))
print("c: %f +/- %f" % (c, c_err))