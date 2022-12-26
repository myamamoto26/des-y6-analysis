import proplot as pplt
import fitsio
import numpy as np
import des_y6utils.mdet
import ngmix
import scipy.interpolate
import glob
from tqdm import tqdm
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import joblib
from sklearn_som.som import SOM
from des_y6utils.mdet import _compute_asinh_mags
import os

class NonBrokenSOM(SOM):
    def fit_predict(self, X, y=None, **kwargs):
        return super().fit_predict(X, **kwargs)
    
    def fit(self, X, y=None, **kwargs):
        if "epochs" in kwargs:
            epochs = kwargs.pop("epochs")
            return super().fit(X, epochs=epochs, **kwargs)
        else:
            return super().fit(X, **kwargs)

def _compute_r(d, msk=None, w=None):
    if msk is None:
        msk = np.ones_like(d["wmom_g_1"]).astype(bool)
        
    if w is None:
        w = np.ones_like(d["wmom_g_1"])
        
    msk_1p = msk & (d["mdet_step"] == "1p")
    nrm = np.sum(w[msk_1p])    
    e1_1p = np.sum(w[msk_1p] * d["wmom_g_1"][msk_1p])/nrm
    e2_1p = np.sum(w[msk_1p] * d["wmom_g_2"][msk_1p])/nrm

    msk_1m = msk & (d["mdet_step"] == "1m")
    nrm = np.sum(w[msk_1m])    
    e1_1m = np.sum(w[msk_1m] * d["wmom_g_1"][msk_1m])/nrm
    e2_1m = np.sum(w[msk_1m] * d["wmom_g_2"][msk_1m])/nrm

    msk_2p = msk & (d["mdet_step"] == "2p")
    nrm = np.sum(w[msk_2p])    
    e1_2p = np.sum(w[msk_2p] * d["wmom_g_1"][msk_2p])/nrm
    e2_2p = np.sum(w[msk_2p] * d["wmom_g_2"][msk_2p])/nrm

    msk_2m = msk & (d["mdet_step"] == "2m")
    nrm = np.sum(w[msk_2m])    
    e1_2m = np.sum(w[msk_2m] * d["wmom_g_1"][msk_2m])/nrm
    e2_2m = np.sum(w[msk_2m] * d["wmom_g_2"][msk_2m])/nrm
    
    return (
        (e1_1p - e1_1m)/0.02 + (e2_2p - e2_2m)/0.02
    )/2.0

def sn_h12(e1, e2, r, w=None):
    if w is None:
        w = np.ones_like(e1)
    
    mean_e1 = np.average(e1, weights=w)
    mean_e2 = np.average(e2, weights=w)

    sum_ws = np.sum(w * r)
    sum_w = np.sum(w)
    sum_w2 = np.sum(w**2)

    sigmae2 = 0.5 * (np.sum(w**2 * (e1 - mean_e1)**2)/sum_ws**2 + np.sum(w**2 * (e2 - mean_e2)**2)/sum_ws**2) * (sum_w**2 / sum_w2)
    return np.sqrt(sigmae2)

def neff_h12(e, w=None):
    if w is None:
        w = np.ones_like(e)
    return np.sum(w)**2/np.sum(w**2)

def prec_h12(e1, e2, r, w=None):
    if w is None:
        w = np.ones_like(e1)

    return sn_h12(e1, e2, r, w=w)/np.sqrt(neff_h12(e1, w=w))

def _comp_stats(d, r, w):
    # R = _compute_r(d, msk=msk, w=w)
    _msk = (d["mdet_step"] == "noshear")
    prec_nrm = 0.07/r/np.sqrt(0.3 * d.shape[0] / 5)

    return (
        np.sum(_msk),
        sn_h12(d["g1"][_msk],  d["g2"][_msk], r, w=w[_msk]),
        neff_h12(d["g1"][_msk], w=w[_msk]),
        prec_nrm,
        prec_h12(d["g1"][_msk], d["g2"][_msk], r, w=w[_msk]),
    ) 

def _compute_som_feats(d, msk):
    """
    Returns features you want SOM to use
    """
    mag_g = _compute_asinh_mags(d["pgauss_band_flux_g"][msk], 0)
    mag_r = _compute_asinh_mags(d["pgauss_band_flux_r"][msk], 1)
    mag_i = _compute_asinh_mags(d["pgauss_band_flux_i"][msk], 2)
    mag_z = _compute_asinh_mags(d["pgauss_band_flux_z"][msk], 3)
    gmr = mag_g - mag_r
    rmi = mag_r - mag_i
    imz = mag_i - mag_z

    return np.vstack([
        np.log10(d["wmom_s2n"][msk]),
        d["wmom_T_ratio"][msk],
        d["pgauss_T"][msk],
        d["mfrac"][msk],
        (d["wmom_g_cov_1_1"][msk] + d["wmom_g_cov_2_2"][msk])/2.0,
        mag_r,
        gmr,
        rmi,
        imz,
    ]).T

def _compute_som_inds(d, msk, est):
    """
    Returns SOM index for each object. Doing it in chunks is easier. 
    """
    X = _compute_som_feats(d, msk)
    
    somind = []
    loc = 0
    chunksize = 1000
    if X.shape[0]%chunksize == 0:
        nchunks = X.shape[0]//chunksize
    else:
        nchunks = X.shape[0]//chunksize + 1

    for chunk in range(nchunks):
        mloc = loc + chunksize
        if mloc > X.shape[0]:
            mloc = X.shape[0]
        somind.append(
            est.predict(X[loc:mloc, :])
        )
        loc += chunksize

    return np.hstack(somind)

def _accumulate_vals(d, evals, msk, somind):

    mag_g = _compute_asinh_mags(d["pgauss_band_flux_g"][msk], 0)
    mag_r = _compute_asinh_mags(d["pgauss_band_flux_r"][msk], 1)
    mag_i = _compute_asinh_mags(d["pgauss_band_flux_i"][msk], 2)
    mag_z = _compute_asinh_mags(d["pgauss_band_flux_z"][msk], 3)
    gmr = mag_g - mag_r
    rmi = mag_r - mag_i
    imz = mag_i - mag_z
    
    np.add.at(evals['n'], somind, np.ones_like(somind))
    np.add.at(evals['e1'], somind, d["wmom_g_1"][msk])
    np.add.at(evals['e2'], somind, d["wmom_g_2"][msk])
    np.add.at(evals['e1_2'], somind, d["wmom_g_1"][msk]**2)
    np.add.at(evals['e2_2'], somind, d["wmom_g_2"][msk]**2)
    np.add.at(evals['s2n'], somind, np.log10(d["wmom_s2n"][msk]))
    np.add.at(evals['tr'], somind, d["wmom_T_ratio"][msk])
    np.add.at(evals['gmr'], somind, gmr)
    np.add.at(evals['rmi'], somind, rmi)
    np.add.at(evals['imz'], somind, imz)
    np.add.at(evals['pt'], somind, d["pgauss_T"][msk])
    np.add.at(evals['mf'], somind, d["mfrac"][msk])
    np.add.at(evals['magr'], somind, mag_r)

    return evals

def _accumulate_shear(d, evals, est):

    for key in ["1p", "1m", "2p", "2m"]:
        msk = d["mdet_step"] == key

        somind = _compute_som_inds(d, msk, est)
        
        np.add.at(evals['num_'+key], somind, np.ones_like(somind))
        if key in ['1p', '1m']:
            np.add.at(evals[key], somind, d["wmom_g_1"][msk])
        elif key in ['2p', '2m']:
            np.add.at(evals[key], somind, d["wmom_g_2"][msk])

    return evals

def _accum_weighted_shear(d, wdict, somind, wsom_grid):

    for key in ["1p", "1m", "2p", "2m"]:
        msk = d["mdet_step"] == key
        
        wsom = wsom_grid[somind[msk]]
        np.add.at(wdict['num_'+key], somind[msk], wsom*np.ones_like(somind[msk]))
        if key in ['1p', '1m']:
            np.add.at(wdict[key], somind[msk], wsom*d["g1"][msk])
        elif key in ['2p', '2m']:
            np.add.at(wdict[key], somind[msk], wsom*d["g2"][msk])

    return wdict


def main(argv):

    if not os.path.exists('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/som_weight_test/som_estimators_25pc.pkl'):
        if os.path.exists('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/som_weight_test/som_training_files.npy'):
            training_files = np.load('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/som_weight_test/som_training_files.npy')
        else:
            mdet_files = glob.glob('/global/cfs/cdirs/des/y6-shear-catalogs/metadetection_patches_v1_blinded/patch-*.fits')
            np.random.seed(314)
            training_files = np.random.choice(mdet_files, size=50, replace=False)
            np.save('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/som_weight_test/som_training_files.npy', training_files)
        # m = 65 #len(mdet_files)
        for i,f in tqdm(enumerate(training_files)):
            d = fitsio.read(f)

            msk_all = des_y6utils.mdet._make_mdet_cuts_wmom(
                d,
                min_s2n=5.0,
                min_t_ratio=1.1,
                n_terr=0.0,
                max_mfrac=0.1,
                max_s2n=np.inf,
            )

            d = d[msk_all]

            msk_feat = d["mdet_step"] == "noshear"
            # Get features for a training
            X = _compute_som_feats(d, msk_feat)
            if i == 0:
                est = make_pipeline(MinMaxScaler(), NonBrokenSOM(m=10, n=10, dim=X.shape[1]))
            est.fit(X, nonbrokensom__epochs=1)

        som_shape = (est.steps[-1][1].m, est.steps[-1][1].n)
        nsom = est.steps[-1][1].m * est.steps[-1][1].n 
        print(som_shape, nsom)
        joblib.dump(est, '/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/som_weight_test/som_estimators_25pc.pkl')
        print('...done training...')
    else:
        training_files = np.load('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/som_weight_test/som_training_files.npy')
        print('reading in trained estimators...')
        est = joblib.load('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/som_weight_test/som_estimators_25pc.pkl')
        nsom = est.steps[-1][1].m * est.steps[-1][1].n 


    # Upon finising the training, get SOM indices for each object. 
    evals = {'n': np.zeros(nsom), 'e1': np.zeros(nsom), 'e2': np.zeros(nsom), 'e1_2': np.zeros(nsom), 'e2_2': np.zeros(nsom), 's2n': np.zeros(nsom), 'tr': np.zeros  (nsom), 'gmr': np.zeros(nsom), 'rmi': np.zeros(nsom), 'imz': np.zeros(nsom), 'pt': np.zeros(nsom), 'mf': np.zeros(nsom), 'magr': np.zeros(nsom)}
    evals_shear = {'1p': np.zeros(nsom), 'num_1p': np.zeros(nsom), 
                '1m': np.zeros(nsom), 'num_1m': np.zeros(nsom),
                '2p': np.zeros(nsom), 'num_2p': np.zeros(nsom),
                '2m': np.zeros(nsom), 'num_2m': np.zeros(nsom)}
    somind_dict = {}
    res_all = []
    for i,f in tqdm(enumerate(training_files)):
        d = fitsio.read(f)
        patch_num = f.split('/')[-1][6:10]

        msk_all = des_y6utils.mdet._make_mdet_cuts_wmom(
            d,
            min_s2n=5.0,
            min_t_ratio=1.1,
            n_terr=0.0,
            max_mfrac=0.1,
            max_s2n=np.inf,
        )

        somind = _compute_som_inds(d, msk_all, est)
        d = d[msk_all]
        res = np.zeros(len(d), dtype=[('mdet_step', object), ('g1', 'f8'), ('g2', 'f8'), ('som_ind', 'i8'), ('w', 'f8'), ('s2n', 'f8'), ('size_ratio', 'f8')])
        res['mdet_step'] = d['mdet_step']
        res['g1'] = d['wmom_g_1']
        res['g2'] = d['wmom_g_2']
        res['s2n'] = d['wmom_s2n']
        res['size_ratio'] = d['wmom_T_ratio']
        res['som_ind'] = somind
        res_all.append(res)

        msk_ = d["mdet_step"] == "noshear"
        somind_noshear = somind[msk_]
        if patch_num not in somind_dict.keys():
            somind_dict[patch_num] = somind
        # accumulate other than shear
        evals = _accumulate_vals(d, evals, msk_, somind_noshear)
        # accumulate shear (used to compute shear response later.)
        evals_shear = _accumulate_shear(d, evals_shear, est)

    cat = np.concatenate(res_all)
    evals.update(evals_shear)
    with open('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/som_weight_test/som_weight_25pc.pickle', 'wb') as handle:
        pickle.dump(evals, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    compute_sensitivity = True
    if compute_sensitivity:
        # get weights from raw shear
        R11 = (evals['1p']/evals['num_1p'] - evals['1m']/evals['num_1m'])/0.02
        R22 = (evals['2p']/evals['num_2p'] - evals['2m']/evals['num_2m'])/0.02
        R = (R11+R22)/2
        sn = (np.sqrt(evals['e1_2']/evals['n'] - (evals['e1']/evals['n'])**2) + np.sqrt(evals['e2_2']/evals['n'] - (evals['e2']/evals['n'])**2))/2
        wsom_grid = R**2/sn**2

        # get global weighted shear response
        weighted_shear = {'1p': np.zeros(nsom), 'num_1p': np.zeros(nsom), 
                          '1m': np.zeros(nsom), 'num_1m': np.zeros(nsom),
                          '2p': np.zeros(nsom), 'num_2p': np.zeros(nsom),
                          '2m': np.zeros(nsom), 'num_2m': np.zeros(nsom)}
        w = []
        for i, patch in tqdm(enumerate(somind_dict.keys())):
            n = len(somind_dict[patch])
            sind = somind_dict[patch]
            d_ = res_all[i]
            weighted_shear = _accum_weighted_shear(d_, weighted_shear, sind, wsom_grid)
            w.append(wsom_grid[sind])
        w = np.concatenate(w)

        R11 = (np.sum(weighted_shear['1p'])/np.sum(weighted_shear['num_1p']) - np.sum(weighted_shear['1m'])/np.sum(weighted_shear['num_1m']))/0.02
        R22 = (np.sum(weighted_shear['2p'])/np.sum(weighted_shear['num_2p']) - np.sum(weighted_shear['2m'])/np.sum(weighted_shear['num_2m']))/0.02
        R_global = (R11+R22)/2
        print(R11, R22, R_global)
        stats = _comp_stats(cat, R_global, w)
        print(stats)

        msk_cat_noshear = cat['mdet_step'] == 'noshear'
        cat = cat[msk_cat_noshear]
        cat['w'] = w[msk_cat_noshear]
        fitsio.write('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/som_weight_test/som_25pc_catalog.fits', cat)

    return None

if __name__ == "__main__":
    main(sys.argv)

