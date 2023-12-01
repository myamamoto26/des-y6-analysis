import fitsio as fio
import numpy as np
import os,sys
import pickle

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

def _find_shear_weight(d, wgt_dict, snmin, snmax, sizemin, sizemax, steps, mdet_mom):

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
    indexx, indexy = assign_loggrid(d[mdet_mom+'_s2n'], d[mdet_mom+'_T_ratio'], snmin, snmax, steps, sizemin, sizemax, steps)
    weights = np.array([shear_wgt[x, y] for x, y in zip(indexx, indexy)])
    
    return weights

def read_mdet_h5(datafile, keys, response=False, subtract_mean_shear=False):

    def _get_shear_weights(dat):
        shape_err = False
        if shape_err:
            return 1/(0.22**2 + 0.5*(np.array(dat['gauss_g_cov_1_1']) + np.array(dat['gauss_g_cov_2_2'])))
        else:
            with open(os.path.join('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b/inverse_variance_weight_v5b_s2n_10-1000_Tratio_0.5-5.pickle'), 'rb') as handle:
                wgt_dict = pickle.load(handle)
                snmin = wgt_dict['xedges'][0]
                snmax = wgt_dict['xedges'][-1]
                sizemin = wgt_dict['yedges'][0]
                sizemax = wgt_dict['yedges'][-1]
                steps = len(wgt_dict['xedges'])-1
            shear_wgt = _find_shear_weight(dat, wgt_dict, snmin, snmax, sizemin, sizemax, steps, 'gauss')
            return shear_wgt
        
    def _wmean(q,w):
        return np.sum(q*w)/np.sum(w)
    
    import h5py as h5
    f = h5.File(datafile, 'r')
    d = f.get('/mdet/noshear')
    nrows = len(np.array( d['ra'] ))
    formats = []
    for key in keys:
        formats.append('f4')
    data = np.recarray(shape=(nrows,), formats=formats, names=keys)
    for key in keys:  
        if key == 'R':
            continue
        elif key == 'w':
            # data['w'] = _get_shear_weights(d)
            data['w'] = _get_shear_weights(d)
        elif key in ('g1', 'g2'):
            data[key] = np.array(d['gauss_'+key[0]+'_'+key[1]])
        elif key in ('g1_cov', 'g2_cov'):
            data[key] = np.array(d['gauss_'+key[0]+'_cov_'+key[1]+'_'+key[1]])
        else:
            data[key] = np.array(d[key])
    print('made recarray with hdf5 file')
    
    # response correction
    if response:
        d_2p = f.get('/mdet/2p')
        d_1p = f.get('/mdet/1p')
        d_2m = f.get('/mdet/2m')
        d_1m = f.get('/mdet/1m')
        # compute response with weights
        g1p = _wmean(np.array(d_1p["gauss_g_1"]), _get_shear_weights(d_1p))                                     
        g1m = _wmean(np.array(d_1m["gauss_g_1"]), _get_shear_weights(d_1m))
        R11 = (g1p - g1m) / 0.02

        g2p = _wmean(np.array(d_2p["gauss_g_2"]), _get_shear_weights(d_2p))
        g2m = _wmean(np.array(d_2m["gauss_g_2"]), _get_shear_weights(d_2m))
        R22 = (g2p - g2m) / 0.02

        R = (R11 + R22)/2.
        data['R'] = np.ones(len(data['g1'])) * R
        print('weighted shear response is ', R)
        # data['g1'] /= R
        # data['g2'] /= R

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

# IN Y3, sigma_e/sqrt(neff*A) = 0.00171474466
def _compute_y3_values():
    import h5py


    #load catalog
    path = '/global/cfs/cdirs/des/www/y3_cats/Y3_mastercat___UNBLIND___final_v1.1_12_22_20.h5'
    m = h5py.File(path,'r')
    # load mask
    mask = np.array(m.get('/index/select'))

    # Load weights
    w = np.array(m.get('/catalog/metacal/unsheared/weight'))
    w_1p = np.array(m.get('/catalog/metacal/sheared_1p/weight'))
    w_2p = np.array(m.get('/catalog/metacal/sheared_2p/weight'))
    w_1m = np.array(m.get('/catalog/metacal/sheared_1m/weight'))
    w_2m = np.array(m.get('/catalog/metacal/sheared_2m/weight'))

    # load sheared ellipticities
    e1_1m = np.array(m.get('/catalog/metacal/sheared_1m/e_1'))
    e1_1p = np.array(m.get('/catalog/metacal/sheared_1p/e_1'))
    e2_2m = np.array(m.get('/catalog/metacal/sheared_2m/e_2'))
    e2_2p = np.array(m.get('/catalog/metacal/sheared_2p/e_2'))

    # compute responses
    dg = 0.01
    m1   = (np.average(e1_1p[mask],weights = w_1p[mask]) - np.average(e1_1m[mask],weights = w_1m[mask])) / (2.*dg)
    m2   = (np.average(e2_2p[mask],weights = w_2p[mask]) - np.average(e2_2m[mask],weights = w_2m[mask])) / (2.*dg)

    print ('rg',m1,m2)

    # compute selection responses
    # load sheared masks
    mask_1m = np.array(m['index']['select_1m'])
    mask_1p = np.array(m['index']['select_1p'])
    mask_2m = np.array(m['index']['select_2m'])
    mask_2p = np.array(m['index']['select_2p'])

    # Load ellipticities
    e1 = np.array(m.get('/catalog/metacal/unsheared/e_1'))
    e2 = np.array(m.get('/catalog/metacal/unsheared/e_2'))

    d_m1   = (np.average(e1[mask_1p],weights = w[mask_1p]) - np.average(e1[mask_1m],weights = w[mask_1m])) / (2.*dg)
    d_m2   = (np.average(e2[mask_2p],weights = w[mask_2p]) - np.average(e2[mask_2m],weights = w[mask_2m])) / (2.*dg)
    print ('rg+rs',m1+d_m1,m2+d_m2)

    # you can also simply use the average responses in the catalogs, but they will be slightly less accurate,
    # because they are defined as R11 =(e1_1p[mask] -e1_1m[mask]) / (2.*dg), whereas ideally you'd want to
    # use the sheared weights as well. The difference is very small, though...
    # Load responses
    R1 = np.array(m.get('/catalog/metacal/unsheared/R11'))
    R2 = np.array(m.get('/catalog/metacal/unsheared/R22'))
    R1 = R1[mask]
    R2 = R2[mask]

    m1_   = np.average(R1,weights = w[mask])
    m2_   = np.average(R2,weights = w[mask])
    print ('rg',m1_,m2_)

    # add responses together
    m1   = (m1+d_m1)*np.ones(len(mask))
    m2   = (m2+d_m2)*np.ones(len(mask))
    s   = (m1+m2)/2. 
    # w = w[mask]

    return s

def _compute_shape_noise(mdet_input_flat, version, method):

    """
    Computes shape noise and effective number density of the shape catalog using two different definitions. 
    This follows https://github.com/des-science/2pt_pipeline/blob/y3kp/pipeline/nofz.py#L667

    Parameters
    ----------
    mdet_input_flat: The input filepath for the flat catalog created in make_flat_catalog.py
        Example) /pscratch/sd/m/myamamot/metadetection_v5_flat.fits

    version: which shape noise and effective number density defitions do you want to use (c13 or h12)
    method: which weight to use
    """

    A = 4435.515784199025 * 60 * 60 # survey area for Y6 (v3 mask)

    # For Y3 catalog
    if version == 'y3':
        w = mdet_input_flat['w']
        s = _compute_y3_values()
    else:
        # Y6
        if method in ('s2n_sizer', 'shape_err'):
            w = mdet_input_flat['w']
            s = mdet_input_flat['R']
        elif method == 'shape_noise':
            w = np.ones(len(mdet_input_flat['w']))
            s = np.ones(len(mdet_input_flat['R']))

    # Common
    e1 = mdet_input_flat['g1']
    e2 = mdet_input_flat['g2']
    e = np.sqrt(e1**2 + e2**2)
    g1 = mdet_input_flat['g1']/s
    g2 = mdet_input_flat['g2']/s
    g = np.sqrt(g1**2 + g2**2)
    e1_cov = mdet_input_flat['g1_cov']
    e2_cov = mdet_input_flat['g2_cov']
    vare2 = (e1/e)**2 * e1_cov + (e2/e)**2 * e2_cov
    mean_e1 = np.average(e1, weights=w)
    mean_e2 = np.average(e2, weights=w)
    mean_g1 = np.average(g1, weights=w)
    mean_g2 = np.average(g2, weights=w)

    sum_ws = np.sum(w * s)
    sum_w = np.sum(w)
    sum_w2 = np.sum(w**2)
    sum_w2s2 = np.sum(w**2 * s**2)
    if version == 'c13':
        sigmae2 = 0.5 * (np.sum(w**2 * (e1**2 + e2**2 - vare2))/sum_w2s2)
        sigma_e = np.sqrt(sigmae2)

        neff = (sigmae2/A) * (sum_ws**2 / np.sum(w**2 * (s**2 * sigmae2 + (vare2 / 2))))
        Neff = sum_w**2 / (sum_w2)
    elif version == 'h12':
        sigmae2 = 0.5 * (np.sum(w**2 * (e1 - mean_e1)**2)/sum_ws**2 + np.sum(w**2 * (e2 - mean_e2)**2)/sum_ws**2) * (sum_w**2 / sum_w2)
        # sigmae2 = 0.5 * (np.sum(w**2 * (e1 - mean_e1)**2)/sum_w**2 + np.sum(w**2 * (e2 - mean_e2)**2)/sum_w**2) * (sum_w**2 / sum_w2)
        sigma_e = np.sqrt(sigmae2)
        # print('shape noise before response correction', sigma_e)
        neff = sum_w**2 / (A*sum_w2)
        Neff = sum_w**2 / (sum_w2)
    elif version == 'y6': # -> this computes the weighted standard deviation rather than the error on the weighted mean for shape noise. Y3 computed error on the weighted mean for sensitivity. 
        mean_g1 = np.average(g1,weights=w)
        mean_g2 = np.average(g2,weights=w)

        sum_we = np.sum(w * (g2 - mean_g2)**2)
        sigmae2 = sum_we / (sum_w - (sum_w2/sum_w))
        sigma_e = np.sqrt(sigmae2)
        neff = sum_w**2 / (A*sum_w2)
        Neff = sum_w**2 / (sum_w2)
    elif version == 'y3':
        # compute sigma_e and n_eff
        a1 = np.sum(w**2 * (e1-mean_e1)**2)
        a2 = np.sum(w**2 * (e2-mean_e2)**2)
        b  = np.sum(w**2)
        c  = np.sum(w * s)
        d  = np.sum(w)

        sigma_e = (np.sqrt( (a1/c**2 + a2/c**2) * (d**2/b) / 2. ) )

        area = 4143.
        a    = np.sum(w)**2
        b    = np.sum(w**2)
        c    = area * 60. * 60.

        neff = ( a/b/c )
        Neff = (a/b)

    sigma_gamma = sigma_e/np.sqrt(Neff)/A # precision per square degree. 
    weighted_c1 = np.average(e1, weights=w)
    weighted_c2 = np.average(e2, weights=w)

    print(sigma_e, neff, Neff, sigma_gamma, weighted_c1, weighted_c2)
    return sigma_e, neff, Neff, sigma_gamma, weighted_c1, weighted_c2

def main(argv):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="the path to input flat catalogs", type=str)
    parser.add_argument("definition", help="the definition of neff and shape noise (h12 or c13)", type=str)
    parser.add_argument("weight", help="the definition of shear weights (s2n_sizer or shape_err)", type=str)
    parser.add_argument("--save_result", help="whether or not save the result to text file", type=bool)
    parser.set_defaults(save_result=False)
    args = parser.parse_args()

    input_file = args.input_file
    neff_version = args.definition
    method = args.weight
    
    keys = ['ra', 'dec', 'g1', 'g2', 'w', 'R', 'g1_cov', 'g2_cov']
    gal_data, mean_shear = read_mdet_h5(input_file, keys, response=True, subtract_mean_shear=False)

    sigma_e, neff, Neff, sigma_gamma, c1, c2 = _compute_shape_noise(gal_data, neff_version, method)
    quant = np.array([sigma_e, neff, Neff, sigma_gamma, c1, c2])
    if args.save_result:
        np.savetxt(os.path.join('/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5/', neff_version+'_shape_noise_neff.csv'), quant, delimiter=',', header="sigma_e,neff,sigma_gamma,c1,c2", comments="")


if __name__ == "__main__":
    main(sys.argv)