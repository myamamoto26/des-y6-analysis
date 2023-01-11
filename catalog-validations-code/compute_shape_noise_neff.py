import fitsio as fio
import numpy as np
import os,sys
import pickle

# IN Y3, sigma_e/sqrt(neff*A) = 0.00171474466
def _compute_y3_values():
    import h5py


    #load catalog
    path = '/project/projectdirs/des/www/y3_cats/Y3_mastercat___UNBLIND___final_v1.1_12_22_20.h5'
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

def _compute_shape_noise(mdet_input_flat, version='c13'):

    """
    Computes shape noise and effective number density of the shape catalog using two different definitions. 
    This follows https://github.com/des-science/2pt_pipeline/blob/y3kp/pipeline/nofz.py#L667

    Parameters
    ----------
    mdet_input_flat: The input filepath for the flat catalog created in make_flat_catalog.py
    Example) 

    version: which shape noise and effective number density defitions do you want to use (c13 or h12)
    """

    A = 4912 * 60 * 60 # survey area

    # For Y3 catalog
    if version == 'y3':
        w = mdet_input_flat['w']
        s = _compute_y3_values()

    # Y6
    w = mdet_input_flat['w']
    s = mdet_input_flat['R_all']
    # w = 1/(0.07**2 + 0.5*(mdet_input_flat['g1_cov'] + mdet_input_flat['g2_cov']))
    # f_response = open('/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/weighted_shear_response_v2_1.txt', 'r')
    # R11, R22 = f_response.read().split('\n')
    # R = (float(R11) + float(R22))/2
    # s = np.full(len(w), R)

    # Common
    e1 = mdet_input_flat['g1']
    e2 = mdet_input_flat['g2']
    e = np.sqrt(e1**2 + e2**2)
    g1 = mdet_input_flat['g1']/s
    g2 = mdet_input_flat['g2']/s
    g = np.sqrt(g1**2 + g2**2)
    # e1_cov = mdet_input_flat['g1_cov']
    # e2_cov = mdet_input_flat['g2_cov']
    # vare2 = (e1/e)**2 * e1_cov + (e2/e)**2 * e2_cov
    # g1_cov = mdet_input_flat['g1_cov']/(s**2)
    # g2_cov = mdet_input_flat['g2_cov']/(s**2)
    mean_e1 = np.average(e1, weights=w)
    mean_e2 = np.average(e2, weights=w)

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
        sigma_e = np.sqrt(sigmae2)
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

    sigma_gamma = sigma_e/np.sqrt(Neff)
    weighted_c1 = np.average(g1, weights=w)
    weighted_c2 = np.average(g2, weights=w)

    print(sigma_e, neff, Neff, sigma_gamma, weighted_c1, weighted_c2)
    return sigma_e, neff, Neff, sigma_gamma, weighted_c1, weighted_c2

def main(argv):

    mdet_input_flat = sys.argv[1]
    neff_version = sys.argv[2]
    # output_csv_filepath = sys.argv[3]
    d = fio.read(mdet_input_flat)

    sigma_e, neff, Neff, sigma_gamma, c1, c2 = _compute_shape_noise(d, version=neff_version)
    quant = np.array([sigma_e, neff, Neff, sigma_gamma, c1, c2])
    # np.savetxt(os.path.join(output_csv_filepath, neff_version+'_shape_noise_neff.csv'), quant, delimiter=',', header="sigma_e,neff,sigma_gamma,c1,c2", comments="")


if __name__ == "__main__":
    main(sys.argv)