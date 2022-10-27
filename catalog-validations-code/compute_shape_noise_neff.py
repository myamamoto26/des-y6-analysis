import fitsio as fio
import numpy as np
import os,sys
import pickle

# IN Y3, sigma_e/sqrt(neff*A) = 0.00171474466

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
    # w = mdet_input_flat['w']
    s_all = mdet_input_flat['R_all']
    e1 = mdet_input_flat['g1']
    e2 = mdet_input_flat['g2']
    e = np.sqrt(e1**2 + e2**2)
    g1 = mdet_input_flat['g1']/s_all
    g2 = mdet_input_flat['g2']/s_all
    e1_cov = mdet_input_flat['g1_cov']
    e2_cov = mdet_input_flat['g2_cov']
    vare2 = (e1/e)**2 * e1_cov + (e2/e)**2 * e2_cov
    g1_cov = mdet_input_flat['g1_cov']/(s_all**2)
    g2_cov = mdet_input_flat['g2_cov']/(s_all**2)
    w = 1/(0.07**2 + vare2)
    mean_e1 = np.average(e1, weights=w)
    mean_e2 = np.average(e2, weights=w)

    sum_ws = np.sum(w * s_all)
    sum_w = np.sum(w)
    sum_w2 = np.sum(w**2)
    sum_w2s2 = np.sum(w**2 * s_all**2)
    if version == 'c13':
        sigmae2 = 0.5 * (np.sum(w**2 * (e1**2 + e2**2 - vare2))/sum_w2s2)
        sigma_e = np.sqrt(sigmae2)

        neff = (sigmae2/A) * (sum_ws**2 / np.sum(w**2 * (s_all**2 * sigmae2 + (vare2 / 2))))
        Neff = sum_w**2 / (sum_w2)
    elif version == 'h12':
        sigmae2 = 0.5 * (np.sum(w**2 * (e1 - mean_e1)**2)/sum_ws**2 + np.sum(w**2 * (e2 - mean_e2)**2)/sum_ws**2) * (sum_w**2 / sum_w2)
        sigma_e = np.sqrt(sigmae2)
        neff = sum_w**2 / (A*sum_w2)
        Neff = sum_w**2 / (sum_w2)

    sigma_gamma = sigma_e/np.sqrt(Neff)
    c1 = np.mean(g1)
    c2 = np.mean(g2)
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