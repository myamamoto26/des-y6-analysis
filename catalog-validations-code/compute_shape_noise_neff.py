import fitsio as fio
import numpy as np
import os,sys
import pickle

def _compute_shape_noise(mdet_input_flat, version='c13'):

    """
    Computes shape noise and effective number density of the shape catalog using two different definitions. 

    Parameters
    ----------
    mdet_input_flat: The input filepath for the flat catalog created in make_flat_catalog.py
    Example) 

    version: which shape noise and effective number density defitions do you want to use (c13 or h12)
    """

    A = 5000 # survey area
    shear_weight = mdet_input_flat['w']
    e1 = mdet_input_flat['e1']
    e2 = mdet_input_flat['e2']
    e1_cov = mdet_input_flat['g_cov_1_1']
    e2_cov = mdet_input_flat['g_cov_2_2']

    if version == 'c13':
        w2 = np.sum(shear_weight**2)
        measurement_noise = e1_cov**2 + e2_cov**2
        numer = shear_weight**2 * (e1**2 + e2**2 - measurement_noise)

        shape_noise_squ = 0.5 * (np.sum(numer)/w2)
        sigma_e = np.sqrt(shape_noise_squ)
        neff = (shape_noise_squ * w2 / np.sum(shear_weight**2 * (shape_noise_squ + measurement_noise**2 / 2.)))/A
    elif version == 'h12':
        shape_noise_squ = 0.5 * (np.sum((shear_weight*e1)**2)/w2 + np.sum((shear_weight*e2)**2)/w2) * (w2 / np.sum(shear_weight)**2)
        sigma_e = np.sqrt(shape_noise_squ)
        neff = np.sum(shear_weight)**2 / (A*w2)

    c1 = np.mean(e1)
    c2 = np.mean(e2)

    return sigma_e, neff, c1, c2

def main(argv):

    mdet_input_flat = sys.argv[1]
    neff_version = sys.argv[2]
    output_csv_filepath = sys.argv[3]
    d = fio.read(mdet_input_flat)

    sigma_e, neff, c1, c2 = _compute_shape_noise(d, version=neff_version)
    quant = np.array([sigma_e, neff, c1, c2])
    np.savetxt(os.path.join(output_csv_filepath, neff_version+'_shape_noise_neff.csv'), quant, delimiter=',', header="sigma_e,neff,c1,c2", comments="")


if __name__ == "__main__":
    main(sys.argv)