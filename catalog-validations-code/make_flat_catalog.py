import numpy as np
import fitsio as fio
import pickle
import glob
from tqdm import tqdm
import os,sys

def _make_nz_for_sample_variance():

    # The function is in prep_for_sample_variance.ipynb. 

    return None

def _make_flat_catalog_with_wgt(shear_wgt_input_filepath, response_input_filepath, mdet_input_filepaths, mdet_flat_output_filepath, mdet_mom):

    """
    Compute the flat shear catalog with the inverse variance shear weight and saves it as a pickle file. 

    Parameters
    ----------
    shear_wgt_input_filepath: The file path where the inverse variance shear weight is stored. This is produced in inverse_weight.py.
    Example) /global/cscratch1/sd/myamamot/metadetect/inverse_variance_weight_v3_Trcut_snmax1000.pickle

    response_input_filepath: The file path where the shear response over all the tiles is stored. 
    Example) /global/cscratch1/sd/myamamot/metadetect/shear_response_v3.txt

    mdet_input_filepaths: The file paths where the input metadetection catalogs exist
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3/*_metadetect-v5_mdetcat_part0000.fits

    mdet_flat_output_filepath: The file path where the output flat catalog is written
    Example) /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3_snmax1000.pkl

    mdet_mom: which measurement do we want to make cuts on
    """

    # This is the result of inverse_weight.py
    with open(shear_wgt_input_filepath, 'rb') as handle:
        res = pickle.load(handle)

    # This is the result of compute_shear_response.py
    f_response = open(response_input_filepath, 'r')
    R11, R22 = f_response.read().split('\n')

    s2n = res['xedges']
    Tratio = res['yedges']
    shear_weight = res['weight']
    shear_response = res['response']

    mdet_f = glob.glob(mdet_input_filepaths)

    def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
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

    def _find_shear_weight(d, snmin, snmax, sizemin, sizemax, steps):
        
        indexx, indexy = assign_loggrid(d['mdet_s2n'], d[mdet_mom+'_T_ratio'], snmin, snmax, steps, sizemin, sizemax, steps)
        weight = [shear_weight[x, y] for x, y in zip(indexx, indexy)]
        response = [shear_response[x, y] for x, y in zip(indexx, indexy)]
        
        return np.array(weight), np.array(response)

    mdet_dict = {0: {}}
    snmin=np.min(s2n)
    snmax=np.max(s2n)
    sizemin=np.min(Tratio)
    sizemax=np.max(Tratio)
    steps=20
    for i, fname in tqdm(enumerate(mdet_f)):
        
        d = fio.read(fname)
        msk_noshear = (d['mdet_step'] == 'noshear')
        
        w_shear, r_shear = _find_shear_weight(d, snmin, snmax, sizemin, sizemax, steps)
        g1 = d[msk_noshear][mdet_mom+'_g_1']/float(R11) # r_shear[msk_noshear]
        g2 = d[msk_noshear][mdet_mom+'_g_2']/float(R22) # r_shear[msk_noshear]

        if i == 0:
            mdet_dict[0]['ra'] = d[msk_noshear]['ra']
            mdet_dict[0]['dec'] = d[msk_noshear]['dec']
            mdet_dict[0]['e1'] = g1
            mdet_dict[0]['e2'] = g2
            mdet_dict[0]['w'] = w_shear[msk_noshear]
            mdet_dict[0][mdet_mom+'_band_flux_g'] = d[msk_noshear][mdet_mom+'_band_flux_g']
            mdet_dict[0][mdet_mom+'_band_flux_r'] = d[msk_noshear][mdet_mom+'_band_flux_r']
            mdet_dict[0][mdet_mom+'_band_flux_i'] = d[msk_noshear][mdet_mom+'_band_flux_i']
            mdet_dict[0][mdet_mom+'_band_flux_z'] = d[msk_noshear][mdet_mom+'_band_flux_z']
            mdet_dict[0][mdet_mom+'_band_flux_err_g'] = d[msk_noshear][mdet_mom+'_band_flux_err_g']
            mdet_dict[0][mdet_mom+'_band_flux_err_r'] = d[msk_noshear][mdet_mom+'_band_flux_err_r']
            mdet_dict[0][mdet_mom+'_band_flux_err_i'] = d[msk_noshear][mdet_mom+'_band_flux_err_i']
            mdet_dict[0][mdet_mom+'_band_flux_err_z'] = d[msk_noshear][mdet_mom+'_band_flux_err_z']
        else:
            mdet_dict[0]['ra'] = np.append(mdet_dict[0]['ra'], d[msk_noshear]['ra'])
            mdet_dict[0]['dec'] = np.append(mdet_dict[0]['dec'], d[msk_noshear]['dec'])
            mdet_dict[0]['e1'] = np.append(mdet_dict[0]['e1'], g1)
            mdet_dict[0]['e2'] = np.append(mdet_dict[0]['e2'], g2)
            mdet_dict[0]['w'] = np.append(mdet_dict[0]['w'], w_shear[msk_noshear])
            mdet_dict[0][mdet_mom+'_band_flux_g'] = np.append(mdet_dict[0][mdet_mom+'_band_flux_g'], d[msk_noshear][mdet_mom+'_band_flux_g'])
            mdet_dict[0][mdet_mom+'_band_flux_r'] = np.append(mdet_dict[0][mdet_mom+'_band_flux_r'], d[msk_noshear][mdet_mom+'_band_flux_r'])
            mdet_dict[0][mdet_mom+'_band_flux_i'] = np.append(mdet_dict[0][mdet_mom+'_band_flux_i'], d[msk_noshear][mdet_mom+'_band_flux_i'])
            mdet_dict[0][mdet_mom+'_band_flux_z'] = np.append(mdet_dict[0][mdet_mom+'_band_flux_z'], d[msk_noshear][mdet_mom+'_band_flux_z'])
            mdet_dict[0][mdet_mom+'_band_flux_err_g'] = np.append(mdet_dict[0][mdet_mom+'_band_flux_err_g'], d[msk_noshear][mdet_mom+'_band_flux_err_g'])
            mdet_dict[0][mdet_mom+'_band_flux_err_r'] = np.append(mdet_dict[0][mdet_mom+'_band_flux_err_r'], d[msk_noshear][mdet_mom+'_band_flux_err_r'])
            mdet_dict[0][mdet_mom+'_band_flux_err_i'] = np.append(mdet_dict[0][mdet_mom+'_band_flux_err_i'], d[msk_noshear][mdet_mom+'_band_flux_err_i'])
            mdet_dict[0][mdet_mom+'_band_flux_err_z'] = np.append(mdet_dict[0][mdet_mom+'_band_flux_err_z'], d[msk_noshear][mdet_mom+'_band_flux_err_z'])
    
    with open(mdet_flat_output_filepath, 'wb') as handle:
        pickle.dump(mdet_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(argv):

    shear_wgt_input_filepath = sys.argv[1]
    response_input_filepath = sys.argv[2]
    mdet_input_filepaths = sys.argv[3]
    mdet_flat_output_filepath = sys.argv[4]
    mdet_mom = sys.argv[5]

    _make_flat_catalog_with_wgt(shear_wgt_input_filepath, response_input_filepath, mdet_input_filepaths, mdet_flat_output_filepath, mdet_mom)

if __name__ == "__main__":
    main(sys.argv)
