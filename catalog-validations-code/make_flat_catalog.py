import numpy as np
import fitsio as fio
import pickle
import glob
from tqdm import tqdm
import os,sys

def _make_nz_for_sample_variance():

    # The function is in prep_for_sample_variance.ipynb. 

    return None

def _make_flat_catalog_with_wgt(shear_wgt_input_filepath, response_input_filepath, mdet_input_filepaths, mdet_flat_pickle_output_filepath, mdet_flat_fits_output_filepath, mdet_mom):

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

    mdet_flat_pickle_output_filepath: The file path where the pickle output flat catalog is written
    Example) /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3_snmax1000.pkl

    mdet_flat_fits_output_filepath: The file path where the fits output flat catalog is written
    Example) /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3_snmax1000.fits

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
    count = np.sum(res['count'])

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

    snmin=np.min(s2n)
    snmax=np.max(s2n)
    sizemin=np.min(Tratio)
    sizemax=np.max(Tratio)
    steps=20

    res = np.zeros(count, dtype=[('ra', float), ('dec', float), ('e1', float), ('e2', float), ('w', float), ('g_cov_1_1', float), ('g_cov_2_2', float), ('g_flux', float), ('r_flux', float), ('i_flux', float), ('z_flux', float), ('g_fluxerr', float), ('r_fluxerr', float), ('i_fluxerr', float), ('z_fluxerr', float)])
    print('number of objects', count)
    start = 0
    end = 0
    for fname in tqdm(mdet_f):
        
        d = fio.read(fname)
        additional_msk = ((d[mdet_mom+'_s2n'] < snmax) & (d[mdet_mom+'_T_ratio'] < sizemax))
        msk_noshear = (d['mdet_step'] == 'noshear') & additional_msk
        
        w_shear, r_shear = _find_shear_weight(d, snmin, snmax, sizemin, sizemax, steps)
        g1 = d[msk_noshear][mdet_mom+'_g_1']/float(R11) # r_shear[msk_noshear]
        g2 = d[msk_noshear][mdet_mom+'_g_2']/float(R22) # r_shear[msk_noshear]
        end += len(g1)

        res['ra'][start:end] = d[msk_noshear]['ra']
        res['dec'][start:end] = d[msk_noshear]['dec']
        res['e1'][start:end] = g1
        res['e2'][start:end] = g2
        res['w'][start:end] = w_shear[msk_noshear]
        res['g_cov_1_1'] = d[msk_noshear][mdet_mom+'_g_cov_1_1']
        res['g_cov_2_2'] = d[msk_noshear][mdet_mom+'_g_cov_2_2']
        res['g_flux'] = d[msk_noshear][mdet_mom+'_band_flux_g']
        res['r_flux'] = d[msk_noshear][mdet_mom+'_band_flux_r']
        res['i_flux'] = d[msk_noshear][mdet_mom+'_band_flux_i']
        res['z_flux'] = d[msk_noshear][mdet_mom+'_band_flux_z']
        res['g_fluxerr'] = d[msk_noshear][mdet_mom+'_band_flux_err_g']
        res['r_fluxerr'] = d[msk_noshear][mdet_mom+'_band_flux_err_r']
        res['i_fluxerr'] = d[msk_noshear][mdet_mom+'_band_flux_err_i']
        res['z_fluxerr'] = d[msk_noshear][mdet_mom+'_band_flux_err_z']

        start += len(g1)    
    print('number of final objects', end)

    # cut down zero elements and save to fits file. 
    res = res[res['ra'] != 0]
    fio.write(mdet_flat_fits_output_filepath, res)

    # make res a dict object and save pickle file.
    mdet_dict = {0: {}}
    mdet_dict[0]['ra'] = res['ra']
    mdet_dict[0]['dec'] =res['dec']
    mdet_dict[0]['e1'] = res['e1']
    mdet_dict[0]['e2'] = res['e2']
    mdet_dict[0]['w'] = res['w']
    mdet_dict[0]['g_cov_1_1'] = res['g_cov_1_1']
    mdet_dict[0]['g_cov_2_2'] = res['g_cov_2_2']
    mdet_dict[0]['g_flux'] = res['g_flux']
    mdet_dict[0]['r_flux'] = res['r_flux']
    mdet_dict[0]['i_flux'] = res['i_flux']
    mdet_dict[0]['z_flux'] = res['z_flux']
    mdet_dict[0]['g_fluxerr'] = res['g_fluxerr']
    mdet_dict[0]['r_fluxerr'] = res['r_fluxerr']
    mdet_dict[0]['i_fluxerr'] = res['i_fluxerr']
    mdet_dict[0]['z_fluxerr'] = res['z_fluxerr']

    with open(mdet_flat_pickle_output_filepath, 'wb') as handle:
        pickle.dump(mdet_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(argv):

    shear_wgt_input_filepath = sys.argv[1]
    response_input_filepath = sys.argv[2]
    mdet_input_filepaths = sys.argv[3]
    mdet_flat_pickle_output_filepath = sys.argv[4]
    mdet_flat_fits_output_filepath = sys.argv[5]
    mdet_mom = sys.argv[6]

    _make_flat_catalog_with_wgt(shear_wgt_input_filepath, response_input_filepath, mdet_input_filepaths, mdet_flat_pickle_output_filepath, mdet_flat_fits_output_filepath, mdet_mom)

if __name__ == "__main__":
    main(sys.argv)
