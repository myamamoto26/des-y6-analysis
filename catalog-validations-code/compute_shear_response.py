import fitsio as fio
import numpy as np
import os, sys
from tqdm import tqdm
import glob
from des_y6utils import mdet
import pickle

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

def _find_shear_weight(d, wgt_dict, mdet_mom, snmin, snmax, sizemin, sizemax, steps):
    
    if wgt_dict is None:
        weights = np.ones(len(d))
        return weights

    shear_wgt = wgt_dict['weight']
    shear_response = wgt_dict['response']
    indexx, indexy = assign_loggrid(d[mdet_mom+'_s2n'], d[mdet_mom+'_T_ratio'], snmin, snmax, steps, sizemin, sizemax, steps)
    weights = np.array([shear_wgt[x, y] for x, y in zip(indexx, indexy)])
    # response = np.array([shear_response[x, y] for x, y in zip(indexx, indexy)])
    
    return weights

def _accum_shear_per_tile(res, mdet_step, g1, g2, weight):

    """
    Returns the dictionary of the accumulated shear (sum of individual shear).

    Parameters
    ----------
    res: A dictionary in which accumulated sums of shear are stored
    mdet_step: An array of metadetection steps (noshear, 1p, 1m, 2p, 2m) for each object in metadetection catalog
    g1: An array of the measured shapes (e1) for each object in metadetection catalog
    g2: An array of the measured shapes (e2) for each object in metadetection catalog
    weight: Weight on each galaxy. 
    """
    for step in ['noshear', '1p', '1m', '2p', '2m']:
        msk_s = (mdet_step == step)
        np.add.at(
            res[step], 
            (0, 0), 
            np.sum(weight[msk_s] * g1[msk_s]),
        )
        np.add.at(
            res[step], 
            (0, 1), 
            np.sum(weight[msk_s] * g2[msk_s]),
        )
        np.add.at(
            res["num_" + step], 
            (0, 0), 
            np.sum(weight[msk_s]),
        )
        np.add.at(
            res["num_" + step], 
            (0, 1), 
            np.sum(weight[msk_s]),
        )
    return res


def compute_response_over_catalogs(mdet_input_filepaths, response_output_filepath, wgt_filepath, mdet_mom, mdet_cuts, weight_scheme):

    """
    Returns the diagonal part of the shear response R11, R22 from the metadetection catalogs over all the tiles.

    Parameters
    ----------
    mdet_input_filepaths: The file path to the directory in which the input metadetection catalogs exist
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3

    response_output_filepath: The file path where the output text file is written
    Example) /global/cscratch1/sd/myamamot/metadetect/shear_response_v3.txt

    wgt_filepath: The file path where the shear weights are written

    mdet_mom: which measurement do we want to make cuts on
    Example) wmom

    mdet_cuts: which version of the cuts

    weight_scheme: which weighting scheme do you want to use (e.g., s2n_size, shape_error, uniform)
    """

    filenames = sorted(glob.glob(mdet_input_filepaths))
    binnum = 1
    res = {'noshear': np.zeros((binnum, 2)), 'num_noshear': np.zeros((binnum, 2)), 
           '1p': np.zeros((binnum, 2)), 'num_1p': np.zeros((binnum, 2)), 
           '1m': np.zeros((binnum, 2)), 'num_1m': np.zeros((binnum, 2)),
           '2p': np.zeros((binnum, 2)), 'num_2p': np.zeros((binnum, 2)),
           '2m': np.zeros((binnum, 2)), 'num_2m': np.zeros((binnum, 2))}
    bin = 0
    for fname in tqdm(filenames):
        fp = os.path.join(mdet_input_filepaths, fname)
        if os.path.exists(fp):
            d = fio.read(fp)
            msk = mdet.make_mdet_cuts(d, mdet_cuts) 
        #     msk = mdet._make_mdet_cuts_wmom(
        #     d,
        #     min_s2n=5.0,
        #     min_t_ratio=1.1,
        #     n_terr=0.0,
        #     max_mfrac=0.1,
        #     max_s2n=np.inf,
        # )
            d = d[msk]
        else:
            continue
        
        if weight_scheme == 'uniform':
            weight = np.ones(len(d['mdet_step']))
        elif weight_scheme == 'shape_error':
            weight = 1/(0.17**2 + 0.5*(d[mdet_mom+'_g_cov_1_1'] + d[mdet_mom+'_g_cov_2_2']))
        elif weight_scheme == 's2n_sizer':
            with open(wgt_filepath, 'rb') as handle:
                wgt_dict = pickle.load(handle)
            weight = _find_shear_weight(d, wgt_dict, mdet_mom, 10, 300, 0.5, 5.0, 20)
        res = _accum_shear_per_tile(res, d['mdet_step'], d[mdet_mom+'_g_1'], d[mdet_mom+'_g_2'], weight)
    
    g1 = res['noshear'][0][0] / res['num_noshear'][0][0]
    g1p = res['1p'][0][0] / res['num_1p'][0][0]
    g1m = res['1m'][0][0] / res['num_1m'][0][0]
    R11 = (g1p - g1m) / 2 / 0.01

    g2 = res['noshear'][0][1] / res['num_noshear'][0][1]
    g2p = res['2p'][0][1] / res['num_2p'][0][1]
    g2m = res['2m'][0][1] / res['num_2m'][0][1]
    R22 = (g2p - g2m) / 2 / 0.01

    f_response = open(response_output_filepath, 'w')
    f_response.write(str(R11))
    f_response.write('\n')
    f_response.write(str(R22))
    
    return R11, R22

def main(argv):
    
    mdet_input_filepaths = sys.argv[1]
    response_output_filepath = sys.argv[2]
    weight_filepath = sys.argv[3]
    mdet_mom = sys.argv[4]
    mdet_cuts = int(sys.argv[5])
    weight_scheme = sys.argv[6]

    compute_response_over_catalogs(mdet_input_filepaths, response_output_filepath, weight_filepath, mdet_mom, mdet_cuts, weight_scheme)

if __name__ == "__main__":
    main(sys.argv)