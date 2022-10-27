import fitsio as fio
import numpy as np
import os, sys
from tqdm import tqdm
import glob
from des_y6utils import mdet

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
        msk_s = np.where(mdet_step == step)[0]
        
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


def compute_response_over_catalogs(mdet_input_filepaths, response_output_filepath, mdet_mom, mdet_cuts):

    """
    Returns the diagonal part of the shear response R11, R22 from the metadetection catalogs over all the tiles.

    Parameters
    ----------
    mdet_input_filepaths: The file path to the directory in which the input metadetection catalogs exist
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3

    response_output_filepath: The file path where the output text file is written
    Example) /global/cscratch1/sd/myamamot/metadetect/shear_response_v3.txt

    mdet_mom: which measurement do we want to make cuts on
    Example) wmom

    mdet_cuts: which version of the cuts
    """

    # f = open(mdet_tilename_filepath, 'r')
    # fs = f.read().split('\n')[:-1]
    # filenames = [fname.split('/')[-1] for fname in fs]
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
            d = d[msk]
        else:
            continue
        # weight = np.ones(len(d['mdet_step']))
        weight = 1/(0.07**2 + 0.5 * (d[mdet_mom+'_g_cov_1_1'] + d[mdet_mom+'_g_cov_2_2']))
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
    mdet_mom = sys.argv[3]
    mdet_cuts = int(sys.argv[4])

    compute_response_over_catalogs(mdet_input_filepaths, response_output_filepath, mdet_mom, mdet_cuts)

if __name__ == "__main__":
    main(sys.argv)