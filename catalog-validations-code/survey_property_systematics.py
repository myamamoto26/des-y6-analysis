from nis import match
import os, sys
from tokenize import group
from tqdm import tqdm
import numpy as np
import fitsio as fio
import matplotlib as mpl
import glob
import pickle

def survey_systematic_maps(fs, mdet_input_filepaths, sample_variance_filepaths, survey_property, method, num_sim, outpath, prop, band, mdet_mom):

    """
    
    Parameters
    ----------
    fs: The input file for the tilenames
    mdet_input_filepaths: The input filepath for the metadetection catalogs
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3

    sample_variance_filepaths: The input filepath for the simulated (PKDGRAV) sample variance catalogs
    Example) /global/cscratch1/sd/myamamot/sample_variance

    survey_property: a list of files that contain the values for survey properties in healpix map
    method: a method of how to compute the survey systematics ('bin' computes the shear response in each bin. 'pixel' copmutes the shear response in each healpixel. This option is very expensive.)
    num_sim: the number of PKDGRAV simulations
    outpath: 
    Example) /global/cscratch1/sd/myamamot/survey_property_maps/airmass

    prop: the systematic map property
    band: the band for which survey systematic is measured
    mdet_mom: which estimator to use
    """

    import healpy as hp
    import time
    import numpy_groupies as npg
    from esutil import stat

    def _compute_shear(mean_shear_output, group_shear_output, group_number_output, binnum, pkdgrav=False):
        
        if pkdgrav:
            for bind in tqdm(range(binnum)):
                g1 = (group_shear_output[0][bind,0]/group_number_output[0][bind,0])
                g2 = (group_shear_output[0][bind,1]/group_number_output[0][bind,1])

                mean_shear_output['g1'][bind] = g1
                mean_shear_output['g2'][bind] = g2
                mean_shear_output['mean_signal'][bind] = hist['mean'][bind]
        else:
            for bind in tqdm(range(binnum)):
                R11 = (group_shear_output[1][bind, 0]/group_number_output[1][bind,0]-group_shear_output[2][bind, 0]/group_number_output[2][bind,0])/0.02
                R22 = (group_shear_output[3][bind, 1]/group_number_output[3][bind,1]-group_shear_output[4][bind, 1]/group_number_output[4][bind,1])/0.02
                g1 = (group_shear_output[0][bind,0]/group_number_output[0][bind,0])/R11
                g2 = (group_shear_output[0][bind,1]/group_number_output[0][bind,1])/R22

                mean_shear_output['g1'][bind] = g1
                mean_shear_output['g2'][bind] = g2
                mean_shear_output['mean_signal'][bind] = hist['mean'][bind]

        return mean_shear_output
    
    def _accum_shear_pixel(d, d_pix, total_shear_output, total_number_output):
   
        for i, step in enumerate(['noshear', '1p', '1m', '2p', '2m']):
            msk_s = np.where(d['mdet_step'] == step)[0]

            t0 = time.time()
            group_e1 = npg.aggregate(d_pix[msk_s], d[msk_s]['mdet_g_1'].astype('float'), func='sum', fill_value=0)
            group_e2 = npg.aggregate(d_pix[msk_s], d[msk_s]['mdet_g_2'].astype('float'), func='sum', fill_value=0)
            group_nume1 =  np.bincount(d_pix[msk_s])
            group_nume2 =  np.bincount(d_pix[msk_s])
            # print('grouping', time.time()-t0)

            t0 = time.time()
            msk_nonzero = (group_e1 != 0)
            index_pixel_e1 = np.where(msk_nonzero)[0]
            msk_nonzero = (group_e2 != 0)
            index_pixel_e2 = np.where(msk_nonzero)[0]
            msk_nonzero = (group_nume1 != 0)
            index_pixel_nume1 = np.where(msk_nonzero)[0]
            msk_nonzero = (group_nume2 != 0)
            index_pixel_nume2 = np.where(msk_nonzero)[0]
            # print('index', time.time()-t0)

            t0 = time.time()
            total_shear_output[i][index_pixel_e1, 0] += group_e1[index_pixel_e1]
            total_shear_output[i][index_pixel_e2, 1] += group_e2[index_pixel_e2]
            total_number_output[i][index_pixel_nume1, 0] += group_nume1[index_pixel_nume1]
            total_number_output[i][index_pixel_nume2, 1] += group_nume2[index_pixel_nume2]
            # print('accumulate', time.time()-t0)

        return total_shear_output, total_number_output

    def _accum_shear_bin(d, d_bin_signal, bin_edges, total_shear_output, total_number_output, mdet_mom):
   
        for i, step in enumerate(['noshear', '1p', '1m', '2p', '2m']):
            msk_s = np.where(d['mdet_step'] == step)[0]
            bin_index = np.digitize(d_bin_signal[msk_s], bin_edges) - 1

            t0 = time.time()
            np.add.at(total_shear_output[i], (bin_index, 0), d[msk_s][mdet_mom+'_g_1']) 
            np.add.at(total_shear_output[i], (bin_index, 1), d[msk_s][mdet_mom+'_g_2']) 
            np.add.at(total_number_output[i], (bin_index, 0), 1)
            np.add.at(total_number_output[i], (bin_index, 1), 1)
            # print('accumulate', time.time()-t0)

        # return total_shear_output, total_number_output

    def _accum_shear_bin_pkdgrav(d, d_bin_signal, bin_edges, total_shear_output, total_number_output):
   
        bin_index = np.digitize(d_bin_signal, bin_edges) - 1

        np.add.at(total_shear_output[0], (bin_index, 0), d['e1']) 
        np.add.at(total_shear_output[0], (bin_index, 1), d['e2']) 
        np.add.at(total_number_output[0], (bin_index, 0), 1)
        np.add.at(total_number_output[0], (bin_index, 1), 1)
            
    syst = fio.read(survey_property)
    healpix = hp.nside2npix(4096)
    pix_signal = {syst[pix]['PIXEL']: syst[pix]['SIGNAL'] for pix in range(len(syst['PIXEL']))}
    if method == 'pixel':
        group_shear_output = [np.zeros((healpix, 2)), np.zeros((healpix, 2)), np.zeros((healpix, 2)), np.zeros((healpix, 2)), np.zeros((healpix, 2))]
        group_number_output = [np.zeros((healpix, 2)), np.zeros((healpix, 2)), np.zeros((healpix, 2)), np.zeros((healpix, 2)), np.zeros((healpix, 2))]
    elif method == 'bin':
        pix_val = np.array(list(pix_signal.values()))
        hist = stat.histogram(pix_val, nperbin=1000000, more=True)
        binnum = len(hist['mean'])
        bin_edges = np.insert(hist['high'], 0, hist['low'][0])
        group_shear_output = [np.zeros((binnum, 2)), np.zeros((binnum, 2)), np.zeros((binnum, 2)), np.zeros((binnum, 2)), np.zeros((binnum, 2))]
        group_number_output = [np.zeros((binnum, 2)), np.zeros((binnum, 2)), np.zeros((binnum, 2)), np.zeros((binnum, 2)), np.zeros((binnum, 2))]

    print('computing survey systematics')
    # accumulate shear in each bin. 
    for i, fname in tqdm(enumerate(fs)):
        fp = os.path.join(mdet_input_filepaths, fname)
        if os.path.exists(fp):
            d = fio.read(fp)
        else:
            continue

        d_pix = hp.ang2pix(4096, d['ra'], d['dec'], nest=True, lonlat=True)
        if method=='pixel':
            group_shear_output, group_number_output = _accum_shear_pixel(d, d_pix, group_shear_output, group_number_output)
        elif method=='bin':
            d_bin_signal = np.array([pix_signal[pix] for pix in d_pix])
            _accum_shear_bin(d, d_bin_signal, bin_edges, group_shear_output, group_number_output, mdet_mom)

    if method == 'pixel':
        mean_shear_output = np.zeros(healpix, dtype=[('pixel', 'i4'), ('signal', 'f8'), ('g1', 'f8'), ('g2', 'f8')])
        for pix in tqdm(range(len(group_shear_output[0][:]))):
            num_noshear = group_number_output[0][pix]
            if num_noshear[0] == 0:
                continue
            R11 = (group_shear_output[1][pix,0]/group_number_output[1][pix,0] - group_shear_output[2][pix,0]/group_number_output[2][pix,0])/0.02
            R22 = (group_shear_output[3][pix,1]/group_number_output[3][pix,1] - group_shear_output[4][pix,1]/group_number_output[4][pix,1])/0.02
            g1 = (group_shear_output[0][pix,0]/group_number_output[0][pix,0])/R11
            g2 = (group_shear_output[0][pix,1]/group_number_output[0][pix,1])/R22

            mean_shear_output['pixel'][pix] = pix
            mean_shear_output['signal'][pix] = pix_signal[pix]
            mean_shear_output['g1'][pix] = g1
            mean_shear_output['g2'][pix] = g2
    elif method == 'bin':
        mean_shear_output = np.zeros(binnum, dtype=[('mean_signal', 'f8'), ('g1', 'f8'), ('g2', 'f8')])
        mean_shear_output = _compute_shear(mean_shear_output, group_shear_output, group_number_output, binnum)

    fio.write(os.path.join(outpath, prop+'_'+band+'_systematics.fits'), mean_shear_output)

    pkdgrav_sim = len(glob.glob(os.path.join(outpath, 'seed__fid_*_'+band+'.fits')))
    if pkdgrav_sim != num_sim:
        print('computing the variance from the pkdgrav sims')

        for n in tqdm(range(num_sim)):
            with open(os.path.join(sample_variance_filepaths, 'seed__fid_'+str(n+1)+'.pkl'), 'rb') as handle:
                res = pickle.load(handle)
                handle.close()
            d = res['sources'][0]
            d_pix = hp.ang2pix(4096, d['ra'], d['dec'], nest=True, lonlat=True)
            d_bin_signal = np.array([pix_signal[pix] for pix in d_pix])
            _accum_shear_bin_pkdgrav(d, d_bin_signal, bin_edges, group_shear_output, group_number_output)

            mean_shear_output = np.zeros(binnum, dtype=[('mean_signal', 'f8'), ('g1', 'f8'), ('g2', 'f8')])
            mean_shear_output = _compute_shear(mean_shear_output, group_shear_output, group_number_output, binnum, pkdgrav=True)
            
            fio.write(os.path.join(outpath, 'seed__fid_'+str(n+1)+'_'+band+'.fits'), mean_shear_output)
        return None
    else:
        return None


def main(argv):

    f = open('/global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt', 'r')
    fs = f.read().split('\n')[:-1]

    # properties; airmass, seeing, exposure time, sky brightness
    # argument1: airmass, fwhm, exposure, sky
    # argument2: g, i
    prop = sys.argv[1]
    band = sys.argv[2]
    survey_property = glob.glob('/global/project/projectdirs/des/myamamot/survey_property_maps/'+prop+'_*_'+band+'.fits')[0]

    num_sim = 200
    method = 'bin'

    mdet_input_filepaths = sys.argv[3]
    sample_variance_filepaths = sys.argv[4]
    outpath = sys.argv[5]
    mdet_mom = sys.argv[6]
    survey_systematic_maps(fs, mdet_input_filepaths, sample_variance_filepaths, survey_property, method, num_sim, outpath, prop, band, mdet_mom)

if __name__ == "__main__":
    main(sys.argv)