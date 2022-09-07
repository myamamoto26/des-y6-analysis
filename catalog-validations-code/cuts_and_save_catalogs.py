
import fitsio as fio
import os, sys
import numpy as np
from tqdm import tqdm
import healsparse
import numpy.lib.recfunctions as rf 


def make_cuts_and_save_catalogs(argv):
    """Saves metadetection catalogs in /CSCRATCH space with appropriate selection cuts from raw metadetection catalogs.

    Parameters
    ----------
    mdet_files: Text file of the list of filenames of the metadetection catalogs
    Example) /global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt

    mdet_input_filepaths: The file path to the directory in which the input metadetection catalogs exist
    Example) /global/project/projectdirs/des/myamamot/metadetect/v3
    
    mdet_output_filepaths: The file path to the directory in which the output file is written
    Example) /global/cscratch1/sd/myamamot/metadetect/cuts_v3

    mask_map: Fits file of the healsparse map of the catalog masking (including gold, hyperleda, metadetect)
    Example) /global/project/projectdirs/des/myamamot/metadetect/y6-combined-hsmap16384-nomdet.fits

    mdet_mom: which measurement do we want to make cuts on
    Example) wmom

    shear_bands: which bands do we want to make cuts on
    Example) 0123
    """

    mdet_files = sys.argv[1]
    mdet_input_filepaths = sys.argv[2]
    mdet_output_filepaths = sys.argv[3]
    mask_map = sys.argv[4]
    mdet_mom = sys.argv[5]
    mdet_bands = sys.argv[6]

    # Apply cuts + image masks, and save the catalogs.
    hmap = healsparse.HealSparseMap.read(mask_map)
    f = open(mdet_files, 'r')
    fs = f.read().split('\n')[:-1]
    mdet_filenames = [fname.split('/')[-1] for fname in fs]
    tilenames = [d.split('_')[0] for d in mdet_filenames]

    for fname in tqdm(mdet_filenames):

        if os.path.exists(os.path.join(mdet_output_filepaths, fname)):
            continue
        elif not os.path.exists(os.path.join(mdet_input_filepaths, fname)):
            print('no file found for this tile', fname)
            continue
        try:
            d = fio.read(os.path.join(mdet_input_filepaths, fname))
        except:
            print(fname)
            continue

        mag_g = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_g"])
        mag_r = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_r"])
        mag_i = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_i"])
        mag_z = 30.0 - 2.5*np.log10(d[mdet_mom+"_band_flux_z"])
        gmr = mag_g - mag_r
        rmi = mag_r - mag_i
        imz = mag_i - mag_z

        msk = ((d[mdet_mom+"_flags"] == 0) & 
                (d["mask_flags"] == 0) & 
                (d["shear_bands"] == mdet_bands) & 
                (d[mdet_mom+"_band_flux_flags_g"] == 0) & 
                (d[mdet_mom+"_band_flux_flags_r"] == 0) & 
                (d[mdet_mom+"_band_flux_flags_i"] == 0) & 
                (d[mdet_mom+"_band_flux_flags_z"] == 0) & 
                (d[mdet_mom+"_s2n"] > 10) & 
                (d["mfrac"] < 0.1) &
                (np.abs(gmr) < 5) & 
                (np.abs(rmi) < 5) & 
                (np.abs(imz) < 5) & 
                np.isfinite(mag_g) & 
                np.isfinite(mag_r) &
                np.isfinite(mag_i) & 
                np.isfinite(mag_z) & 
                (mag_g < 26.5) & 
                (mag_r < 26.5) & 
                (mag_i < 26.2) & 
                (mag_z < 25.6))
        if mdet_mom == 'wmom':
            msk_add = ((d[mdet_mom+"_T_ratio"] > 1.2))
            msk = msk & msk_add
        elif mdet_mom == 'pgauss':
            # msk_add = ((d[mdet_mom+"_T_ratio"] > 0.5) & (d[mdet_mom+"_T"] < 1.9 - 2.8*d[mdet_mom+"_T_err"]))
            msk_add = ((d[mdet_mom+"_T_ratio"] > 0.5) & (d[mdet_mom+"_T"] < 1.9 - 2.8*d[mdet_mom+"_T_err"]) & (d[mdet_mom+"_T"] > -4.25 + 27*d[mdet_mom+"_T_err"]))
            msk = msk & msk_add
        elif mdet_mom == 'pgauss_reg0.90':
            msk_add = ((d[mdet_mom+"_T_ratio"] > 0.5) & (d[mdet_mom+"_T"] < 1.9 - 2.8*d[mdet_mom+"_T_err"]))
            msk = msk & msk_add
        
        in_footprint = hmap.get_values_pos(d["ra"], d["dec"], valid_mask=True)

        total_msk = (msk & in_footprint)
        d_msk = d[total_msk]

        if mdet_mom == 'wmom':
            columns = ['slice_id', 'mdet_step', 'ra', 'dec', 'ra_noshear', 'dec_noshear', 'y_noshear', 'x_noshear', 'y', 'x', 'slice_y', 'slice_x', 'slice_y_noshear', 'slice_x_noshear', 'hpix_16384', 'hpix_16384_noshear', 'filename', 'tilename', 'mask_flags', 'mask_flags_noshear', 'nepoch_g', 'nepoch_r', 'nepoch_i', 'nepoch_z','nepoch_eff_g', 'nepoch_eff_r', 'nepoch_eff_i', 'nepoch_eff_z', 'wmom_flags', 'wmom_psf_flags', 'wmom_psf_g_1', 'wmom_psf_g_2', 'wmom_psf_T', 'wmom_obj_flags', 'wmom_s2n', 'wmom_g_1', 'wmom_g_2', 'wmom_g_cov_1_1', 'wmom_g_cov_1_2', 'wmom_g_cov_2_2', 'wmom_T', 'wmom_T_flags', 'wmom_T_err', 'wmom_T_ratio', 'wmom_band_flux_flags_g', 'wmom_band_flux_flags_r', 'wmom_band_flux_flags_i', 'wmom_band_flux_flags_z', 'wmom_band_flux_g', 'wmom_band_flux_r', 'wmom_band_flux_i', 'wmom_band_flux_z', 'wmom_band_flux_err_g', 'wmom_band_flux_err_r', 'wmom_band_flux_err_i', 'wmom_band_flux_err_z', 'shear_bands', 'ormask', 'mfrac', 'bmask', 'mfrac_img', 'ormask_noshear', 'mfrac_noshear', 'bmask_noshear', 'psfrec_flags', 'psfrec_g_1', 'psfrec_g_2', 'psfrec_T']
        elif mdet_mom == 'pgauss':
            columns = ['slice_id', 'mdet_step', 'ra', 'dec', 'ra_noshear', 'dec_noshear', 'y_noshear', 'x_noshear', 'y', 'x', 'slice_y', 'slice_x', 'slice_y_noshear', 'slice_x_noshear', 'hpix_16384', 'hpix_16384_noshear', 'filename', 'tilename', 'mask_flags', 'mask_flags_noshear', 'nepoch_g', 'nepoch_r', 'nepoch_i', 'nepoch_z','nepoch_eff_g', 'nepoch_eff_r', 'nepoch_eff_i', 'nepoch_eff_z', 'pgauss_flags', 'pgauss_psf_flags', 'pgauss_psf_g_1', 'pgauss_psf_g_2', 'pgauss_psf_T', 'pgauss_obj_flags', 'pgauss_s2n', 'pgauss_g_1', 'pgauss_g_2', 'pgauss_g_cov_1_1', 'pgauss_g_cov_1_2', 'pgauss_g_cov_2_2', 'pgauss_T', 'pgauss_T_flags', 'pgauss_T_err', 'pgauss_T_ratio', 'pgauss_band_flux_flags_g', 'pgauss_band_flux_flags_r', 'pgauss_band_flux_flags_i', 'pgauss_band_flux_flags_z', 'pgauss_band_flux_g', 'pgauss_band_flux_r', 'pgauss_band_flux_i', 'pgauss_band_flux_z', 'pgauss_band_flux_err_g', 'pgauss_band_flux_err_r', 'pgauss_band_flux_err_i', 'pgauss_band_flux_err_z', 'shear_bands', 'ormask', 'mfrac', 'bmask', 'mfrac_img', 'ormask_noshear', 'mfrac_noshear', 'bmask_noshear', 'psfrec_flags', 'psfrec_g_1', 'psfrec_g_2', 'psfrec_T']
        elif mdet_mom == 'pgauss_reg0.90':
            columns = ['slice_id', 'mdet_step', 'ra', 'dec', 'ra_noshear', 'dec_noshear', 'y_noshear', 'x_noshear', 'y', 'x', 'slice_y', 'slice_x', 'slice_y_noshear', 'slice_x_noshear', 'hpix_16384', 'hpix_16384_noshear', 'filename', 'tilename', 'mask_flags', 'mask_flags_noshear', 'nepoch_g', 'nepoch_r', 'nepoch_i', 'nepoch_z','nepoch_eff_g', 'nepoch_eff_r', 'nepoch_eff_i', 'nepoch_eff_z', 'pgauss_reg0.90_flags', 'pgauss_reg0.90_psf_flags', 'pgauss_reg0.90_psf_g_1', 'pgauss_reg0.90_psf_g_2', 'pgauss_reg0.90_psf_T', 'pgauss_reg0.90_obj_flags', 'pgauss_reg0.90_s2n', 'pgauss_reg0.90_g_1', 'pgauss_reg0.90_g_2', 'pgauss_reg0.90_g_cov_1_1', 'pgauss_reg0.90_g_cov_1_2', 'pgauss_reg0.90_g_cov_2_2', 'pgauss_reg0.90_T', 'pgauss_reg0.90_T_flags', 'pgauss_reg0.90_T_err', 'pgauss_reg0.90_T_ratio', 'pgauss_reg0.90_band_flux_flags_g', 'pgauss_reg0.90_band_flux_flags_r', 'pgauss_reg0.90_band_flux_flags_i', 'pgauss_reg0.90_band_flux_flags_z', 'pgauss_reg0.90_band_flux_g', 'pgauss_reg0.90_band_flux_r', 'pgauss_reg0.90_band_flux_i', 'pgauss_reg0.90_band_flux_z', 'pgauss_reg0.90_band_flux_err_g', 'pgauss_reg0.90_band_flux_err_r', 'pgauss_reg0.90_band_flux_err_i', 'pgauss_reg0.90_band_flux_err_z', 'shear_bands', 'ormask', 'mfrac', 'bmask', 'mfrac_img', 'ormask_noshear', 'mfrac_noshear', 'bmask_noshear', 'psfrec_flags', 'psfrec_g_1', 'psfrec_g_2', 'psfrec_T']
        
        d_out = rf.repack_fields(d_msk[columns])
        fio.write(os.path.join(mdet_output_filepaths, fname), d_out)

if __name__ == "__main__":
    make_cuts_and_save_catalogs(sys.argv)