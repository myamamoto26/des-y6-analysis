# How to run the validation scripts.

This directory contains the following python scripts:
1. cuts_and_save_catalogs.py
2. compute_shear_response.py
3. inverse_weight.py
4. make_flat_catalog.py
5. mdet_rho_stats.py & rho_stats.py
-- <- up to date
6. 2pt_corr.py
7. compute_ebmode.py
8. survey_property_systematics.py
9. mdet_systematics.py & get_ccdnum_expnum.py
10. mean_shear_spatial_variations.py

Files that are not used for Y6 shear catalog paper:
- piff_diagnostics.py: computes the PSF residuals plots.
- mean_shear_bin_statistics.py & predefine_bins.py: computes the mean shear as a function of PSF size/shape and galaxy size. 
- plot_individual_objects.py: renders the pizza-slice images from the object ID in metadetection catalog. 
- metadetect_mask.py: figures out the radius of the mask around hyperleda objects. 


## cuts_and_save_catalogs.py
If one wants to run the validation tests with raw data (no selection cuts), go ahead and skip this step. 
Otherwise, if one would like to make selection cuts and save catalogs, 

```python cuts_and_save_catalogs.py /global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt /global/project/projectdirs/des/myamamot/metadetect/v3 /global/cscratch1/sd/myamamot/metadetect/cuts_v3 /global/project/projectdirs/des/myamamot/metadetect/y6-combined-hsmap16384-nomdet.fits```


## compute_shear_response.py
If one wants to compute the shear response over all the catalogs, 

```python compute_shear_response.py /global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt /global/cscratch1/sd/myamamot/metadetect/cuts_v3 /global/cscratch1/sd/myamamot/metadetect/shear_response_v2.txt```


## inverse_weight.py
If one wants to compute galaxy count, RMS of the shear, the shear response, and the inverse variance shear weight as a function of S/N and Tratio (T/Tpsf), 

```python inverse_weight.py /global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt /global/cscratch1/sd/myamamot/metadetect/cuts_v3 /global/cscratch1/sd/myamamot/metadetect/inverse_variance_weight_v3_Trcut_snmax1000.pickle```


## make_flat_catalog.py
If one wants to make a flat catalog (all the tiles combined) for simulating convergence/shear field for sample variance from simulations like PKDGRAV or 2pt analysis (this is under construction), 

```python make_flat_catalog.py /global/cscratch1/sd/myamamot/metadetect/inverse_variance_weight_v3_Trcut_snmax1000.pickle /global/cscratch1/sd/myamamot/metadetect/cuts_v3 /global/cscratch1/sd/myamamot/metadetect/shear_response_v3.txt /global/cscratch1/sd/myamamot/metadetect/cuts_v3/*_metadetect-v5_mdetcat_part0000.fits /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3_snmax1000.pkl```

Note that this inherits what is computed in inverse_weight.py. The shear in this flat catalog is corrected for the shear weight in the bins of S/N and T/Tpsf. 


## mdet_rho_stats.py & rho_stats.py
If one wants to compute the 2pt auto-/cross-correlation functions of PSF size/shape or galaxy size/shape for computing rho-/tau-statistics, 

```python mdet_rho_stats.py True True 60 all_JK /global/cscratch1/sd/myamamot/metadetect/shear_response_v3.txt /global/cscratch1/sd/myamamot/metadetect/cuts_v3/*_metadetect-v5_mdetcat_part0000.fits /global/cscratch1/sd/myamamot/metadetect/rho_tau_stats

Note that the computation of the correlation function with treecorr happens in rho-stats.py. 