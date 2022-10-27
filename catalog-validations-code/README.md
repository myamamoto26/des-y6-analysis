# How to run the validation scripts.

This directory contains the following python scripts:
1. cuts_and_save_catalogs.py
2. compute_shear_response.py
3. mean_shear_stats.py
4. inverse_weight.py
5. make_flat_catalog.py
6. mdet_rho_stats.py & rho_stats.py
7. 2pt_corr.py
8. compute_shape_noise_neff.py
9. stars_fieldcenters_cross_correlation.py
10. mean_shear_spatial_variations.py
11. survey_property_systematics.py

Files that are not used for Y6 shear catalog paper:
- piff_diagnostics.py: computes the PSF residuals plots.
- mean_shear_bin_statistics.py & predefine_bins.py: computes the mean shear as a function of PSF size/shape and galaxy size. 
- plot_individual_objects.py: renders the pizza-slice images from the object ID in metadetection catalog. 
- metadetect_mask.py: figures out the radius of the mask around hyperleda objects. 
- get_ccdnum_expnum.py: Gets CCD numbers and exposure numbers from pizza-cutter meds files. This can simply be done querying from desdm database. 


## cuts_and_save_catalogs.py
If one wants to run the validation tests with raw data (no selection cuts), go ahead and skip this step. 
Otherwise, if one would like to make selection cuts and save catalogs, 

```python cuts_and_save_catalogs.py /global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt /global/project/projectdirs/des/myamamot/metadetect/v3 /global/cscratch1/sd/myamamot/metadetect/cuts_v3 /global/project/projectdirs/des/myamamot/metadetect/y6-combined-hsmap16384-nomdet.fits wmom 0123```

### Required items: raw metadetection catalogs, the file which contains metadetection file names, the survey mask map


## compute_shear_response.py
If one wants to compute the shear response over all the catalogs, 

```python compute_shear_response.py /global/project/projectdirs/des/myamamot/metadetect/mdet_files.txt /global/cscratch1/sd/myamamot/metadetect/cuts_v3 /global/cscratch1/sd/myamamot/metadetect/shear_response_v2.txt wmom```

### Required items: selection-cut metadetection catalogs, the file which contains metadetection file names


## mean_shear_stats.py (updated 10/05/22)
If one wants to compute the mean shear over properties like PSF shape/size and galaxy size, 

```python mean_shear_stats.py /global/cfs/cdirs/des/y6-shear-catalogs/metadetection_patches/patch-*.fits bin_stats.fits bin_hist.pickle wmom /global/cscratch1/sd/myamamoto/des-y6-analysis/y6_measurement 5000000 mean_shear_stats.pickle /global/cscratch1/sd/myamamot/metadetect/inverse_variance_weight_v3_Trcut_snmax1000.pickle None```

### Required items: selection-cut metadetection catalogs, the file which contains shear response+shear weight in the bins of S/N and T/Tpsf


## inverse_weight.py (updated 10/05/22)
If one wants to compute galaxy count, RMS of the shear, the shear response, and the inverse variance shear weight as a function of S/N and Tratio (T/Tpsf), 

```python inverse_weight.py /global/cfs/cdirs/des/y6-shear-catalogs/metadetection_patches/patch-*.fits /global/cscratch1/sd/myamamot/metadetect/inverse_variance_weight_v3_Trcut_snmax1000.pickle wmom```

### Required items: selection-cut metadetection catalogs, the file which contains metadetection file names


## make_flat_catalog.py
If one wants to make a flat catalog (all the tiles combined) for simulating convergence/shear field for sample variance from simulations like PKDGRAV or 2pt analysis or photo-z estimate, 

```python make_flat_catalog.py /global/cscratch1/sd/myamamot/metadetect/inverse_variance_weight_v3_Trcut_snmax1000.pickle /global/cscratch1/sd/myamamot/metadetect/cuts_v3 /global/cscratch1/sd/myamamot/metadetect/shear_response_v3.txt /global/cscratch1/sd/myamamot/metadetect/cuts_v3/*_metadetect-v5_mdetcat_part0000.fits /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3_snmax1000.pkl /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3_snmax1000.fits wmom```

### Required items: selection-cut metadetection catalogs, the file which contains shear response+shear weight in the bins of S/N and T/Tpsf, the file which contains the shear reponse over all the tiles
Note that this inherits what is computed in inverse_weight.py. The shear in this flat catalog is corrected for the shear weight in the bins of S/N and T/Tpsf. 


## mdet_rho_stats.py & rho_stats.py
If one wants to compute the 2pt auto-/cross-correlation functions of PSF size/shape or galaxy size/shape for computing rho-/tau-statistics, 

```python mdet_rho_stats.py True True 60 all_JK /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3_snmax1000.pkl /global/cscratch1/sd/myamamot/metadetect/rho_tau_stats```

### Required items: PIFF catalogs, selection-cut metadetection flat catalog
Note that the computation of the correlation function with treecorr happens in rho-stats.py. 

## 2pt_corr.py
If one wants to compute 2pt correlation function in a very fine bin (~1000 bins) for the use of B-mode estimation and do the B-mode estimation, 

```python 2pt_corr.py /global/cscratch/sd/myamamot/des-y6-analysis/y6_measurement/metadetection_v2.fits /global/project/projectdirs/des/myamamot/2pt_corr /global/project/projectdirs/des/myamamot/2pt_corr/y6_shear2pt_nontomo_v3.fits /global/project/projectdirs/des/myamamot/2pt_corr/y6_shear2pt_nontomo_v3.pkl True jackknife```

### Required items:  selection-cut metadetection flat catalog
Note that this script includes building the B-mode estimator with hybrideb. 


## compute_shape_noise_neff.py
If one wants to compute the shape noise and effective number density (C13, H12) from the flat catalog, 

```python compute_shape_noise_neff.py /global/cscratch1/sd/myamamot/sample_variance/data_catalogs_weighted_v3_snmax1000.fits c13 /global/cscratch1/sd/myamamot/metadetect```

### Required items:  selection-cut metadetection flat catalog


## stars_fieldcenters_cross_correlation.py
If one wants to compute the cross-correlation function of shear with bright/faint stars or CCD field centers, 

```python stars_fieldcenters_cross_correlation.py /global/cscratch1/sd/myamamot/metadetect/shear_response_v2.txt /global/cscratch1/sd/myamamot/metadetect/cuts_v2/*_metadetect-v5_mdetcat_part0000.fits wmom /global/cscratch1/sd/myamamot/metadetect/systematics /global/homes/m/myamamot/DES/des-y6-analysis/y6-combined-hsmap_random.fits Y6A2_PIZACUTTER```

### Required items:  selection-cut metadetection catalogs, the file which contains a random point map made from the survey mask map, the file which contains the shear reponse over all the tiles


## mean_shear_spatial_variations.py
If one wants to compute the mean shear variations across focal planes (CCDs) to check for charge transfer inefficiency (CTI) effect, 

```python /global/homes/m/myamamot/DES/des-y6-analysis/catalog-validations-code/mean_shear_spatial_variations.py /global/cfs/cdirs/des/y6-shear-catalogs/metadetection /global/cfs/cdirs/des/y6-shear-catalogs/metadetection/mdet_files.txt /global/cfs/cdirs/des/myamamot/pizza-slice/pizza-cutter-coadds-info.fits /global/cscratch/sd/myamamot/des-y6-analysis/y6_measurement/shear_variations True True wmom 2```

### Required items:  selection-cut metadetection catalogs, the file which contains metadetection file names, the file which contains the filenames of pizza-cutter meds files


## survey_property_systematics.py
If one wants to compute the survey map systematics (mean shear vs survey property like airmass) based on the survey property maps and the sample variance sims, 

```python survey_property_systematics.py airmass g /global/cscratch1/sd/myamamot/metadetect/cuts_v3 /global/cscratch1/sd/myamamot/sample_variance /global/cscratch1/sd/myamamot/survey_property_maps/airmass wmom```

### Required items:  selection-cut metadetection catalogs, the survey property catalogs, PKDGRAV simulation catalogs