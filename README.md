# des-y6-analysis

<<<<<<< Updated upstream
## catalog-validations-code
This directory contains the codes to validate the shape and PSF catalogs with null tests. 

## download-query-concatenation-code
This directory contains the codes to download catalog files with wget, query (GOLD, PIFF, METADETECT) objects from dessci/desoper, and concatenate the individual catalogs to make master catalog (although it is usually not a good idea to concatenate all the large files).

## eastlake-examples
This directory contains some of the test configuration files and the result for testing the performance of eastlake (https://github.com/des-science/eastlake). Once the conda environment is set up, you can set the output path etc. with setup_nersc_my.sh and run the sim. (eastlake is a refactored code of DES Y3 image simulations, https://github.com/des-science/y3-wl_image_sims.)

## image_simulations_config
This directory contains the configuration files to be injected to run the pizza-cutter. It also includes the code to produce the config file from the tiles necessary to create the coadd. 
=======
- test des-science/eastlake code.

- analyze metadetect catalogs. 

## How to run each test. 
It needs to be run on interactive nodes on NERSC due to the limited number of procs on login node. 

For each ```config.yaml```, 

```
source setup_nersc_my.sh

run-eastlake-sim your_config_file.yaml base_dir_(e.g., /global/cscratch1/sd/myamamot/)
```

__If the job exits out in the middle of steps.__

Change the step names you want to continue from and run

```
run-eastlake-sim your_config_file.yaml base_dir_(e.g., /global/cscratch1/sd/myamamot/) --resume_from path_to_previous_job_record.pkl
```

## How to analyze shear calibration from mcal catalogs. 
__If you only have one simulation with artificial shear (g) (e.g., g1=0.02, g2=0.00)__

Run ```python measure_shear_nocancel.py mcal_catalog_name.fits```

to calculate m1,c1. 

__If you have full sets of simulation (g1,g2)={(0.02,0.00), (-0.02,0.00), (0.00,0.02), (0.00,-0.02)}__

Run ```python measure_shear_cancel_patches.py mcal_catalog_name_g10.02_g20.00.fits```

to calculate m1,c1. 

## Runs
__v001_singleband_nostar_grid.yaml__

Exp galaxy profile, Gaussian PSF, No stars, single band (r), 80,000 objects on grid. 

```
m1: 0.000382 +/- 0.000113

c1: -0.000008 +/- 0.000002

m2: 0.000350 +/- 0.000063

c2: 0.000025 +/- 0.000006
```
>>>>>>> Stashed changes
