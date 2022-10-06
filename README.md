# des-y6-analysis

## catalog-validations-code
This directory contains the codes to validate the shape and PSF catalogs with null tests. 

## download-query-concatenation-code
This directory contains the codes to download catalog files with wget, query (GOLD, PIFF, METADETECT) objects from dessci/desoper, and concatenate the individual catalogs to make master catalog (although it is usually not a good idea to concatenate all the large files).

## example_notebook
This directory contains various notebooks for analyzing Y6 shear data. 

## image_simulations_analysis
This directory contains the configuration files for Y6 image simulations with Y6 pipeline like ```pizza-cutter``` and ```metadetection```. It also includes the analysis script to compute the shear calibration bias from the simulation results. 

## eastlake-examples (last updated a while ago)
This includes tests with ```metacalibration```, not ```metadetection```. 
This directory contains some of the test configuration files and the result for testing the performance of eastlake (https://github.com/des-science/eastlake). Once the conda environment is set up, you can set the output path etc. with setup_nersc_my.sh and run the sim. (eastlake is a refactored code of DES Y3 image simulations, https://github.com/des-science/y3-wl_image_sims.)
 
