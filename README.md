# des-y6-analysis

## catalog-validations-code
This directory contains the codes to validate the shape and PSF catalogs with null tests. 

## download-query-concatenation-code
This directory contains the codes to download catalog files with wget, query (GOLD, PIFF, METADETECT) objects from dessci/desoper, and concatenate the individual catalogs to make master catalog (although it is usually not a good idea to concatenate all the large files).

## eastlake-examples
This directory contains some of the test configuration files and the result for testing the performance of eastlake (https://github.com/des-science/eastlake). Once the conda environment is set up, you can set the output path etc. with setup_nersc_my.sh and run the sim. (eastlake is a refactored code of DES Y3 image simulations, https://github.com/des-science/y3-wl_image_sims.)

## image_simulations_config
This directory contains the configuration files to be injected to run the pizza-cutter. It also includes the code to produce the config file from the tiles necessary to create the coadd. 
