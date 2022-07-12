
## This test runs the following steps

```galsim_montara, true_detection, pizza-cutter, metadetect```

__Pre-requisite to run this configuration__
You would need to download the single-epoch images from DESDM with ```des-pizza-cutter-prep-tile``` in pizza-cutter package (https://github.com/beckermr/pizza-cutter/blob/main/bin/des-pizza-cutter-prep-tile). 

How to download these images is written in ```run_download.sh```. 

__Once you download the single-epoch images and make sure you have the correct environment,__
(You can create the environment from environment.yaml in eastlake. )

```run-eastlake-sim v007_grid_mdet_config.yaml ./sim_outputs```
