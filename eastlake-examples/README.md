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
