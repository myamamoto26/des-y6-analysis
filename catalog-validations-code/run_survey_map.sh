#!/bin/bash

#SBATCH -N 5
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
##SBATCH --mem=64G
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH --mail-user=masaya.yamamoto@duke.edu
#SBATCH --mail-type=ALL
#SBATCH -t 08:00:00

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
source ~/mpi_env.sh

#run the application
srun --cpu-bind=cores python survey_property_systematics.py airmass "/global/cfs/cdirs/des/myamamot/survey_property_maps/y6a2_decasu_r_airmass_wmean.hs" "/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V4/jackknife_patches_blinded/patch-*.fits" /global/cscratch1/sd/myamamot/sample_variance /global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/v3/survey_sys/airmass gauss 3 1000000 shape_err
