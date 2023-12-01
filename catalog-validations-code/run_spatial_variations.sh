#!/bin/bash

#SBATCH -N 1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=32
##SBATCH --mem=64G
#SBATCH -C cpu
#SBATCH --qos regular
#SBATCH --account des
#SBATCH --mail-user=masaya.yamamoto@duke.edu
#SBATCH --mail-type=ALL
#SBATCH -t 03:00:00

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
# source ~/mpi_env.sh
source activate mpi_dev

#run the application
srun --cpu-bind=cores python /global/homes/m/myamamot/DES/des-y6-analysis/catalog-validations-code/mean_shear_spatial_variations.py /global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5b/tiles_blinded "/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5b/tiles_blinded/mdet_files.txt" "/global/cfs/cdirs/des/myamamot/pizza-slice/pizza-cutter-coadds-info.fits" /pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/shear_variations_red "/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b/inverse_variance_weight_v5b_s2n_10-1000_Tratio_0.5-5.pickle" True False False gauss 5

