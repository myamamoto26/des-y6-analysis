#!/bin/bash

#SBATCH -N 20
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=8
##SBATCH --mem=64G
#SBATCH -C cpu
#SBATCH --qos regular
#SBATCH --account des
#SBATCH --mail-user=masaya.yamamoto@duke.edu
#SBATCH --mail-type=ALL
#SBATCH -t 10:00:00

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
# source ~/mpi_env.sh
source activate mpi_dev

#run the application
srun --cpu-bind=cores python -u stars_fieldcenters_cross_correlation.py None "/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5b/metadetect_desdmv5a_cutsv5_patchesv5b.h5" None gauss "/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/field_centers/" "/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hsmap_random_v3_fcenters.fits" 5 "ng_fields" s2n_sizer '/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b/inverse_variance_weight_v5b_s2n_10-1000_Tratio_0.5-5.pickle' bootstrap None True True
