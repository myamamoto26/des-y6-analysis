#!/bin/bash

#SBATCH -N 1
##SBATCH --tasks-per-node=1
##SBATCH --cpus-per-task=8
##SBATCH --mem=64G
#SBATCH -C cpu
#SBATCH --qos regular
#SBATCH --account des
#SBATCH --mail-user=masaya.yamamoto@duke.edu
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
# source ~/mpi_env.sh
source activate mpi_dev

#run the application
python mean_shear_cell_coordinates.py "/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5b/metadetect_desdmv5a_cutsv5_patchesv5b.h5" s2n_sizer '/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b/inverse_variance_weight_v5b_s2n_10-1000_Tratio_0.5-5.pickle' "/pscratch/sd/m/myamamot/des-y6-analysis/y6_measurement/v5b_paper/" 6 mean_shear_coadd_grid
