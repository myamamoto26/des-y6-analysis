#!/bin/bash

#Set up python (I use my own conda enviroment - see http://www.nersc.gov/users/data-analytics/data-analytics-2/python/anaconda-python/)
module load python
module load cray-fftw gcc
#This seems to be required to setup conda
#. /usr/common/software/python/2.7-anaconda-2019.07/etc/profile.d/conda.sh
conda activate eastlake-dev

#Set up some enviroment variables
export DES_TEST_DATA=/global/project/projectdirs/des/y3-image-sims #you'll need to change this if not working at nersc
export SIM_OUTPUT_DIR=/global/cscratch1/sd/myamamot/
export IMSIM_DIR=/global/homes/m/myamamot/DES/montara
export NGMIX_CONFIG_DIR=${IMSIM_DIR}/ngmix_config
export PIFF_DATA_DIR=/global/cscratch1/sd/maccrann/DES #Leave this as-is - this is where the piff files are stored.
export PIFF_RUN=y3a1-v29

export DESDATA=${DES_TEST_DATA}

mkdir -p ${SIM_OUTPUT_DIR}
mkdir -p ${SIM_OUTPUT_DIR}/job_outputs

#add sextractor and swarp binaries to path
export PATH=$PATH:/project/projectdirs/des/rrollins/.MEPY3A2+19/bin

export OMP_NUM_THREADS=1
