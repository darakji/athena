#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=04:00:00
#PBS -q debug
#PBS -A Catalyst
#PBS -l filesystems=home:eagle
#PBS -j oe

cd ${PBS_O_WORKDIR}

module load conda
conda activate base
source ~/venvs/2025-09-25/bin/activate

GPU_ID=${PBS_ARRAY_INDEX}

mapfile -t CIFS < group_gpu${GPU_ID}.txt

python run_md.py "${CIFS[@]}"