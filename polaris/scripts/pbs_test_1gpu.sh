#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -A Catalyst
#PBS -l filesystems=home:eagle
#PBS -j oe

cd ${PBS_O_WORKDIR}

module load conda
conda activate base
source ~/venvs/2025-09-25/bin/activate

python run_md.py LLZO_010_Li100_order0.cif