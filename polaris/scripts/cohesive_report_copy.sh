#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:29:59
#PBS -q debug
#PBS -A DFTCalculations
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o /eagle/DFTCalculations/mehul/ml/athena/scripts/output_cohesive_report.log

cd /eagle/DFTCalculations/mehul/ml/athena/scripts

# -------------------------
# Environment
# -------------------------
module use /soft/modulefiles
module load conda
conda activate base
source /home/phanim/venvs/2025-09-25/bin/activate

export OMP_NUM_THREADS=1

# -------------------------
# Paths
# -------------------------
SCRIPT=/eagle/DFTCalculations/mehul/ml/athena/scripts/extract_seed_cohesive_polaris.py

# -------------------------
# Run
# -------------------------
echo "Starting cohesive energy extraction..."
echo "PBS_O_WORKDIR: ${PBS_O_WORKDIR}"
echo "Script: ${SCRIPT}"
echo "Python: $(which python)"

python ${SCRIPT}

echo "JOB COMPLETED"

