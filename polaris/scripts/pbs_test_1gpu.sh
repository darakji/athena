#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:59:59
#PBS -q debug
#PBS -A DFTCalculations
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o pilot_output.log
#PBS -e pilot_error.log

# -------------------------------------------------
# Go to scripts directory (where run_md.py lives)
# -------------------------------------------------
cd ${PBS_O_WORKDIR}

echo "Running pilot ML-MD job on:"
hostname
nvidia-smi

# -------------------------------------------------
# Python environment
# -------------------------------------------------
module use /soft/modulefiles
module load conda
conda activate base
source /home/phanim/venvs/2025-09-25/bin/activate

# -------------------------------------------------
# Thread safety (critical for ML codes)
# -------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------------------------------------------------
# Output base (absolute path)
# -------------------------------------------------
OUT_BASE=/eagle/DFTCalculations/mehul/ml/athena/polaris/li_llzo_md_polaris
mkdir -p ${OUT_BASE}/logs

# -------------------------------------------------
# Run SINGLE CIF on SINGLE GPU (pilot)
# -------------------------------------------------
# Pick one representative CIF
TEST_CIF=LLZO_010_Li100_order0.cif

CUDA_VISIBLE_DEVICES=0 \
python run_md.py ${TEST_CIF} > ${OUT_BASE}/logs/pilot_gpu0.out 2>&1

echo "PILOT ML-MD JOB COMPLETED"