#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:59:59
#PBS -q debug-scaling
#PBS -A DFTCalculations
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o output.log
#PBS -e error.log

# -------------------------------------------------
# Move to scripts directory
# -------------------------------------------------
cd ${PBS_O_WORKDIR}

echo "Running on node:"
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
# Thread safety (critical)
# -------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------------------------------------------------
# Output dirs (absolute paths)
# -------------------------------------------------
OUT_BASE=/eagle/DFTCalculations/mehul/ml/athena/polaris/li_llzo_md_polaris
mkdir -p ${OUT_BASE}/logs

# -------------------------------------------------
# Launch 4 independent ML-MD jobs (1 GPU each)
# -------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python run_md.py $(cat group_gpu0.txt) > ${OUT_BASE}/logs/gpu0.out 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python run_md.py $(cat group_gpu1.txt) > ${OUT_BASE}/logs/gpu1.out 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python run_md.py $(cat group_gpu2.txt) > ${OUT_BASE}/logs/gpu2.out 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python run_md.py $(cat group_gpu3.txt) > ${OUT_BASE}/logs/gpu3.out 2>&1 &
PID3=$!

# -------------------------------------------------
# Wait for all GPUs
# -------------------------------------------------
wait $PID0 $PID1 $PID2 $PID3

echo "ALL ML-MD TASKS COMPLETED"