#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=04:00:00
#PBS -q debug
#PBS -A Catalyst
#PBS -l filesystems=home:eagle
#PBS -j oe

cd ${PBS_O_WORKDIR}

echo "Running on host:"
hostname
nvidia-smi

# ---- environment ----
module load conda
conda activate base
source ~/venvs/2025-09-25/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---- launch 4 independent MD jobs ----

CUDA_VISIBLE_DEVICES=0 python run_md.py $(cat group_gpu0.txt) > logs/gpu0.out 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python run_md.py $(cat group_gpu1.txt) > logs/gpu1.out 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python run_md.py $(cat group_gpu2.txt) > logs/gpu2.out 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python run_md.py $(cat group_gpu3.txt) > logs/gpu3.out 2>&1 &
PID3=$!

# ---- wait for all ----
wait $PID0 $PID1 $PID2 $PID3

echo "ALL GPU TASKS COMPLETED"