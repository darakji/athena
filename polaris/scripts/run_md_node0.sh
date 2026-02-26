#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:59:59
#PBS -q debug-scaling
#PBS -A DFTCalculations
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o output_node0.log
#PBS -e error_node0.log

cd ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda
conda activate base
source /home/phanim/venvs/2025-09-25/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUT_BASE=/eagle/DFTCalculations/mehul/ml/athena/polaris/li_llzo_md_polaris
mkdir -p ${OUT_BASE}/logs

mapfile -t CIFS < group_gpu0.txt

CUDA_VISIBLE_DEVICES=0 python run_md.py "${CIFS[0]}" > ${OUT_BASE}/logs/node0_gpu0.out 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python run_md.py "${CIFS[1]}" > ${OUT_BASE}/logs/node0_gpu1.out 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=2 python run_md.py "${CIFS[2]}" > ${OUT_BASE}/logs/node0_gpu2.out 2>&1 &
PID2=$!
CUDA_VISIBLE_DEVICES=3 python run_md.py "${CIFS[3]}" > ${OUT_BASE}/logs/node0_gpu3.out 2>&1 &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3
echo "NODE 0 COMPLETED"