#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:59:59
#PBS -q debug
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

read -r -a CIFS < group_gpu0.txt

for i in 0 1 2 3; do
  CIF="${CIFS[$i]}"
  if [[ -z "${CIF// }" ]]; then
    echo "GPU $i: empty entry, skipping"
    continue
  fi
  echo "GPU $i running $CIF"
  CUDA_VISIBLE_DEVICES=$i python \
    /eagle/DFTCalculations/mehul/ml/athena/polaris/scripts/run_md.py "$CIF" \
    > ${OUT_BASE}/logs/node0_gpu${i}.out 2>&1 &
done
wait
echo "NODE 0 COMPLETED"