#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:59:59
#PBS -q debug
#PBS -A DFTCalculations
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o output_embeddings_trial0.log
#PBS -e error_embeddings_trial0.log

cd ${PBS_O_WORKDIR}

# -------------------------
# Environment
# -------------------------
module use /soft/modulefiles
module load conda
conda activate base
source /home/phanim/venvs/2025-09-25/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------------------------
# Paths
# -------------------------
MODEL=/eagle/DFTCalculations/mehul/ml/MACE_models/mace_seed_1_trial0.model
SCRIPT=/eagle/DFTCalculations/mehul/ml/mace/mace/extract_embeddings.py
CIFS=/eagle/DFTCalculations/mehul/ml/athena/polaris/remaining_slab_md_unfreeze_li/cifs

OUT_BASE=/eagle/DFTCalculations/mehul/ml/athena/polaris/remaining_slab_md_unfreeze_li/embeddings_trial0
LOG_DIR=${OUT_BASE}/logs

mkdir -p ${OUT_BASE}
mkdir -p ${LOG_DIR}

WORLD_SIZE=4
BATCH_SIZE=7

E0S='{"3": -190.7590256408, "8": -442.9888796243, "40": -1380.1817128081, "57": -958.0774205521}'

# -------------------------
# Launch 1 process per GPU
# -------------------------
for GPU_ID in 0 1 2 3; do
  echo "Launching GPU ${GPU_ID}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} \
    --model ${MODEL} \
    --configs ${CIFS} \
    --output ${OUT_BASE}/polaris_emb_trial0_${GPU_ID}.extxyz \
    --device cuda \
    --batch_size ${BATCH_SIZE} \
    --gpu_id ${GPU_ID} \
    --world_size ${WORLD_SIZE} \
    --resume \
    --e0s "${E0S}" \
    > ${LOG_DIR}/gpu${GPU_ID}_${PBS_JOBID}.out 2>&1 &

done

# -------------------------
# Synchronize GPUs
# -------------------------
wait
echo "All GPU processes finished."

# -------------------------
# Merge outputs (safe)
# -------------------------
FINAL_OUT=${OUT_BASE}/polaris_emb.extxyz
rm -f ${FINAL_OUT}

for GPU_ID in 0 1 2 3; do
  PART=${OUT_BASE}/polaris_emb_gpu_trial0_${GPU_ID}.extxyz
  if [[ -f "${PART}" ]]; then
    cat "${PART}" >> "${FINAL_OUT}"
  else
    echo "Warning: missing ${PART}"
  fi
done

echo "Merged embeddings written to ${FINAL_OUT}"
echo "JOB COMPLETED"
