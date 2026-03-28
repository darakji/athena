#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:59:59
#PBS -q debug
#PBS -A DFTCalculations
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o ensemble_uncertainty.log
#PBS -e ensemble_uncertainty_err.log

cd ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda
conda activate base
source /home/phanim/venvs/2025-09-25/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Paths ──────────────────────────────────────────────────────────────────────

CONFIGS=/eagle/DFTCalculations/mehul/ml/athena/polaris/md_from_mace_fps_split17_it0/cifs

OUT_BASE=/eagle/DFTCalculations/mehul/ml/athena/polaris/ensemble_uncertainty_it0
mkdir -p ${OUT_BASE}/embeddings ${OUT_BASE}/logs

EXTRACT=/eagle/DFTCalculations/mehul/ml/mace/mace/extract_embeddings.py

# ── 4 models → 4 GPUs ─────────────────────────────────────────────────────────

declare -A MODELS
MODELS[0]="/eagle/DFTCalculations/mehul/ml/MACE_models/mace_fps_split17_run-1.model"
MODELS[1]="/eagle/DFTCalculations/mehul/ml/MACE_models/mace_fps_split2_run-1.model"
MODELS[2]="/eagle/DFTCalculations/mehul/ml/MACE_models/mace_fps_split7_run-1.model"
MODELS[3]="/eagle/DFTCalculations/mehul/ml/MACE_models/mace_fps_split12_run-1.model"

declare -A TAGS
TAGS[0]="split17"
TAGS[1]="split2"
TAGS[2]="split7"
TAGS[3]="split12"

for i in 0 1 2 3; do
  MODEL="${MODELS[$i]}"
  TAG="${TAGS[$i]}"
  OUTPUT="${OUT_BASE}/embeddings/ensemble_${TAG}.extxyz"

  echo "GPU $i → model: $MODEL"
  CUDA_VISIBLE_DEVICES=$i python ${EXTRACT} \
    --model    "${MODEL}" \
    --configs  "${CONFIGS}" \
    --output   "${OUTPUT}" \
    --device   cuda \
    --batch_size 7 \
    --resume \
    > ${OUT_BASE}/logs/gpu${i}_${TAG}_${PBS_JOBID}.out 2>&1 &
done

wait
echo "All ensemble extractions complete."

# ── Post-process: compute force uncertainty + write Excel ─────────────────────

python /eagle/DFTCalculations/mehul/ml/athena/polaris/scripts/force_uncertainty_excel.py \
  --embeddings_dir ${OUT_BASE}/embeddings \
  --output_excel   ${OUT_BASE}/ensemble_force_uncertainty.xlsx \
  > ${OUT_BASE}/logs/postprocess_${PBS_JOBID}.out 2>&1

echo "NODE COMPLETED"
