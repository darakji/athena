#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:29:59
#PBS -q debug
#PBS -A DFTCalculations
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o output_pool_latents_slab_md.log
#PBS -e error_pool_latents_slab_md.log

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
SCRIPT=/eagle/DFTCalculations/mehul/ml/athena/polaris/scripts/pool_latents.py

# Directory containing the per-GPU .extxyz embedding files
INPUT_DIR=/eagle/DFTCalculations/mehul/ml/athena/polaris/remaining_slab_md_unfreeze_li/embeddings_trial0

# Output JSON
OUT_DIR=/eagle/DFTCalculations/mehul/ml/athena/polaris/remaining_slab_md_unfreeze_li/structure_latents
mkdir -p ${OUT_DIR}

OUTPUT_JSON=${OUT_DIR}/slab_md_unfreeze_li_structure_latents.json

# -------------------------
# Run (CPU only — just numpy mean-pooling, no GPU needed)
# -------------------------
echo "Starting structure-level latent pooling..."
echo "Input  : ${INPUT_DIR}"
echo "Output : ${OUTPUT_JSON}"

python ${SCRIPT} \
    --input      ${INPUT_DIR} \
    --output     ${OUTPUT_JSON} \
    --array_name mace_latent \
    --pool       mean \
    --key_mode   auto

echo "JOB COMPLETED"
