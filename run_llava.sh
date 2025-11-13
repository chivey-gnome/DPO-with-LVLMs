#!/usr/bin/env bash
# ==============================================================
# Script: run_llava_dpo.sh
# Purpose: Run LLaVA DPO fine-tuning locally (non-SLURM)
# ==============================================================

START_TIME=$(date +%s)
# Exit immediately on error
set -e

# ----------- CONFIGURATION -----------
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
DATASET_NAME="Eftekhar/HA-DPO-Dataset"
OUTPUT_DIR="./llava-base-dpo"
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
NUM_PROC=16
NUM_WORKERS=16
BF16=true
USE_LORA=true
GRADIENT_CHECKPOINTING=true
LOG_STEPS=10
PYTHON_SCRIPT="llava_dpo.py"
LOG_FILE="train_logs-llava-base-dpo.txt"
# -------------------------------------

echo "=============================================================="
echo "🚀 Starting LLaVA DPO Fine-Tuning"
echo "Model:      ${MODEL_NAME}"
echo "Dataset:    ${DATASET_NAME}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================================="
echo ""

# ----------- ENVIRONMENT SETUP -----------
# Activate your conda environment if needed
# source ~/.bashrc
# conda activate new-hadpo

# Optional: set visible GPU
export CUDA_VISIBLE_DEVICES=0

# Print environment info
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""

# ----------- RUN TRAINING -----------
python "$PYTHON_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --num_proc "$NUM_PROC" \
    --num_workers "$NUM_WORKERS" \
    --log_steps "$LOG_STEPS" \
    $( [ "$BF16" = true ] && echo "--bf16" ) \
    $( [ "$USE_LORA" = true ] && echo "--use_lora" ) \
    $( [ "$GRADIENT_CHECKPOINTING" = true ] && echo "--gradient_checkpointing" ) \
    | tee "$LOG_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Script finished in $ELAPSED seconds ($(($ELAPSED / 60)) minutes and $(($ELAPSED % 60)) seconds)"