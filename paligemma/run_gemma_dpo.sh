#!/usr/bin/env bash
# ==============================================================
# Script: run_paligemma_dpo.sh
# Purpose: Run PaliGemma DPO fine-tuning locally (non-SLURM)
# ==============================================================

START_TIME=$(date +%s)
set -e   # Exit on any error

# ----------- CONFIGURATION -----------
MODEL_NAME="google/paligemma-3b-pt-224"     # Main PaliGemma model
DATASET_NAME="Eftekhar/HA-DPO-Dataset"       # Your dataset
OUTPUT_DIR="./paligemma-hadpo"          # Save directory

EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM_STEPS=8
NUM_PROC=16
NUM_WORKERS=16

BF16=true
USE_LORA=true
GRADIENT_CHECKPOINTING=true

LOG_STEPS=10
JSON_LOG_NAME="paligemma_dpo_logs.json"   # NEW — JSON logs filename

PYTHON_SCRIPT="gemma_dpo.py"
LOG_FILE="train_logs-paligemma-dpo.txt"

# -------------------------------------

echo "=============================================================="
echo "🚀 Starting PaliGemma DPO Fine-Tuning"
echo "Model:      ${MODEL_NAME}"
echo "Dataset:    ${DATASET_NAME}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "JSON Logs:  ${JSON_LOG_NAME}"
echo "=============================================================="
echo ""

# ----------- ENVIRONMENT SETUP -----------
source ~/.bashrc
conda activate vis-reasoning     # ✔ Your environment

export CUDA_VISIBLE_DEVICES=0

# Print environment info
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers:  $(python -c 'import transformers; print(transformers.__version__)')"
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
    --json_log_name "$JSON_LOG_NAME" \
    $( [ "$BF16" = true ] && echo "--bf16" ) \
    $( [ "$USE_LORA" = true ] && echo "--use_lora" ) \
    $( [ "$GRADIENT_CHECKPOINTING" = true ] && echo "--gradient_checkpointing" ) \
    | tee "$LOG_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Script finished in $ELAPSED seconds ($(($ELAPSED / 60)) minutes and $(($ELAPSED % 60)) seconds)"
