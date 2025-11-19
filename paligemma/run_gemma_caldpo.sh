#!/usr/bin/env bash
# ==============================================================
# Script: run_paligemma_caldpo.sh
# Purpose: Run PaliGemma Cal-DPO fine-tuning locally
# ==============================================================

START_TIME=$(date +%s)
set -e   # Exit on any error

# ----------- CONFIGURATION -----------
MODEL_NAME="google/paligemma-3b-pt-224"
DATASET_NAME="Eftekhar/HA-DPO-Dataset"
OUTPUT_DIR="./paligemma-caldpo-output"

EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8

NUM_PROC=16
NUM_WORKERS=16

BF16=true
USE_LORA=true
GRADIENT_CHECKPOINTING=true
BETA=0.1

LOG_STEPS=10
JSON_LOG_NAME="caldpo_paligemma_logs.json"

PYTHON_SCRIPT="gemma_caldpo.py"
LOG_FILE="train_logs-palgemma-caldpo.txt"

# -------------------------------------

echo "=============================================================="
echo "🚀 Starting PaliGemma Cal-DPO Fine-Tuning"
echo "Model:           ${MODEL_NAME}"
echo "Dataset:         ${DATASET_NAME}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "JSON Log File:   ${JSON_LOG_NAME}"
echo "=============================================================="
echo ""

# ----------- ENVIRONMENT SETUP -----------
source ~/.bashrc
conda activate vis-reasoning       # ✔ your conda environment

export CUDA_VISIBLE_DEVICES=0   # Use GPU 0 only

# Print environment info
echo "Torch version:       $(python -c 'import torch; print(torch.__version__)')"
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
    --beta "$BETA" \
    --json_log_name "$JSON_LOG_NAME" \
    $( [ "$BF16" = true ] && echo "--bf16" ) \
    $( [ "$USE_LORA" = true ] && echo "--use_lora" ) \
    $( [ "$GRADIENT_CHECKPOINTING" = true ] && echo "--gradient_checkpointing" )

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=============================================================="
echo "Training finished in $(($ELAPSED / 60)) min $(($ELAPSED % 60)) sec"
echo "Logs saved to: $LOG_FILE"
echo "Models saved to: saved_models/$(basename $OUTPUT_DIR)"
echo "=============================================================="
