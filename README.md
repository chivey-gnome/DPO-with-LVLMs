# LLaVA DPO Training & POPE Evaluation

This repository contains scripts for **fine-tuning LLaVA models using DPO** and **evaluating them on the POPE benchmark**.

---

## Table of Contents

- [LLaVA DPO Training \& POPE Evaluation](#llava-dpo-training--pope-evaluation)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
  - [File Structure](#file-structure)
  - [Training LLaVA with DPO](#training-llava-with-dpo)
    - [Run Training](#run-training)
    - [Key Configuration](#key-configuration)
  - [Evaluating on POPE Benchmark](#evaluating-on-pope-benchmark)
    - [Run Evaluation](#run-evaluation)
    - [Key Configuration](#key-configuration-1)
  - [Performance on POPE Benchmark](#performance-on-pope-benchmark)
    - [Run Evaluation](#run-evaluation-1)
  - [Evaluating on AMBER Benchmark](#evaluating-on-amber-benchmark)
  - [Possible Changes](#possible-changes)

---

## Environment Setup

1. Create a Conda environment:

```bash
conda create -n new-hadpo python=3.10 -y
conda activate new-hadpo
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

---
## File Structure

```
DPO-with-LVLMs/
├── POPE/
  └── Version_1                # POPE dataset folder with JSON files
├── llava_dpo.py               # DPO fine-tuning script
├── run_llava.sh               # Bash script to run fine-tuning
├── llava_pope_eval.py         # Evaluation script on POPE benchmark
├── run_pope_llava.sh          # Bash script to run POPE evaluation
├── evaluate.py                # to compute accuracy on POPE results
├── requirements.txt           # Python dependencies
├── results/                   # Default folder for evaluation outputs
└── logs/                      # TensorBoard logs for training
```

---

## Training LLaVA with DPO

Use `llava_dpo.py` along with `run_llava.sh` to fine-tune the model.

### Run Training

```bash
bash run_llava.sh
```

### Key Configuration

- `MODEL_NAME`: Base model (e.g., `llava-hf/llava-1.5-7b-hf`)  
- `DATASET_NAME`: Hugging Face dataset (e.g., `Eftekhar/HA-DPO-Dataset`)  
- `OUTPUT_DIR`: Directory to save checkpoints  
- `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`: Training hyperparameters  
- `BF16`, `USE_LORA`, `GRADIENT_CHECKPOINTING`: Training options  

> Logs are saved to `train_logs.txt` and TensorBoard logs are in `logs/`.

---

## Evaluating on POPE Benchmark

Use `llava_pope_eval.py` along with `run_pope_llava.sh` to evaluate **base** or **DPO-trained** models.

### Run Evaluation

```bash
bash run_pope_llava.sh
```

### Key Configuration

- `MODEL_TYPE`: `base` or `dpo`  
- `MODEL_NAME`: Base LLaVA model  
- `DPO_CHECKPOINT`: Path to DPO checkpoint (required if `MODEL_TYPE=dpo`)  
- `POPE_PATH`: Path to POPE dataset folder having JSON files 
- `COCO_PATH`: Path to COCO images (`val2014`)  
- `SET_NAME`: POPE split (`random`, `popular`, `adv`)  
- `OUTPUT_DIR`: Directory to save results  

> Logs are saved to `eval_logs.txt`, results are saved as JSONL in `OUTPUT_DIR`.


---

## Performance on POPE Benchmark

Use `evaluate.py`  to obtain performance on POPE benchmark.

### Run Evaluation

```bash
python evaluate.py
```
- Make sure to set the correct paths for `ans_file` (`results` folder) and `label_file` (`POPE/Version_1` folder) in `evaluate.py`.

## Evaluating on AMBER Benchmark

First you must run the `amber_setup.sh` script to properly pull the annotations and image data.  This will also install certain python packages required for the AMBER evaluation. All necessary files
will be downloaded to `data/AMBER`

Use `llava_pope_eval.py` with argument --amber with the relevant model to exercise the entire benchmark
on the model. Please use the help command of this script for documentation of arguments for running sub
portions of the script.

## Possible Changes
- Update models name in `run_llava.sh` and `run_pope_llava.sh` to use different LLaVA versions.
  - `"llava-hf/llava-1.5-13b-hf"`
  - `"llava-hf/llava-v1.6-mistral-7b-hf"`
 
- Adjust training hyperparameters in `run_llava.sh` as needed.
- Make sure to change output directories to avoid overwriting previous results.