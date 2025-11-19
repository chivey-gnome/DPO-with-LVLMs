"""
Fine-tune a PaliGemma model using Calibrated Direct Preference Optimization (Cal-DPO)
on a multimodal preference dataset (images + questions + chosen/rejected answers).
"""

import sys, os

# ------------------------------------------------
# Add parent directory (DPO-with-LVLMs/) to Python path
# ------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import warnings
warnings.filterwarnings("ignore")

import argparse
from io import BytesIO
import base64
from PIL import Image

import torch
from datasets import load_dataset, load_from_disk, features
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig
from trainers.caldpo import CalDPOTrainer   # your existing CalDPO trainer
from trl import DPOConfig

# Optional: JSON logger
from dpo_json_logger import DPOJSONLoggerCallback


# ============================================================
# 1. Argument Parser
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train PaliGemma using Cal-DPO")

    parser.add_argument("--dataset_name", type=str, default="Eftekhar/HA-DPO-Dataset")
    parser.add_argument("--on_disk_data_set", action="store_true")

    parser.add_argument("--model_name", type=str, default="google/paligemma-3b-pt-224")
    parser.add_argument("--output_dir", type=str, default="./paligemma-caldpo-output")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=32)

    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--log_steps", type=int, default=10)

    parser.add_argument("--json_log_name", type=str,
                        default="caldpo_paligemma_logs.json",
                        help="JSON file to save training step logs")

    return parser.parse_args()


# ============================================================
# 2. PaliGemma Chat Template
# ============================================================
def set_paligemma_template(processor):
    processor.chat_template = """
{% for message in messages %}
{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}
{% for item in message['content'] %}
{% if item['type'] == 'text' %}{{ item['text'] }}
{% elif item['type'] == 'image' %}<image>
{% endif %}
{% endfor %}
{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}
{% endfor %}
"""
    return processor


# ============================================================
# 3. Format dataset example for PaliGemma
# ============================================================
def format_paligemma(example, processor):
    img = example["image"]

    # Ensure PIL Image
    if not isinstance(img, Image.Image):
        if isinstance(img, str):
            if os.path.exists(img):
                img = Image.open(img)
            else:
                img = Image.open(BytesIO(base64.b64decode(img)))
        else:
            img = Image.fromarray(img)
    img = img.convert("RGB")

    images = [img]

    # User prompt
    prompt = [{
        "role": "user",
        "content": [{"type": "image"}] + [{"type": "text", "text": example["question"]}]
    }]
    chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
    rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]

    prompt_text = processor.apply_chat_template(prompt, tokenize=False)
    chosen_text = processor.apply_chat_template(chosen, tokenize=False)
    rejected_text = processor.apply_chat_template(rejected, tokenize=False)

    return {
        "images": images,
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


# ============================================================
# 4. Load & prepare dataset
# ============================================================
def prepare_dataset(dataset_name, processor, num_proc=16, ondisk=False):
    if ondisk:
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name, split="train")

    dataset = dataset.map(
        lambda ex: format_paligemma(ex, processor),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    # Ensure proper image decoding
    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)

    return dataset


# ============================================================
# 5. TRAIN FUNCTION (Cal-DPO)
# ============================================================
def train(args):

    # -------------------------------
    # Save inside saved_models/
    # -------------------------------
    args.output_dir = os.path.join("saved_models",
                                   os.path.basename(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    processor = set_paligemma_template(processor)

    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_name, processor,
                              args.num_proc, args.on_disk_data_set)

    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto"
    )

    # ---------------------------------
    # DPO (Cal-DPO) Training Arguments
    # ---------------------------------
    training_args = DPOConfig(
        output_dir=args.output_dir,
        bf16=args.bf16,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        beta=args.beta,       # Cal-DPO still uses beta
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        dataset_num_proc=args.num_proc,
        dataloader_num_workers=args.num_workers,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.log_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["tensorboard"],   # no JSON here
    )

    # ---------------------------------
    # Optional LoRA config for PaliGemma
    # ---------------------------------
    peft_config = (
        LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LLM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ) if args.use_lora else None
    )

    print("Initializing CalDPOTrainer...")
    trainer = CalDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    # ---------------------------------
    # JSON Logging
    # ---------------------------------
    json_log_path = os.path.join(args.output_dir, args.json_log_name)
    trainer.add_callback(DPOJSONLoggerCallback(json_log_path))

    # ---------------------------------
    # Auto Resume
    # ---------------------------------
    checkpoint = None
    ckpts = [d for d in os.listdir(args.output_dir)
             if d.startswith("checkpoint-")]
    if ckpts:
        latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
        checkpoint = os.path.join(args.output_dir, latest)
        print(f"Resuming from: {checkpoint}")
    else:
        print("Starting fresh training")

    # ---------------------------------
    # Train
    # ---------------------------------
    trainer.train(resume_from_checkpoint=checkpoint)
    print("Training complete — saved to:", args.output_dir)


# ============================================================
if __name__ == "__main__":
    args = parse_args()
    train(args)
