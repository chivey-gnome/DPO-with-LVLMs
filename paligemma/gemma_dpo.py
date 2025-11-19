"""
Fine-tune a PaliGemma model using Direct Preference Optimization (DPO).
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import json
from io import BytesIO
import base64

from PIL import Image
import torch
from datasets import load_dataset, load_from_disk, features
from transformers import AutoModelForVision2Seq, AutoProcessor
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

# JSON logger import
from json_logger import DPOJSONLoggerCallback

import tensorboard
print("TensorBoard:", tensorboard.__version__)


# --------------------------------------------------------
# 1. Argument Parser
# --------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train PaliGemma using DPO")

    parser.add_argument("--dataset_name", type=str, default="omarftt010/HA-Perturb-DPO-Dataset")
    parser.add_argument("--on_disk_data_set", action="store_true")

    parser.add_argument("--model_name", type=str, default="google/paligemma-3b-pt-224")
    parser.add_argument("--output_dir", type=str, default="./paligemma-dpo-output")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=32)

    parser.add_argument("--num_proc", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=32)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--log_steps", type=int, default=10)

    # ⭐ NEW ARGUMENT: json log file name
    parser.add_argument(
        "--json_log_name",
        type=str,
        default="dpo_step_logs.json",
        help="Filename for saving step-by-step DPO metrics in JSON."
    )

    return parser.parse_args()


# --------------------------------------------------------
# 2. Chat Template for PaliGemma
# --------------------------------------------------------
def set_paligemma_template(processor):
    processor.chat_template = """
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}
{% endif %}
{% for message in messages %}
{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}
{% for item in message['content'] %}
{% if item['type'] == 'text' %}{{ item['text'] }}
{% elif item['type'] == 'image' %}<image>
{% endif %}
{% endfor %}
{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}
{% endfor %}
{% if add_generation_prompt %}ASSISTANT: {% endif %}
"""
    return processor


# --------------------------------------------------------
# 3. Format Dataset Example for PaliGemma
# --------------------------------------------------------
def format_paligemma(example, processor):
    img = example["image"]

    if not isinstance(img, Image.Image):
        if isinstance(img, str):
            if os.path.exists(img):
                img = Image.open(img)
            else:
                img = example["image_data"]
                img = Image.open(BytesIO(base64.b64decode(img)))
        else:
            img = Image.fromarray(img)

    img = img.convert("RGB")

    max_size = max(
        getattr(processor.image_processor.size, "height", 224),
        getattr(processor.image_processor.size, "width", 224)
    )
    img.thumbnail((max_size, max_size))

    images = [img]

    prompt = [{"role": "user", "content": [{"type": "image"}] + [{"type": "text", "text": example["question"]}]}]
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


# --------------------------------------------------------
# 4. Prepare Dataset
# --------------------------------------------------------
def prepare_dataset(dataset_name, processor, num_proc=32, ondisk=False):
    dataset = (
        load_dataset(dataset_name, split="train")
        if not ondisk
        else load_from_disk(dataset_name)
    )

    dataset = dataset.map(
        lambda ex: format_paligemma(ex, processor),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)

    return dataset


# --------------------------------------------------------
# 5. Trainer
# --------------------------------------------------------
def train(args):
    processor = AutoProcessor.from_pretrained(args.model_name, do_image_splitting=False)
    processor = set_paligemma_template(processor)

    dataset = prepare_dataset(args.dataset_name, processor, args.num_proc, args.on_disk_data_set)

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    #  Updated: JSON logging enabled
    training_args = DPOConfig(
        output_dir=args.output_dir,
        bf16=args.bf16,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        beta=0.1,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=os.path.join(args.output_dir, "logs"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        dataset_num_proc=args.num_proc,
        dataloader_num_workers=args.num_workers,
        logging_steps=args.log_steps,
        report_to=["tensorboard"],   # JSON logging added
    )

    peft_config = (
        LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            bias="none",
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"
            ],
        ) if args.use_lora else None
    )

    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    #  Add custom JSON logger
    json_log_path = os.path.join(args.output_dir, args.json_log_name)
    trainer.add_callback(DPOJSONLoggerCallback(json_log_path))

    # Redirect output_dir inside saved_models
    args.output_dir = os.path.join("saved_models", os.path.basename(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    # Resume checkpoint if exists
    checkpoint = None
    checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        checkpoint = os.path.join(args.output_dir, latest)
        print(f" Resuming from checkpoint: {checkpoint}")
    else:
        print(" Starting fresh training")

    trainer.train(resume_from_checkpoint=checkpoint)
    print("Training finished. Saved to:", args.output_dir)


# --------------------------------------------------------
# 6. Main
# --------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
