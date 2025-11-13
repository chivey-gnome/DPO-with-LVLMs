"""
Fine-tune a LLaVA model using Direct Preference Optimization (DPO) on a multimodal dataset.

This script:
- Loads the HA-DPO dataset from Hugging Face Hub
- Formats data for LLaVA input (image + question + chosen/rejected responses)
- Prepares model and LoRA configuration
- Runs DPO training with TensorBoard logging
- Automatically resumes from last checkpoint if training is interrupted
"""

import warnings
warnings.filterwarnings("ignore")

## Libraries
import argparse
import os
from PIL import Image
import torch, transformers
from datasets import load_dataset, features
from transformers import AutoModelForVision2Seq, AutoProcessor
import trl
from trl import DPOConfig #, DPOTrainer
from trainers.caldpo import CalDPOTrainer
import peft
from peft import LoraConfig
import tensorboard

print("TRL version:", trl.__version__)
print("PEFT version:", peft.__version__)
print("TensorBoard version:", tensorboard.__version__)



def parse_args():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description="Train LLaVA using DPO on multimodal preference data")

    parser.add_argument("--dataset_name", type=str, default="omarftt010/HA-Perturb-DPO-Dataset",
                        help="Name or path of the dataset to load from Hugging Face Hub.")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="Pretrained LLaVA model to fine-tune.")
    parser.add_argument("--output_dir", type=str, default="./llava-dpo-output",
                        help="Directory to save trained model and logs.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size for training.")
    parser.add_argument("--grad_accum_steps", type=int, default=32, help="Gradient accumulation steps.")
    parser.add_argument("--num_proc", type=int, default=32, help="Number of CPU processes for dataset mapping.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of dataloader workers.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision training if supported.")
    parser.add_argument("--log_steps", type=int, default=10, help="Logging interval during training.")
    parser.add_argument("--use_lora", action="store_true", help="Apply LoRA fine-tuning to reduce memory use.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory.")
    
    return parser.parse_args()


def format_llava(example, processor):
    """
    Format dataset examples for LLaVA model input.
    
    Args:
        example (dict): Single example containing image, question, chosen, and rejected.
        processor (AutoProcessor): Processor used to format inputs for LLaVA.

    Returns:
        dict: Processed example with prompt, chosen, rejected, and resized images.
    """
    # Handle image input - the dataset has 'image' column, not 'image_path'
    # Images are already loaded as PIL Images
    img = example["image"]
    
    # Ensure it's a PIL Image
    if not isinstance(img, Image.Image):
        if hasattr(img, 'convert'):  # It might be in a different format
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unexpected image type: {type(img)}")
    
    images = [img]

    # Build chat messages
    prompt = [{"role": "user", "content": [{"type": "image"} for _ in images] +
                                         [{"type": "text", "text": example["question"]}]}]
    chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
    rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]

    # Apply chat templates
    prompt_text = processor.apply_chat_template(prompt, tokenize=False)
    chosen_text = processor.apply_chat_template(chosen, tokenize=False)
    rejected_text = processor.apply_chat_template(rejected, tokenize=False)

    # Resize images to avoid OOM
    max_size = processor.image_processor.size.get("max_size", 1024)
    resized_images = []
    for img in images:
        img.thumbnail((max_size, max_size))
        resized_images.append(img)

    return {
        "images": resized_images,
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text
    }


def prepare_dataset(dataset_name, processor, num_proc=32):
    """
    Load and process the dataset for Cal-DPO training.
    
    Args:
        dataset_name (str): Hugging Face dataset identifier or local path.
        processor (AutoProcessor): Processor used for input formatting.
        num_proc (int): Number of processes for parallel mapping.
    
    Returns:
        Dataset: Processed dataset ready for training.
    """
    dataset = load_dataset(dataset_name, split="train")

    dataset = dataset.map(
        lambda ex: format_llava(ex, processor),
        remove_columns=dataset.column_names,
        num_proc=num_proc
    )

    # Ensure proper image decoding
    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)

    return dataset


def train(args):
    """
    Main training routine.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    print("Loading processor and dataset...")
    processor = AutoProcessor.from_pretrained(args.model_name, do_image_splitting=False)
    dataset = prepare_dataset(args.dataset_name, processor, args.num_proc)
    print("==============================================================")
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto"
    )

     # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    print("==============================================================")
    print("Setting up training configuration...")
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
        report_to="tensorboard",
    )

    # Optional LoRA config
    peft_config = LoraConfig(
    r=128,  
    lora_alpha=256,  # 2*r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=False, 
    
    target_modules=[
        # Language model (definitely include)
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        
    ],
) if args.use_lora else None
    print("==============================================================")
    print("Initializing DPO trainer...")
    trainer = CalDPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    print("==============================================================")
    
    # Check for existing checkpoints and resume if found
    checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by checkpoint number and get the latest
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint = os.path.join(args.output_dir, latest_checkpoint)
            print(f"🔄 Resuming from checkpoint: {checkpoint}")
        else:
            print("🆕 No checkpoint found, starting fresh training")
    else:
        print("🆕 Starting fresh training")
    
    print("==============================================================")
    print("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    print("==============================================================")
    print("Training completed! Model saved at:", args.output_dir)
    print("==============================================================")


if __name__ == "__main__":
    args = parse_args()
    train(args)