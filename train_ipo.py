#!/usr/bin/env python3
"""
train_ipo.py - Phase 4: Training with IPO on CWEval preference pairs
Loads the train/val preference datasets.
Configures LoRA + quantization (QLoRA) for Qwen3 Coder.
Runs DPOTrainer with loss_type="ipo" across multiple independent seeds.
Monitors validation loss and applies early stopping.
Logs and plots loss curves.
"""

import os
import json
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: IPO Training Loop")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B", help="Hugging Face model ID")
    parser.add_argument("--dataset_dir", type=str, default="results/dataset", help="Path to train/val pairs")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated random seeds for training")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--beta", type=float, default=0.5, help="IPO/DPO beta parameter (controls preference strength)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Max number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--quant_4bit", type=str, default="False", choices=["True", "False"], help="Load model in 4-bit QLoRA")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    return parser.parse_args()

def load_json_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Expected columns: prompt, chosen, rejected
    return Dataset.from_list(data)

def plot_loss_curves(log_history, output_path, seed):
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        step = entry.get("step")
        if "loss" in entry and step is not None:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and step is not None:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label="Train Loss", color="blue", alpha=0.7)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color="red", marker='o')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"IPO Training Curves (Seed {seed})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    dataset_path = Path(args.dataset_dir)
    train_file = dataset_path / "train_pairs.json"
    val_file = dataset_path / "val_pairs.json"

    if not train_file.exists() or not val_file.exists():
        print(f"Error: Datasets not found at {dataset_path}. Please run build_preference_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading datasets...")
    train_dataset = load_json_dataset(train_file)
    val_dataset = load_json_dataset(val_file)
    print(f"Loaded {len(train_dataset)} train pairs and {len(val_dataset)} val pairs.")

    # Detect device
    if torch.cuda.is_available():
        device_map = "auto"
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_fp16 = not torch.cuda.is_bf16_supported()
        use_bf16 = torch.cuda.is_bf16_supported()
    elif torch.backends.mps.is_available():
        device_map = None  # We'll move manually
        model_dtype = torch.float32  # MPS doesn't support fp16/bf16 training reliably
        use_fp16 = False
        use_bf16 = False
        print("Detected Apple Silicon (MPS). Using float32 for training.")
    else:
        device_map = None
        model_dtype = torch.float32
        use_fp16 = False
        use_bf16 = False

    bnb_config = None
    if args.quant_4bit == "True":
        if not torch.cuda.is_available():
            print("Warning: 4-bit quantization requires CUDA. Skipping QLoRA on this device.")
        else:
            print("Configuring QLoRA 4-bit bitsandbytes...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True
            )

    # 5. Loop over independent seeds
    for seed in seeds:
        print(f"\n==========================================")
        print(f"Starting Training for Seed {seed}")
        print(f"==========================================")
        
        set_seed(seed)
        seed_output_dir = Path(args.output_dir) / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            # Use a dedicated pad token instead of eos_token to preserve EOS semantics during generation
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            print("Added dedicated <|pad|> token to tokenizer.")

        # Load Model
        print(f"Loading base model {args.model_id}...")
        load_kwargs = dict(
            trust_remote_code=True,
            dtype=model_dtype,
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)

        # Resize embeddings if we added a new pad token
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))

        # Move to MPS if needed
        if torch.backends.mps.is_available() and device_map is None:
            model = model.to("mps")

        if args.quant_4bit == "True" and bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        # Configure PEFT / LoRA
        # Qwen models typically use q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Setup training arguments
        # DPOConfig inherits from TrainingArguments
        training_args = DPOConfig(
            output_dir=str(seed_output_dir),
            loss_type="ipo",
            beta=args.beta,  # IPO: target offset = 1/(2*beta). beta=0.5 → target=1.0 (reachable)
            label_smoothing=0.1,  # Regularizes preference signal on small datasets
            max_length=1024,  # Increased from 512 to avoid truncating code completions
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,  # Saves significant memory
            learning_rate=args.learning_rate,
            weight_decay=0.05,  # L2 regularization to prevent overfitting
            num_train_epochs=args.epochs,
            eval_strategy="epoch",  # Epoch-based eval is more stable on tiny datasets
            save_strategy="epoch",
            logging_steps=1,  # Log every step for detailed monitoring
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_pin_memory=False,  # Not supported on MPS
            remove_unused_columns=False,  # Important for DPOTrainer
            seed=seed,
            report_to="none"
        )

        # Setup DPOTrainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Set to None to automatically use base model with adapter disabled
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
        )

        print("Training in progress...")
        train_result = trainer.train()

        # Save the best PEFT adapter checkpoint
        best_checkpoint_dir = seed_output_dir / "best_model"
        trainer.save_model(str(best_checkpoint_dir))
        print(f"Best adapter checkpoints saved to {best_checkpoint_dir}")

        # Save training logs history
        log_history = trainer.state.log_history
        with open(seed_output_dir / "log_history.json", "w") as f:
            json.dump(log_history, f, indent=2)

        # Plot training curves
        plot_path = seed_output_dir / "loss_curves.png"
        plot_loss_curves(log_history, plot_path, seed)
        print(f"Loss curves plot saved to {plot_path}")

        # Clean up model to free up memory before next seed
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n=== All Seeds Completed Successfully! ===")

if __name__ == "__main__":
    main()
