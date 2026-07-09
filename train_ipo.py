#!/usr/bin/env python3
"""
train_ipo.py - Phase 4: IPO training on CWEval preference pairs (optimized)

Key fixes vs previous version:
1. Effective batch size reduced (was 32 on ~60 pairs -> ~2 optimizer steps/epoch,
   degenerate LR schedule). Now ~15 steps/epoch with a meaningful scheduler.
2. Best-model selection & early stopping now use eval_rewards/accuracies
   (IPO eval_loss is nearly flat at 1.0 and carries no signal).
3. No new pad token / no resize_token_embeddings (embedding rows are NOT trained
   by LoRA -> random pad embedding + vocab mismatch with the base model at eval
   time). Padding is masked in the DPO/IPO loss, so pad = eos is safe.
4. Dataset converted to conversational format so DPOTrainer applies the SAME
   chat template used at generation/eval time (raw-string training creates a
   train/eval distribution mismatch).
5. Pre-training dataset audit: label sanity + chosen/rejected length-bias report
   (train logs showed chosen much longer than rejected in train, reversed in
   val -> possible length shortcut instead of security signal).
6. label_smoothing removed (not used by the IPO loss in TRL).
7. gradient_checkpointing with use_reentrant=False (required with PEFT).
8. max_prompt_length set explicitly; save_total_limit to avoid disk bloat.

IMPORTANT: --model_id MUST be the exact same checkpoint used to generate the
preference pairs (on-policy) and used later for evaluation.
"""

import sys
import json
import inspect
import argparse
import statistics
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: IPO Training Loop (optimized)")
    p.add_argument(
        "--model_id",
        type=str,
        required=True,  # no silent default: must match the pair-generation model
        help="HF model ID. MUST be the exact model used to generate the pairs.",
    )
    p.add_argument("--dataset_dir", type=str, default="results/dataset")
    p.add_argument("--output_dir", type=str, default="results/checkpoints")
    p.add_argument("--seeds", type=str, default="42,123,456")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.5,
                   help="IPO beta. Target margin = 1/(2*beta); 0.5 -> 1.0")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=8,
                   help="Max epochs; early stopping on eval accuracy cuts it short.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2,
                   help="Effective batch = batch_size * this. Keep SMALL on tiny datasets.")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--quant_4bit", action="store_true",
                   help="Enable QLoRA 4-bit loading (CUDA only).")
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--force_gradient_checkpointing", action="store_true",
                   help="Force activation of gradient checkpointing even on CUDA.")
    p.add_argument("--skip_audit", action="store_true",
                   help="Skip the pre-training dataset audit (not recommended).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data loading + audits
# --------------------------------------------------------------------------- #
def load_pairs(file_path: Path) -> list[dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def to_conversational(pairs: list[dict]) -> Dataset:
    """
    Convert {prompt, chosen, rejected} raw strings into TRL conversational
    format so DPOTrainer applies the model's chat template — the SAME format
    used at generation and evaluation time. Training on raw strings while
    generating/evaluating with a chat template is a distribution mismatch
    that can silently null the adapter's effect.
    """
    rows = []
    for ex in pairs:
        rows.append(
            {
                "prompt": [{"role": "user", "content": ex["prompt"]}],
                "chosen": [{"role": "assistant", "content": ex["chosen"]}],
                "rejected": [{"role": "assistant", "content": ex["rejected"]}],
            }
        )
    return Dataset.from_list(rows)


def audit_dataset(name: str, pairs: list[dict], tokenizer) -> None:
    """
    Pre-training sanity checks. Catches the two failure modes suggested by the
    previous runs' logs:
      a) label problems (identical/empty chosen-rejected),
      b) systematic length bias between chosen and rejected
         (IPO does not length-normalize -> the model can learn a length
          shortcut instead of a security preference).
    """
    problems = 0
    len_deltas = []
    for i, ex in enumerate(pairs):
        c, r = ex.get("chosen", ""), ex.get("rejected", "")
        if not c.strip() or not r.strip():
            print(f"  [WARN] {name}[{i}]: empty chosen or rejected")
            problems += 1
        if c.strip() == r.strip():
            print(f"  [WARN] {name}[{i}]: chosen == rejected")
            problems += 1
        len_deltas.append(
            len(tokenizer(c, add_special_tokens=False).input_ids)
            - len(tokenizer(r, add_special_tokens=False).input_ids)
        )

    mean_d = statistics.mean(len_deltas)
    med_d = statistics.median(len_deltas)
    pos = sum(d > 0 for d in len_deltas)
    print(
        f"  [{name}] pairs={len(pairs)} | token-length delta (chosen - rejected): "
        f"mean={mean_d:+.1f}, median={med_d:+.1f}, chosen-longer in {pos}/{len(len_deltas)}"
    )
    if abs(mean_d) > 30:
        print(
            f"  [ALERT] {name}: strong systematic length bias between chosen and "
            f"rejected. IPO can exploit this as a shortcut. Consider length-"
            f"matching pairs (pair each chosen with the rejected closest in length)."
        )
    if problems:
        print(f"  [ALERT] {name}: {problems} malformed pairs found — fix before training.")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_curves(log_history, output_path, seed):
    def series(key):
        xs, ys = [], []
        for e in log_history:
            if key in e and "step" in e:
                xs.append(e["step"])
                ys.append(e[key])
        return xs, ys

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ts, tl = series("loss")
    es, el = series("eval_loss")
    axes[0].plot(ts, tl, label="train loss", alpha=0.7)
    if el:
        axes[0].plot(es, el, "o-", label="eval loss")
    axes[0].set_title(f"IPO loss (seed {seed})")
    axes[0].set_xlabel("step")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # The metric that actually matters on tiny IPO datasets:
    tas, ta = series("rewards/accuracies")
    eas, ea = series("eval_rewards/accuracies")
    if ta:
        axes[1].plot(tas, ta, label="train reward acc", alpha=0.7)
    if ea:
        axes[1].plot(eas, ea, "o-", label="eval reward acc")
    axes[1].axhline(0.5, color="gray", linestyle=":", label="chance (0.5)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Reward accuracy (seed {seed})")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    dataset_path = Path(args.dataset_dir)
    train_file = dataset_path / "train_pairs.json"
    val_file = dataset_path / "val_pairs.json"
    if not train_file.exists() or not val_file.exists():
        print(f"Error: datasets not found in {dataset_path}.", file=sys.stderr)
        sys.exit(1)

    train_pairs = load_pairs(train_file)
    val_pairs = load_pairs(val_file)
    print(f"Loaded {len(train_pairs)} train / {len(val_pairs)} val pairs.")

    steps_per_epoch = max(
        1, len(train_pairs) // (args.batch_size * args.gradient_accumulation_steps)
    )
    print(
        f"Effective batch = {args.batch_size * args.gradient_accumulation_steps} "
        f"-> ~{steps_per_epoch} optimizer steps/epoch."
    )
    if steps_per_epoch < 8:
        print(
            "[ALERT] Fewer than 8 optimizer steps per epoch: reduce batch_size or "
            "gradient_accumulation_steps, otherwise the LR schedule is degenerate."
        )

    eval_steps = max(1, steps_per_epoch // 2)
    print(f"Evaluating / saving every {eval_steps} steps (~0.5 epochs).")

    # ---- device / dtype / TF32 / diagnostics --------------------------------
    if torch.cuda.is_available():
        # Enable TF32 for speedup on Ampere/Ada/Blackwell architectures
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        device_map = "auto"
        bf16 = torch.cuda.is_bf16_supported()
        model_dtype = torch.bfloat16 if bf16 else torch.float16
        use_bf16, use_fp16 = bf16, not bf16
        
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        total_mem = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
        print("CUDA Diagnostics:")
        print(f"  Device: {device_name} (index {device_idx})")
        print(f"  Total VRAM: {total_mem:.2f} GB")
        print(f"  BF16 supported: {bf16}")
        print(f"  TF32 enabled: True (allow_tf32 = True)")
    elif torch.backends.mps.is_available():
        device_map, model_dtype = None, torch.float32
        use_bf16 = use_fp16 = False
        print("Apple Silicon (MPS): float32 training.")
    else:
        device_map, model_dtype = None, torch.float32
        use_bf16 = use_fp16 = False
        print("CPU: float32 training.")

    # Determine gradient checkpointing status
    if args.force_gradient_checkpointing:
        use_grad_checkpointing = True
        grad_checkpointing_reason = "forced via CLI flag"
    elif not torch.cuda.is_available():
        use_grad_checkpointing = True
        grad_checkpointing_reason = "active by default on CPU/MPS"
    elif args.quant_4bit:
        use_grad_checkpointing = True
        grad_checkpointing_reason = "active by default with 4-bit quantization"
    else:
        use_grad_checkpointing = False
        grad_checkpointing_reason = "disabled by default on CUDA (non-quantized)"

    print(f"Gradient Checkpointing: {'ON' if use_grad_checkpointing else 'OFF'} ({grad_checkpointing_reason})")

    bnb_config = None
    if args.quant_4bit:
        if not torch.cuda.is_available():
            print("Warning: 4-bit quantization requires CUDA. Ignoring --quant_4bit.")
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_use_double_quant=True,
            )

    # ---- tokenizer (once; identical across seeds) ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # SAFE with DPO/IPO: padding tokens are masked out of the loss, and we
        # avoid resize_token_embeddings, which LoRA cannot train and which
        # desynchronizes vocab size between adapter and base model at eval time.
        tokenizer.pad_token = tokenizer.eos_token
        print("pad_token was None -> using eos as pad (no vocab resize).")

    # ---- dataset audit + conversion -----------------------------------------
    if not args.skip_audit:
        print("Auditing datasets (labels + length bias)...")
        audit_dataset("train", train_pairs, tokenizer)
        audit_dataset("val", val_pairs, tokenizer)

    train_dataset = to_conversational(train_pairs)
    val_dataset = to_conversational(val_pairs)

    # ---- per-seed training ---------------------------------------------------
    for seed in seeds:
        print(f"\n========== Seed {seed} ==========")
        set_seed(seed)
        seed_dir = Path(args.output_dir) / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        load_kwargs = dict(trust_remote_code=True, dtype=model_dtype)
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
        # keep model/generation config aligned with tokenizer, silences the
        # PAD/EOS mismatch warning without touching the vocab:
        model.config.pad_token_id = tokenizer.pad_token_id

        if torch.backends.mps.is_available() and device_map is None:
            model = model.to("mps")
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Build the config as a dict, then filter out any kwarg not supported
        # by the installed TRL version (param names move across TRL releases,
        # e.g. max_prompt_length). Unsupported keys are dropped with a warning
        # instead of crashing.
        config_kwargs = dict(
            output_dir=str(seed_dir),
            loss_type="ipo",
            beta=args.beta,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=use_grad_checkpointing,
            learning_rate=args.learning_rate,
            weight_decay=0.05,
            num_train_epochs=args.epochs,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=2,
            logging_steps=1,
            warmup_steps=max(2, steps_per_epoch // 3),  # step-based, not ratio:
            lr_scheduler_type="cosine",                 # ratio breaks on tiny runs
            # --- the fix that matters most: select/stop on a metric that MOVES.
            # IPO eval_loss sits ~flat at 1.0 and is uninformative; reward
            # accuracy on held-out pairs is the actual generalization signal.
            load_best_model_at_end=True,
            metric_for_best_model="eval_rewards/accuracies",
            greater_is_better=True,
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_pin_memory=torch.cuda.is_available(),
            remove_unused_columns=False,
            seed=seed,
            report_to="none",
        )
        if use_grad_checkpointing:
            config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        supported = set(inspect.signature(DPOConfig.__init__).parameters)
        dropped = sorted(k for k in config_kwargs if k not in supported)
        if dropped:
            print(
                f"[WARN] Installed TRL's DPOConfig does not support: {dropped}. "
                f"Dropping them. Check your TRL version (pip show trl) — "
                f"if 'max_prompt_length' was dropped, long prompts will only be "
                f"truncated by max_length."
            )
        training_args = DPOConfig(
            **{k: v for k, v in config_kwargs.items() if k in supported}
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # PEFT: reference = base model with adapter disabled
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                )
            ],
        )

        print("Training...")
        trainer.train()

        best_dir = seed_dir / "best_model"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))  # ship tokenizer with adapter
        print(f"Best adapter saved to {best_dir}")

        with open(seed_dir / "log_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        plot_curves(trainer.state.log_history, seed_dir / "curves.png", seed)

        # summary line for quick cross-seed comparison
        best_eval_acc = max(
            (e["eval_rewards/accuracies"] for e in trainer.state.log_history
             if "eval_rewards/accuracies" in e),
            default=None,
        )
        print(f"Seed {seed}: best eval reward accuracy = {best_eval_acc}")

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n=== All seeds completed ===")


if __name__ == "__main__":
    main()