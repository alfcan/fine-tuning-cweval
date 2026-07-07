# Preference-based Security Alignment via IPO on CWEval

This repository contains the complete implementation suite for aligning code generation models to security preferences using Identity Preference Optimization (IPO) on Python tasks from the CWEval benchmark, and evaluating transferability to unseen programming languages (JavaScript, C, C++, Go).

---

## Prerequisites & Installation

### Step 1: Clone the Repo & Setup Dependencies
Run the provided setup script to clone the CWEval repository, pull the official Docker execution container, and install python dependencies:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

Alternatively, install the required packages manually:
```bash
pip install -r requirements.txt
```

---

## Execution Workflow

Follow these steps sequentially to run the entire pilot study:

### Step 0: Model Inference
The pipeline scripts automatically launch and terminate a local, unquantized OpenAI-compatible inference server using `openai_server.py`. You do not need to manually configure LM Studio or vLLM.
- **Model**: `Qwen/Qwen3.5-2B` (hosted on Hugging Face, downloaded automatically)
- **API Endpoint**: `http://127.0.0.1:1234/v1` (managed automatically by background processes)

---

### Step 1: Phase 1 — Baseline Evaluation
Evaluate the untouched base model across all 5 programming languages in CWEval to establish the baseline:

```bash
python run_baseline.py \
  --model "openai/Qwen/Qwen3.5-2B" \
  --eval_path "results/baseline" \
  --api_base "http://localhost:1234/v1" \
  --api_key "sk-local-research" \
  --docker True \
  --num_proc 8
```

---

### Step 2: Phase 2 & 3 — Preference Pair Construction & Split
Query the model server at multiple temperatures to generate correct secure/vulnerable pairs on 25 selected Python tasks. If a task fails to yield balanced outputs, the script will trigger a prompt paraphrasing retry loop:

```bash
python build_preference_dataset.py \
  --model "openai/Qwen/Qwen3.5-2B" \
  --api_base "http://localhost:1234/v1" \
  --api_key "sk-local-research" \
  --n_samples 10 \
  --max_pairs_per_task 8 \
  --train_split 0.8 \
  --docker True
```
This produces `results/dataset/train_pairs.json` and `results/dataset/val_pairs.json` grouped by task (no task leakage).

---

### Step 3: Phase 4 — IPO Alignment Training
Train LoRA/PEFT adapters with TRL's `DPOTrainer` (using the IPO loss) across 3 independent seeds. This script runs fine-tuning in full precision (no quantization) on the Qwen 2B model and implements early stopping:

```bash
python train_ipo.py \
  --model_id "Qwen/Qwen3.5-2B" \
  --dataset_dir "results/dataset" \
  --seeds "42,123,456" \
  --epochs 3 \
  --batch_size 4 \
  --quant_4bit False
```
This saves adapter checkpoints and outputs loss curves under `results/checkpoints/seed_<seed>/`.

---

### Step 4: Phase 5 — Multilingual Evaluation
Merge the trained adapter weights with the base model, and run comparative evaluations on both in-distribution (Python) and out-of-distribution (JS, C, C++, Go) tasks. Results are split by known vs. novel CWE categories, and bootstrap confidence intervals are computed:

```bash
python run_evaluation.py \
  --model_id "openai/Qwen/Qwen3.5-2B" \
  --seeds "42,123,456" \
  --api_base "http://localhost:1234/v1" \
  --api_key "sk-local-research" \
  --docker False
```
This writes the aggregated stats to `results/eval_summary.json`.

---

### Step 5: Phase 6 — Report Generation
Analyze the compiled evaluation summary and generate the final report documenting baseline performance, base vs. tuned transferability, and limitations:

```bash
python analyze_results.py \
  --eval_summary "results/eval_summary.json" \
  --output_report "results/report.md"
```

The completed markdown report will be saved at `results/report.md`.

---

## Code Directory Map

- [setup_env.sh](setup_env.sh): Clones CWEval, pulls Docker container, and installs python dependencies.
- [run_baseline.py](run_baseline.py): Measures untouched model correctness and security rates.
- [build_preference_dataset.py](build_preference_dataset.py): Explores generations, paraphrases prompts, deduplicates, and splits DPO preference pairs.
- [train_ipo.py](train_ipo.py): Fine-tunes the QLoRA model using IPO DPO optimization.
- [run_evaluation.py](run_evaluation.py): Merges adapters and runs final evaluation pipelines using bootstrap metrics.
- [analyze_results.py](analyze_results.py): Formats final metrics tables and produces a research report.
- [requirements.txt](requirements.txt): Lists Python libraries required for this project.
