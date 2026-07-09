#!/usr/bin/env python3
"""
recover_preference_pairs.py
Harvests all evaluated completions from results/preference_gen, constructs new pairs for
all tasks using length-matching (delta <= 50 tokens), and merges them into the train/validation datasets.
"""

import os
import re
import sys
import json
import random
import difflib
from pathlib import Path
import tiktoken

# Disable parallel tokenization and fork safety locks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

def normalize_code(code):
    # Strip comments and docstrings, then compress whitespace to identify duplicates
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code).strip()
    return code

def is_too_similar(new_code, existing_codes, threshold=0.95):
    norm_new = normalize_code(new_code)
    for ext_code in existing_codes:
        norm_ext = normalize_code(ext_code)
        if norm_new == norm_ext:
            return True
        ratio = difflib.SequenceMatcher(None, norm_new, norm_ext).ratio()
        if ratio >= threshold:
            return True
    return False

def main():
    dataset_dir = Path("results/dataset").resolve()
    cweval_dir = Path("CWEval").resolve()
    pref_gen_dir = Path("results/preference_gen").resolve()
    
    max_pairs_per_task = 8
    similarity_threshold = 0.95
    max_token_delta = 50

    # Initialize tiktoken
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error initializing tiktoken: {e}", file=sys.stderr)
        sys.exit(1)

    def count_tokens(text):
        return len(encoding.encode(text))

    if not cweval_dir.exists():
        print(f"CWEval repository not found at {cweval_dir}.", file=sys.stderr)
        sys.exit(1)

    python_bench = cweval_dir / "benchmark" / "core" / "py"
    if not python_bench.exists():
        print(f"Python benchmarks not found under {python_bench}", file=sys.stderr)
        sys.exit(1)

    # 1. Identify all Python tasks
    task_files = sorted(list(python_bench.glob("cwe_*_task.py")))
    all_tasks = []
    task_prompts = {}
    prompt_to_task_id = {}
    for task_file in task_files:
        task_name = task_file.name.replace("_task.py", "")
        if task_name in ["cwe_918_0", "cwe_918_1"]:
            continue
        task_id = f"core/py/{task_name}"
        with open(task_file, "r") as f:
            content = f.read()
        task_prompts[task_id] = content
        prompt_to_task_id[content] = task_id
        
        all_tasks.append({
            "task_id": task_id,
            "task_name": task_name,
            "py_file": task_file,
        })

    # 2. Re-establish train/val task assignment deterministically
    task_ids = [t["task_id"] for t in all_tasks]
    random.seed(1337)
    random.shuffle(task_ids)
    split_idx = int(len(task_ids) * 0.8)
    train_task_ids = set(task_ids[:split_idx])
    val_task_ids = set(task_ids[split_idx:])

    # 3. Load existing pairs to harvest completions from them
    train_file = dataset_dir / "train_pairs.json"
    val_file = dataset_dir / "val_pairs.json"

    train_pairs = []
    val_pairs = []
    if train_file.exists():
        with open(train_file, "r") as f:
            train_pairs = json.load(f)
    if val_file.exists():
        with open(val_file, "r") as f:
            val_pairs = json.load(f)

    # Collect existing completions from existing pairs
    existing_completions = {tid: {"secure": [], "vulnerable": []} for tid in task_ids}
    for pair in train_pairs + val_pairs:
        prompt_content = pair["prompt"]
        if prompt_content in prompt_to_task_id:
            tid = prompt_to_task_id[prompt_content]
            existing_completions[tid]["secure"].append(pair["chosen"])
            existing_completions[tid]["vulnerable"].append(pair["rejected"])

    # 4. Harvest all evaluated completions from results/preference_gen
    harvested_completions = {tid: {"secure": [], "vulnerable": []} for tid in task_ids}

    # Helper to check evaluation results and extract completions
    def process_res_all(res_all_path, generated_dir_base):
        try:
            with open(res_all_path, "r") as f:
                res_data = json.load(f)
        except Exception as e:
            print(f"Error reading {res_all_path}: {e}")
            return

        for key, value in res_data.items():
            if "/py/" in key and key.endswith("_test.py"):
                task_name = key.split("/")[-1].replace("_test.py", "")
                tid = f"core/py/{task_name}"
                if tid not in harvested_completions:
                    continue

                functional_list = value.get("functional", [])
                secure_list = value.get("secure", [])

                for idx, (func, sec) in enumerate(zip(functional_list, secure_list)):
                    if not func:
                        continue
                    
                    # Try generated_{idx} first, fallback to generated_0
                    raw_file = generated_dir_base / f"generated_{idx}" / "core" / "py" / f"{task_name}_raw.py"
                    if not raw_file.exists():
                        raw_file = generated_dir_base / "generated_0" / "core" / "py" / f"{task_name}_raw.py"
                        if not raw_file.exists():
                            continue
                    
                    try:
                        with open(raw_file, "r") as f_raw:
                            content = f_raw.read()
                        
                        cat = "secure" if sec else "vulnerable"
                        if content not in harvested_completions[tid][cat]:
                            harvested_completions[tid][cat].append(content)
                    except Exception as e:
                        print(f"Error reading raw file {raw_file}: {e}")

    # Scan primary models
    models = ["claude-3.5-sonnet", "claude-3.7-sonnet", "claude-sonnet-4", "claude-sonnet-4.5", "gpt-4-turbo", "gpt-4.1", "gpt-4o", "gpt-5.2", "llama_3", "llama_3.1", "llama_3.3", "llama_4"]
    for model in models:
        model_dir = pref_gen_dir / model
        if not model_dir.exists():
            continue
        for run_dir in sorted(model_dir.glob("run_*")):
            res_all_path = run_dir / model / "res_all.json"
            if res_all_path.exists():
                process_res_all(res_all_path, run_dir / model)

    # Scan retry directories
    for retry_dir in sorted(pref_gen_dir.glob("retry_*")):
        res_all_path = retry_dir / "res_all.json"
        if res_all_path.exists():
            process_res_all(res_all_path, retry_dir)

    # Scan temp directories
    for temp_dir in sorted(pref_gen_dir.glob("temp_*")):
        res_all_path = temp_dir / "res_all.json"
        if res_all_path.exists():
            process_res_all(res_all_path, temp_dir)

    # 5. Reconstruct all pairs for each task with length-matching
    reconstructed_pairs_by_task = {}

    print("\n=== Reconstructing Pairs with Length-Matching ===")
    for tid in task_ids:
        # Combine existing and harvested
        all_sec = existing_completions[tid]["secure"] + harvested_completions[tid]["secure"]
        all_vuln = existing_completions[tid]["vulnerable"] + harvested_completions[tid]["vulnerable"]

        # Deduplicate
        final_unique_sec = []
        for s in all_sec:
            if not is_too_similar(s, final_unique_sec, similarity_threshold):
                final_unique_sec.append(s)

        final_unique_vuln = []
        for v in all_vuln:
            if not is_too_similar(v, final_unique_vuln, similarity_threshold):
                final_unique_vuln.append(v)

        if not final_unique_sec or not final_unique_vuln:
            reconstructed_pairs_by_task[tid] = []
            print(f"Task {tid}: No pairs can be constructed (missing one or both categories)")
            continue

        # Count tokens for all unique candidates
        sec_tokens = [count_tokens(s) for s in final_unique_sec]
        vuln_tokens = [count_tokens(v) for v in final_unique_vuln]

        # For each secure completion, find the closest vulnerable completion
        candidate_pairs = []
        for s_idx, s_code in enumerate(final_unique_sec):
            s_len = sec_tokens[s_idx]
            
            best_v_idx = -1
            best_delta = float('inf')
            
            for v_idx, v_code in enumerate(final_unique_vuln):
                v_len = vuln_tokens[v_idx]
                delta = s_len - v_len
                abs_delta = abs(delta)
                
                if abs_delta < best_delta:
                    best_delta = abs_delta
                    best_v_idx = v_idx

            if best_v_idx != -1 and best_delta <= max_token_delta:
                chosen_v_code = final_unique_vuln[best_v_idx]
                v_len = vuln_tokens[best_v_idx]
                delta = s_len - v_len
                
                candidate_pairs.append({
                    "prompt": task_prompts[tid],
                    "chosen": s_code,
                    "rejected": chosen_v_code,
                    "abs_delta": best_delta,
                    "delta": delta
                })

        # Sort candidate pairs by smallest absolute delta first to ensure optimal matching
        candidate_pairs.sort(key=lambda x: x["abs_delta"])

        # Take the top max_pairs_per_task
        selected_pairs = candidate_pairs[:max_pairs_per_task]
        reconstructed_pairs_by_task[tid] = selected_pairs

        # Calculate metrics for this task
        if selected_pairs:
            deltas = [p["delta"] for p in selected_pairs]
            mean_delta = sum(deltas) / len(selected_pairs)
            chosen_longer = sum(1 for p in selected_pairs if p["delta"] > 0) / len(selected_pairs) * 100
            print(f"Task {tid}: Constructed {len(selected_pairs)} pairs. Mean delta: {mean_delta:.1f} tokens, Chosen longer: {chosen_longer:.1f}%")
        else:
            print(f"Task {tid}: Constructed 0 pairs (all pairs exceeded delta > {max_token_delta} tokens)")

    # 6. Save updated datasets
    new_train_dataset = []
    new_val_dataset = []

    for tid in task_ids:
        pairs = reconstructed_pairs_by_task[tid]
        # Remove metadata keys before saving to JSON
        cleaned_pairs = []
        for p in pairs:
            cleaned_pairs.append({
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"]
            })

        if tid in train_task_ids:
            new_train_dataset.extend(cleaned_pairs)
        elif tid in val_task_ids:
            new_val_dataset.extend(cleaned_pairs)

    print(f"\nSaving reconstructed datasets to {dataset_dir}...")
    with open(train_file, "w") as f:
        json.dump(new_train_dataset, f, indent=2)
    with open(val_file, "w") as f:
        json.dump(new_val_dataset, f, indent=2)

    # 7. Print final statistics
    def calculate_split_stats(pairs, name):
        if not pairs:
            print(f"\n=== {name} Statistics ===\nNo pairs.")
            return
        deltas = []
        chosen_longer_count = 0
        for p in pairs:
            c_len = count_tokens(p["chosen"])
            r_len = count_tokens(p["rejected"])
            delta = c_len - r_len
            deltas.append(delta)
            if delta > 0:
                chosen_longer_count += 1

        mean_delta = sum(deltas) / len(pairs)
        chosen_longer_pct = (chosen_longer_count / len(pairs)) * 100
        print(f"\n=== {name} Statistics ===")
        print(f"Total pairs: {len(pairs)}")
        print(f"Mean token length delta (chosen - rejected): {mean_delta:.2f} tokens")
        print(f"Chosen is longer: {chosen_longer_pct:.2f}%")

    calculate_split_stats(new_train_dataset, "Train Split")
    calculate_split_stats(new_val_dataset, "Validation Split")

    # Save metadata summary
    total_pairs = len(new_train_dataset) + len(new_val_dataset)
    summary = {
        "total_tasks": len(all_tasks),
        "total_pairs": total_pairs,
        "train_tasks": len(train_task_ids),
        "train_pairs": len(new_train_dataset),
        "val_tasks": len(val_task_ids),
        "val_pairs": len(new_val_dataset),
        "task_distribution": {tid: len(pairs) for tid, pairs in reconstructed_pairs_by_task.items()}
    }
    with open(dataset_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nReconstruction process completed successfully!")

if __name__ == "__main__":
    main()
