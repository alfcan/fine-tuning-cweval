#!/usr/bin/env python3
"""
generate_additional_pairs.py
Uses a configurable LLM (e.g., via LM Studio or another API endpoint) to generate
secure and vulnerable completions for underrepresented CWE tasks in the dataset,
evaluates them with CWEval, filters duplicates/similar completions, and merges them.
"""

import os
import re
import sys
import json
import argparse
import random
import difflib
import subprocess

from pathlib import Path
import litellm

# Disable parallel tokenization and fork safety locks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

def parse_args():
    parser = argparse.ArgumentParser(description="Supplement underrepresented CWE tasks with LLM-generated pairs")
    parser.add_argument("--model", type=str, default="openai/Qwen/Qwen3.5-2B", help="Model name for litellm")
    parser.add_argument("--api_base", type=str, default="http://localhost:1234/v1", help="API base URL")
    parser.add_argument("--api_key", type=str, default="sk-local-research", help="API key")
    parser.add_argument("--dataset_dir", type=str, default="results/dataset", help="Path to train/val pairs")
    parser.add_argument("--cweval_dir", type=str, default="CWEval", help="Path to cloned CWEval directory")
    parser.add_argument("--eval_base_dir", type=str, default="results/supplement_gen", help="Directory for evaluation runs")
    parser.add_argument("--docker", type=str, default="True", choices=["True", "False"], help="Run evaluation inside Docker")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel processes for evaluation")
    parser.add_argument("--target_pairs", type=int, default=8, help="Target minimum number of pairs per task")
    parser.add_argument("--max_pairs_per_task", type=int, default=8, help="Maximum number of pairs to keep per task")
    parser.add_argument("--n_candidates", type=int, default=5, help="Number of candidates to generate per category (secure/vuln)")
    parser.add_argument("--similarity_threshold", type=float, default=0.95, help="Similarity threshold for near-duplicate filtering (0.0 to 1.0)")
    parser.add_argument("--temp", type=float, default=0.7, help="LLM generation temperature")
    return parser.parse_args()

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
        # Calculate sequence matcher similarity ratio
        ratio = difflib.SequenceMatcher(None, norm_new, norm_ext).ratio()
        if ratio >= threshold:
            return True
    return False

def clean_extracted_code(code_content):
    cleaned = code_content.strip()
    cleaned = re.sub(r'^```python\s*', '', cleaned)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'```$', '', cleaned)
    cleaned = cleaned.strip()
    return f"```python\n{cleaned}\n```"

def query_model(api_base, api_key, model, messages, temperature, n):
    extra_args = {}
    if api_base:
        extra_args["api_base"] = api_base
    if api_key:
        extra_args["api_key"] = api_key
        
    completions = []
    # Try requesting all at once
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            n=n,
            **extra_args
        )
        completions = [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"All-in-one completion query failed or 'n' not supported: {e}. Falling back to sequential queries...")
        for i in range(n):
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=1,
                    **extra_args
                )
                completions.append(response.choices[0].message.content)
            except Exception as e2:
                print(f"Sequential query {i+1} failed: {e2}")
    return completions

def run_command(cmd, cwd=None):
    print(f"Running command: {' '.join(cmd)}")
    env = os.environ.copy()
    cweval_abs = os.path.abspath("CWEval")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = cweval_abs + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = cweval_abs
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    return ""

def main():
    args = parse_args()
    
    # Configure API base and key in environment for litellm
    if args.api_base:
        os.environ["OPENAI_API_BASE"] = args.api_base
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    cweval_path = Path(args.cweval_dir).resolve()
    if not cweval_path.exists():
        print(f"CWEval repository not found at {cweval_path}.", file=sys.stderr)
        sys.exit(1)

    python_bench = cweval_path / "benchmark" / "core" / "py"
    if not python_bench.exists():
        print(f"Python benchmarks not found under {python_bench}", file=sys.stderr)
        sys.exit(1)

    # 1. Identify all Python tasks (exactly 23 tasks exist in the benchmark)
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

    # 2. Re-establish train/val task assignment deterministically (prevent task leakage)
    task_ids = [t["task_id"] for t in all_tasks]
    random.seed(1337)
    random.shuffle(task_ids)
    split_idx = int(len(task_ids) * 0.8) # 80-20 split by default
    train_task_ids = set(task_ids[:split_idx])
    val_task_ids = set(task_ids[split_idx:])

    # 3. Load existing pairs
    dataset_path = Path(args.dataset_dir)
    train_file = dataset_path / "train_pairs.json"
    val_file = dataset_path / "val_pairs.json"

    train_pairs = []
    val_pairs = []
    if train_file.exists():
        with open(train_file, "r") as f:
            train_pairs = json.load(f)
    if val_file.exists():
        with open(val_file, "r") as f:
            val_pairs = json.load(f)

    # Group existing pairs by task
    existing_pairs_by_task = {tid: [] for tid in task_ids}
    existing_completions = {tid: {"secure": [], "vulnerable": []} for tid in task_ids}

    for pair in train_pairs + val_pairs:
        prompt_content = pair["prompt"]
        if prompt_content in prompt_to_task_id:
            tid = prompt_to_task_id[prompt_content]
            existing_pairs_by_task[tid].append(pair)
            existing_completions[tid]["secure"].append(pair["chosen"])
            existing_completions[tid]["vulnerable"].append(pair["rejected"])

    # 4. Find tasks that need supplementation
    tasks_to_supplement = []
    print("\n=== Dataset Pair Statistics ===")
    for tid in task_ids:
        cnt = len(existing_pairs_by_task[tid])
        print(f"Task {tid}: {cnt} pairs")
        if cnt < args.target_pairs:
            tasks_to_supplement.append(tid)

    if not tasks_to_supplement:
        print("\nAll tasks already have at least target_pairs pairs. No generation needed.")
        sys.exit(0)

    print(f"\nFound {len(tasks_to_supplement)} tasks with fewer than {args.target_pairs} pairs. Starting supplement generation...")

    # 5. Generate secure/vulnerable completions for targets
    # We will generate n_candidates secure and n_candidates vulnerable completions
    # We map them to directories generated_0 to generated_{2*n_candidates - 1}
    eval_path = Path(args.eval_base_dir).resolve() / "supplement_run"
    eval_path.mkdir(parents=True, exist_ok=True)

    # Extract the prompt for each task as understood by CWEval
    # CWEval splits by BEGIN SOLUTION
    for task in all_tasks:
        tid = task["task_id"]
        if tid not in tasks_to_supplement:
            continue
            
        task_code = task_prompts[tid]
        begin_solution_line_src = ''
        for line in task_code.splitlines():
            if 'BEGIN SOLUTION' in line:
                begin_solution_line_src = line
                break
        if not begin_solution_line_src:
            continue
            
        raw_file_name = task["py_file"].name.replace("_task.py", "_raw.py")

        # Check if generated files already exist for this task (resume support)
        all_gen_exist = True
        for idx in range(2 * args.n_candidates):
            gen_file = eval_path / f"generated_{idx}" / "core" / "py" / raw_file_name
            if not gen_file.exists():
                all_gen_exist = False
                break

        if all_gen_exist:
            print(f"\nSkipping generation for {tid}: all {2 * args.n_candidates} candidate files already exist.")
            continue

        code_prompt = (
            task_code.split('BEGIN PROMPT')[-1]
            .split(begin_solution_line_src)[0]
            .strip()
        )
        
        print(f"\nGenerating completions for {tid}...")
        
        # Secure generation prompt
        secure_msgs = [
            {"role": "system", "content": "You are a secure coding assistant. You write secure and correct Python code that is free from vulnerabilities, specifically addressing OWASP and CWE issues."},
            {"role": "user", "content": f"Please complete the following Python code snippet.\nEnsure that the implementation is functionally correct and highly secure, containing no vulnerabilities or security flaws. Do not include a main function, tests, or usage examples.\nOutput ONLY the code starting from the imports and function definition. Wrap your output in a single markdown block using ```python.\n\nSnippet:\n{code_prompt}"}
        ]
        
        # Vulnerable generation prompt
        vuln_msgs = [
            {"role": "system", "content": "You are a programming assistant helping write test cases. You write code that is functionally correct but intentionally contains common security vulnerabilities to test static analysis tools."},
            {"role": "user", "content": f"Please complete the following Python code snippet.\nEnsure that the implementation is functionally correct (i.e., it passes functional test cases) but deliberately contains a security vulnerability corresponding to the description in the prompt. For example, if it's about path traversal, it must be vulnerable to path traversal. Write it realistically, as if an inexperienced programmer wrote it without thinking about security. Do not include a main function, tests, or usage examples.\nOutput ONLY the code starting from the imports and function definition. Wrap your output in a single markdown block using ```python.\n\nSnippet:\n{code_prompt}"}
        ]
        
        secure_completions = query_model(args.api_base, args.api_key, args.model, secure_msgs, args.temp, args.n_candidates)
        vuln_completions = query_model(args.api_base, args.api_key, args.model, vuln_msgs, args.temp, args.n_candidates)

        # Save secure completions: generated_0 to generated_{n_candidates-1}
        for idx, comp in enumerate(secure_completions):
            formatted_code = clean_extracted_code(comp)
            out_file = eval_path / f"generated_{idx}" / "core" / "py" / raw_file_name
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                f.write(formatted_code)

        # Save vulnerable completions: generated_{n_candidates} to generated_{2*n_candidates-1}
        for idx, comp in enumerate(vuln_completions):
            formatted_code = clean_extracted_code(comp)
            out_file = eval_path / f"generated_{idx + args.n_candidates}" / "core" / "py" / raw_file_name
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                f.write(formatted_code)

    # 6. Run evaluation (skip if results already exist)
    res_file = eval_path / "res_all.json"
    if res_file.exists():
        print(f"\nEvaluation results already exist at {res_file}. Skipping evaluation.")
    else:
        print("\nRunning evaluation on generated candidates...")
        try:
            eval_script = str(cweval_path / "cweval" / "evaluate.py")
            if args.docker == "True":
                from cweval_orchestrator import run_evaluation_in_docker
                run_evaluation_in_docker(eval_path, num_proc=args.num_proc)
            else:
                eval_cmd = [
                    sys.executable, eval_script, "pipeline",
                    "--eval_path", str(eval_path),
                    "--num_proc", str(args.num_proc),
                    "--docker", "False"
                ]
                run_command(eval_cmd, cwd=str(cweval_path))
        except Exception as e:
            print(f"Evaluation failed: {e}")
            sys.exit(1)

    # 7. Parse results and build supplemented pairs
    if not res_file.exists():
        print(f"Error: Evaluation results file not found at {res_file}", file=sys.stderr)
        sys.exit(1)

    with open(res_file, "r") as f:
        res_data = json.load(f)

    # Build a set of all existing pair signatures to prevent re-adding duplicates
    # (covers the case where pairs were already generated, evaluated, and merged
    #  in a previous run of this script)
    existing_pair_signatures = set()
    for pair in train_pairs + val_pairs:
        norm_c = normalize_code(pair["chosen"])
        norm_r = normalize_code(pair["rejected"])
        existing_pair_signatures.add((norm_c, norm_r))

    supplemented_pairs_by_task = {}

    for task in all_tasks:
        tid = task["task_id"]
        if tid not in tasks_to_supplement:
            supplemented_pairs_by_task[tid] = existing_pairs_by_task[tid]
            continue

        test_file_name = task["py_file"].name.replace("_task.py", "_test.py")
        raw_file_name = task["py_file"].name.replace("_task.py", "_raw.py")

        matching_key = None
        for key in res_data.keys():
            if key.endswith(test_file_name):
                matching_key = key
                break

        if not matching_key:
            print(f"No evaluation results for task {tid}. Keeping existing pairs.")
            supplemented_pairs_by_task[tid] = existing_pairs_by_task[tid]
            continue

        attempts = res_data[matching_key]
        functional_list = attempts.get("functional", [])
        secure_list = attempts.get("secure", [])

        # Lists to store new unique completions
        new_unique_sec = []
        new_unique_vuln = []

        # Secure candidates are in indices 0 to n_candidates-1
        for idx in range(args.n_candidates):
            if idx >= len(functional_list) or idx >= len(secure_list):
                continue
            is_func = functional_list[idx]
            is_sec = secure_list[idx]

            if is_func and is_sec:
                gen_file = eval_path / f"generated_{idx}" / "core" / "py" / raw_file_name
                if gen_file.exists():
                    with open(gen_file, "r") as f:
                        code_content = f.read()
                    
                    # Deduplicate and check similarity threshold
                    if not is_too_similar(code_content, existing_completions[tid]["secure"], args.similarity_threshold):
                        if not is_too_similar(code_content, new_unique_sec, args.similarity_threshold):
                            new_unique_sec.append(code_content)

        # Vulnerable candidates are in indices n_candidates to 2*n_candidates-1
        for idx in range(args.n_candidates, 2 * args.n_candidates):
            if idx >= len(functional_list) or idx >= len(secure_list):
                continue
            is_func = functional_list[idx]
            is_sec = secure_list[idx]

            if is_func and not is_sec:
                gen_file = eval_path / f"generated_{idx}" / "core" / "py" / raw_file_name
                if gen_file.exists():
                    with open(gen_file, "r") as f:
                        code_content = f.read()

                    # Deduplicate and check similarity threshold
                    if not is_too_similar(code_content, existing_completions[tid]["vulnerable"], args.similarity_threshold):
                        if not is_too_similar(code_content, new_unique_vuln, args.similarity_threshold):
                            new_unique_vuln.append(code_content)

        print(f"Task {tid}: Found {len(new_unique_sec)} new valid secure and {len(new_unique_vuln)} new valid vulnerable completions.")

        # Build all possible cross-product pairs from the combination of existing and new completions
        all_sec = existing_completions[tid]["secure"] + new_unique_sec
        all_vuln = existing_completions[tid]["vulnerable"] + new_unique_vuln

        # Deduplicate the combined sets themselves to be robust
        final_unique_sec = []
        for s in all_sec:
            if not is_too_similar(s, final_unique_sec, args.similarity_threshold):
                final_unique_sec.append(s)

        final_unique_vuln = []
        for v in all_vuln:
            if not is_too_similar(v, final_unique_vuln, args.similarity_threshold):
                final_unique_vuln.append(v)

        # Re-construct pairs prioritising existing pairs first
        # Use the global signature set to catch pairs already in the dataset
        # (including those added by a previous run of this script)
        task_pairs = list(existing_pairs_by_task[tid])
        
        # Add new combinations (skip any pair whose signature already exists globally)
        new_constructed_pairs = []
        for sc in final_unique_sec:
            for vc in final_unique_vuln:
                norm_sc = normalize_code(sc)
                norm_vc = normalize_code(vc)
                if (norm_sc, norm_vc) not in existing_pair_signatures:
                    new_constructed_pairs.append({
                        "prompt": task_prompts[tid],
                        "chosen": sc,
                        "rejected": vc
                    })

        needed = args.max_pairs_per_task - len(task_pairs)
        if needed > 0 and new_constructed_pairs:
            # Shuffle the new constructed pairs for diversity
            random.seed(42)
            sampled = random.sample(new_constructed_pairs, min(needed, len(new_constructed_pairs)))
            task_pairs.extend(sampled)

        supplemented_pairs_by_task[tid] = task_pairs
        print(f"Task {tid} final pair count: {len(task_pairs)} (was {len(existing_pairs_by_task[tid])})")

    # 8. Save updated datasets
    new_train_dataset = []
    new_val_dataset = []

    for tid in task_ids:
        pairs = supplemented_pairs_by_task[tid]
        if tid in train_task_ids:
            new_train_dataset.extend(pairs)
        elif tid in val_task_ids:
            new_val_dataset.extend(pairs)

    print(f"\nSaving updated datasets to {args.dataset_dir}...")
    with open(train_file, "w") as f:
        json.dump(new_train_dataset, f, indent=2)
    with open(val_file, "w") as f:
        json.dump(new_val_dataset, f, indent=2)

    # Save metadata summary
    total_pairs = len(new_train_dataset) + len(new_val_dataset)
    summary = {
        "total_tasks": len(all_tasks),
        "total_pairs": total_pairs,
        "train_tasks": len(train_task_ids),
        "train_pairs": len(new_train_dataset),
        "val_tasks": len(val_task_ids),
        "val_pairs": len(new_val_dataset),
        "task_distribution": {tid: len(pairs) for tid, pairs in supplemented_pairs_by_task.items()}
    }
    with open(dataset_path / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSupplement process completed successfully!")
    print(f"Updated train pairs: {len(new_train_dataset)}")
    print(f"Updated validation pairs: {len(new_val_dataset)}")
    print(f"Summary saved to {dataset_path / 'dataset_summary.json'}")

if __name__ == "__main__":
    main()
