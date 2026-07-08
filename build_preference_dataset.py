#!/usr/bin/env python3
"""
build_preference_dataset.py - Phase 2 & 3: Preference Pair Construction and Splitting
Generates N completions for 25 Python tasks using on-policy temperature sampling.
Evaluates completions via CWEval, retries with paraphrased prompts if needed,
constructs chosen (secure) / rejected (vulnerable) preference pairs,
deduplicates and splits into train/validation by task.
"""

import os
import json
import argparse
import subprocess
import shutil
import sys
import re
import random
from pathlib import Path
import requests

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 & 3: Preference Dataset Builder")
    parser.add_argument("--model", type=str, default="openai/Qwen/Qwen3.5-2B", help="Model to query via local API server")
    parser.add_argument("--api_base", type=str, default="http://localhost:1234/v1", help="Local model server API base url")
    parser.add_argument("--api_key", type=str, default="sk-local-research", help="API key for inference server")
    parser.add_argument("--eval_base_dir", type=str, default="results/preference_gen", help="Directory for generations")
    parser.add_argument("--cweval_dir", type=str, default="CWEval", help="Path to cloned CWEval directory")
    parser.add_argument("--docker", type=str, default="True", choices=["True", "False"], help="Run evaluation inside Docker")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate per temperature")
    parser.add_argument("--max_pairs_per_task", type=int, default=8, help="Maximum number of pairs to keep per task")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of tasks to assign to train set")
    return parser.parse_args()

def start_local_server(model_id_or_path, api_base):
    import time
    import requests
    import urllib.parse
    
    # Parse port from api_base
    port = 1234
    try:
        parsed = urllib.parse.urlparse(api_base)
        if parsed.port:
            port = parsed.port
    except Exception:
        pass

    model_id = model_id_or_path
    if model_id.startswith("openai/"):
        model_id = model_id[len("openai/"):]

    print(f"Starting local model server for '{model_id}' on port {port}...")
    server_process = subprocess.Popen([
        sys.executable, "openai_server.py",
        "--model", model_id,
        "--port", str(port)
    ])

    # Wait for the server to be ready by checking /health endpoint
    max_retries = 60
    for i in range(max_retries):
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=1)
            if resp.status_code == 200:
                print("Local model server is ready!")
                return server_process
        except Exception:
            pass
        time.sleep(1)
        if server_process.poll() is not None:
            print("Local model server process terminated unexpectedly.")
            sys.exit(1)

    print("Timeout waiting for local model server to start.")
    server_process.terminate()
    server_process.wait()
    sys.exit(1)

def check_func_and_sec(attempt):
    if isinstance(attempt, dict):
        func = None
        for k in ["func", "func_correct", "functional", "correct", "passed", "pass", "functional_correctness"]:
            if k in attempt:
                func = bool(attempt[k])
                break
        if func is None:
            func = attempt.get("status") == "success" or attempt.get("exit_code") == 0

        sec = None
        for k in ["sec", "secure", "security", "safe"]:
            if k in attempt:
                sec = bool(attempt[k])
                break
        if sec is None:
            if "vulnerable" in attempt:
                sec = not bool(attempt["vulnerable"])
            elif "vuln" in attempt:
                sec = not bool(attempt["vuln"])
        
        if func is None: func = False
        if sec is None: sec = False
        return func, sec
    elif isinstance(attempt, (list, tuple)):
        if len(attempt) >= 2:
            return bool(attempt[0]), bool(attempt[1])
        elif len(attempt) == 1:
            return bool(attempt[0]), True
    return False, False

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

def normalize_code(code):
    # Strip comments and docstrings, then compress whitespace to identify duplicates
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code).strip()
    return code

def call_paraphrase_api(api_base, model, file_content):
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    prompt = (
        "You are given a Python file representing a coding task. "
        "Paraphrase the docstring, instructions, or natural language comments in this file to request the same logic in different words. "
        "Do NOT change any code syntax, imports, function names, parameter names, type hints, or return statements. "
        "Keep the structure of the python script exactly identical. "
        "Return ONLY the modified python file code, with no markdown code blocks, explanation or formatting.\n\n"
        f"File content:\n{file_content}"
    )
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant. You return only code without markdown formatting."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            res_content = response.json()["choices"][0]["message"]["content"]
            # Clean potential markdown wrappers from LLM response
            res_content = re.sub(r'^```python\n', '', res_content)
            res_content = re.sub(r'^```\n', '', res_content)
            res_content = re.sub(r'\n```$', '', res_content)
            return res_content
    except Exception as e:
        print(f"API Paraphrase failed: {e}")
    return None

def rule_based_paraphrase(file_content, iteration):
    # Simple rule-based fallbacks to modify docstrings or append comments
    extra_notes = [
        "\n# Note: Please implement this function cleanly and securely, avoiding vulnerability risks.\n",
        "\n# Note: Focus on correct validation, strict type handling, and secure defaults.\n",
        "\n# Note: Ensure the function behaves robustly and securely under untrusted input.\n"
    ]
    note = extra_notes[iteration % len(extra_notes)]
    # Look for function definition def ...: and append note right after or inside docstring
    if '"""' in file_content:
        parts = file_content.split('"""', 2)
        if len(parts) == 3:
            return f"{parts[0]}\"\"\"{parts[1]}\n{note.strip()}\"\"\"{parts[2]}"
    return file_content + note

def main():
    args = parse_args()
    
    # Disable parallel tokenization and bypass macOS fork safety locks to prevent deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    
    # Configure API base and key in environment for litellm
    if args.api_base:
        os.environ["OPENAI_API_BASE"] = args.api_base
        print(f"Set environment OPENAI_API_BASE={args.api_base}")
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        print(f"Set environment OPENAI_API_KEY={args.api_key}")

    cweval_path = Path(args.cweval_dir).resolve()
    if not cweval_path.exists():
        print(f"CWEval repository not found at {cweval_path}.", file=sys.stderr)
        sys.exit(1)

    python_bench = cweval_path / "benchmark" / "core" / "py"
    if not python_bench.exists():
        print(f"Python benchmarks not found under {python_bench}", file=sys.stderr)
        sys.exit(1)

    # 1. Identify all Python tasks (exactly 25 tasks exist in the benchmark)
    task_files = sorted(list(python_bench.glob("cwe_*_task.py")))
    all_tasks = []
    for task_file in task_files:
        task_name = task_file.name.replace("_task.py", "")
        parts = task_name.split("_")
        cwe_id = f"cwe-{parts[1]}"
        all_tasks.append({
            "task_id": f"core/py/{task_name}",
            "cwe": cwe_id,
            "task_name": task_name,
            "py_file": task_file,
            "dir": python_bench
        })

    if len(all_tasks) < 25:
        print(f"Warning: Found only {len(all_tasks)} Python tasks. Using all of them.")
        selected_tasks = all_tasks
    else:
        # Use a stable seed to select 25 tasks deterministically
        random.seed(42)
        selected_tasks = random.sample(all_tasks, 25)
        print(f"Selected {len(selected_tasks)} Python tasks out of {len(all_tasks)} available.")

    selected_task_ids = {t["task_id"] for t in selected_tasks}

    # We will build and store generated completions in memory mapped by task_id
    # task_id -> list of {"code": str, "func": bool, "sec": bool, "temp": float}
    task_completions = {t["task_id"]: [] for t in selected_tasks}
    
    # Track prompt contents to know the exact instruction/prompt used
    task_prompts = {}
    for t in selected_tasks:
        with open(t["py_file"], "r") as f:
            task_prompts[t["task_id"]] = f.read()

    temperatures = [0.4, 0.6, 0.8, 1.0]
    
    # Start the local model server
    server_proc = start_local_server(args.model, args.api_base)
    try:
        # 3. Generate samples at multiple temperatures
        for temp in temperatures:
            print(f"\n--- Generating samples at temperature {temp} ---")
            eval_path = Path(args.eval_base_dir).resolve() / f"temp_{temp}"
            if eval_path.exists():
                shutil.rmtree(eval_path)
            
            gen_script = str(Path("run_generation.py").resolve())
            eval_script = str(cweval_path / "cweval" / "evaluate.py")

            # Run generation (cwd must be CWEval for benchmark discovery)
            gen_cmd = [
                sys.executable, gen_script, "gen",
                "--model", args.model,
                "--n", str(args.n_samples),
                "--temperature", str(temp),
                "--eval_path", str(eval_path),
                "--api_base", args.api_base,
                "--api_key", args.api_key,
                "--langs", "['py']",
                "--num_proc", str(args.num_proc)
            ]
            run_command(gen_cmd, cwd=str(cweval_path))

            # Run evaluation
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

            # Load results
            res_file = eval_path / "res_all.json"
            if not res_file.exists():
                print(f"Warning: Result file {res_file} not found.")
                continue

            with open(res_file, "r") as f:
                res_data = json.load(f)

            # For each task, map the files to the results
            for task in selected_tasks:
                tid = task["task_id"]
                
                # The test file basename corresponding to this task
                test_file_name = task["py_file"].name.replace("_task.py", "_test.py")
                raw_file_name = task["py_file"].name.replace("_task.py", "_raw.py")
                
                # Find the key in res_all.json that ends with test_file_name
                matching_key = None
                for key in res_data.keys():
                    if key.endswith(test_file_name):
                        matching_key = key
                        break
                
                if not matching_key:
                    continue
                    
                attempts = res_data[matching_key]
                n_samples = len(attempts.get("functional", []))
                
                for idx in range(n_samples):
                    is_func = attempts["functional"][idx]
                    is_sec = attempts["secure"][idx]
                    
                    # The file containing the generated code for this sample
                    gen_file = eval_path / f"generated_{idx}" / "core" / "py" / raw_file_name
                    if gen_file.exists():
                        with open(gen_file, "r") as f:
                            code_content = f.read()
                        
                        task_completions[tid].append({
                            "code": code_content,
                            "func": is_func,
                            "sec": is_sec,
                            "temp": temp
                        })

        # 4. Paraphrase Retry Loop
        # Check which tasks do not have both a secure and vulnerable functionally correct completion
        print("\n=== Checking for complete pairs per task ===")
        failing_tasks = []
        for tid in selected_task_ids:
            completions = task_completions[tid]
            has_secure = any(c["func"] and c["sec"] for c in completions)
            has_vuln = any(c["func"] and not c["sec"] for c in completions)
            if not (has_secure and has_vuln):
                print(f"Task {tid} lacks balance: has_secure={has_secure}, has_vuln={has_vuln}")
                failing_tasks.append(tid)
            else:
                print(f"Task {tid} has balanced samples.")

        # Run retries for failing tasks
        for tid in failing_tasks:
            task = next(t for t in selected_tasks if t["task_id"] == tid)
            print(f"\nAttempting paraphrasing retry loop for {tid}...")
            
            # Read original task content
            with open(task["py_file"], "r") as f:
                original_content = f.read()

            success = False
            for iter_idx in range(3):  # up to 3 paraphrases
                print(f"Paraphrase iteration {iter_idx+1} for {tid}...")
                
                # Attempt API paraphrase, fall back to rule-based
                paraphrased = call_paraphrase_api(args.api_base, args.model, original_content)
                if not paraphrased:
                    paraphrased = rule_based_paraphrase(original_content, iter_idx)

                # Overwrite task file
                with open(task["py_file"], "w") as f:
                    f.write(paraphrased)

                eval_path = Path(args.eval_base_dir).resolve() / f"retry_{task['task_name']}_iter_{iter_idx}"
                if eval_path.exists():
                    shutil.rmtree(eval_path)

                # Generate and evaluate (cwd must be CWEval for benchmark discovery)
                # We filter generation to only run for this specific python task file
                gen_cmd = [
                    sys.executable, str(Path("run_generation.py").resolve()), "gen",
                    "--model", args.model,
                    "--n", str(args.n_samples),
                    "--temperature", "0.8",  # high temp for retry diversity
                    "--eval_path", str(eval_path),
                    "--api_base", args.api_base,
                    "--api_key", args.api_key,
                    "--langs", "['py']",
                    "--include_path", f"['{task['task_name']}_task.py']",
                    "--num_proc", str(args.num_proc)
                ]
                run_command(gen_cmd, cwd=str(cweval_path))

                if args.docker == "True":
                    from cweval_orchestrator import run_evaluation_in_docker
                    run_evaluation_in_docker(eval_path, num_proc=args.num_proc)
                else:
                    eval_cmd = [
                        sys.executable, str(cweval_path / "cweval" / "evaluate.py"), "pipeline",
                        "--eval_path", str(eval_path),
                        "--num_proc", str(args.num_proc),
                        "--docker", "False"
                    ]
                    run_command(eval_cmd, cwd=str(cweval_path))

                # Parse and merge
                res_file = eval_path / "res_all.json"
                if res_file.exists():
                    with open(res_file, "r") as f:
                        res_data = json.load(f)
                    
                    # The test file basename corresponding to this task
                    test_file_name = task["py_file"].name.replace("_task.py", "_test.py")
                    raw_file_name = task["py_file"].name.replace("_task.py", "_raw.py")
                    
                    # Find the key in res_all.json that ends with test_file_name
                    matching_key = None
                    for key in res_data.keys():
                        if key.endswith(test_file_name):
                            matching_key = key
                            break
                    
                    if matching_key:
                        attempts = res_data[matching_key]
                        n_samples = len(attempts.get("functional", []))
                        
                        for idx in range(n_samples):
                            is_func = attempts["functional"][idx]
                            is_sec = attempts["secure"][idx]
                            
                            # The file containing the generated code for this sample
                            gen_file = eval_path / f"generated_{idx}" / "core" / "py" / raw_file_name
                            if gen_file.exists():
                                with open(gen_file, "r") as f:
                                    code_content = f.read()
                                
                                task_completions[tid].append({
                                    "code": code_content,
                                    "func": is_func,
                                    "sec": is_sec,
                                    "temp": 0.8
                                })

                # Check if we now have both secure and vuln
                completions = task_completions[tid]
                has_secure = any(c["func"] and c["sec"] for c in completions)
                has_vuln = any(c["func"] and not c["sec"] for c in completions)
                if has_secure and has_vuln:
                    print(f"Success! Balanced completions found for {tid} after paraphrase iteration {iter_idx+1}.")
                    success = True
                    break
            
            # Restore original task file
            with open(task["py_file"], "w") as f:
                f.write(original_content)

            if not success:
                print(f"Paraphrasing retry loop failed to balance {tid}.")

    finally:
        print("\nCompleted generation/evaluation attempts.")
        print("Stopping local model server...")
        server_proc.terminate()
        server_proc.wait()

    # 5. Pair Construction and Deduplication
    print("\n=== Constructing Preference Pairs ===")
    preference_pairs_by_task = {}
    total_pairs = 0

    for tid, completions in task_completions.items():
        # Get correct completions
        correct_completions = [c for c in completions if c["func"]]
        sec_completions = [c for c in correct_completions if c["sec"]]
        vuln_completions = [c for c in correct_completions if not c["sec"]]

        # Deduplicate within categories
        unique_sec = []
        seen_sec_norms = set()
        for sc in sec_completions:
            norm = normalize_code(sc["code"])
            if norm not in seen_sec_norms:
                seen_sec_norms.add(norm)
                unique_sec.append(sc["code"])

        unique_vuln = []
        seen_vuln_norms = set()
        for vc in vuln_completions:
            norm = normalize_code(vc["code"])
            if norm not in seen_vuln_norms:
                seen_vuln_norms.add(norm)
                unique_vuln.append(vc["code"])

        task_pairs = []
        # Pair them up
        for sc_code in unique_sec:
            for vc_code in unique_vuln:
                # Add pair
                task_pairs.append({
                    "prompt": task_prompts[tid],
                    "chosen": sc_code,
                    "rejected": vc_code
                })

        # Apply per-task cap
        if len(task_pairs) > args.max_pairs_per_task:
            # Seed for reproducibility
            random.seed(42)
            task_pairs = random.sample(task_pairs, args.max_pairs_per_task)

        preference_pairs_by_task[tid] = task_pairs
        total_pairs += len(task_pairs)
        print(f"Task {tid}: Constructed {len(task_pairs)} pairs (secure_unique={len(unique_sec)}, vuln_unique={len(unique_vuln)})")

    print(f"Total preference pairs constructed: {total_pairs}")

    # 6. Data Splitting (Train/Validation split by task)
    print("\n=== Performing Train/Validation Split ===")
    task_ids = list(preference_pairs_by_task.keys())
    # Shuffle task ids reproducibly
    random.seed(1337)
    random.shuffle(task_ids)

    split_idx = int(len(task_ids) * args.train_split)
    train_task_ids = task_ids[:split_idx]
    val_task_ids = task_ids[split_idx:]

    train_dataset = []
    for tid in train_task_ids:
        train_dataset.extend(preference_pairs_by_task[tid])

    val_dataset = []
    for tid in val_task_ids:
        val_dataset.extend(preference_pairs_by_task[tid])

    print(f"Train Set: {len(train_task_ids)} tasks, {len(train_dataset)} pairs")
    print(f"Validation Set: {len(val_task_ids)} tasks, {len(val_dataset)} pairs")

    # Save datasets
    output_dir = Path("results/dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train_pairs.json", "w") as f:
        json.dump(train_dataset, f, indent=2)
    with open(output_dir / "val_pairs.json", "w") as f:
        json.dump(val_dataset, f, indent=2)

    # Save metadata summary
    summary = {
        "total_tasks": len(selected_tasks),
        "total_pairs": total_pairs,
        "train_tasks": len(train_task_ids),
        "train_pairs": len(train_dataset),
        "val_tasks": len(val_task_ids),
        "val_pairs": len(val_dataset),
        "task_distribution": {tid: len(pairs) for tid, pairs in preference_pairs_by_task.items()}
    }
    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDatasets saved to {output_dir}/")
    print(f"Dataset summary saved to {output_dir / 'dataset_summary.json'}")

if __name__ == "__main__":
    main()
