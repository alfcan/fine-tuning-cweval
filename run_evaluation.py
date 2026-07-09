#!/usr/bin/env python3
"""
run_evaluation.py - Phase 5: Evaluation
Performs weight merging for trained PEFT adapters.
Automates evaluation of base and tuned seed models on Python, JS, C, C++, Go.
Calculates pass@1, security-rate, splits by CWE category (known vs novel),
computes bootstrap confidence intervals, and aggregates variance across seeds.
"""

import os
import json
import argparse
import subprocess
import sys
import random
import time
import requests
import urllib.parse
from pathlib import Path
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5: Evaluation & Analysis")
    parser.add_argument("--model_id", type=str, default="openai/Qwen/Qwen3.5-2B", help="Base model ID")
    parser.add_argument("--adapter_dir", type=str, default="results/checkpoints", help="Path containing seed checkpoints")
    parser.add_argument("--merged_dir", type=str, default="results/merged", help="Directory to save merged models")
    parser.add_argument("--dataset_dir", type=str, default="results/dataset", help="Path containing dataset summaries")
    parser.add_argument("--eval_base_dir", type=str, default="results/eval", help="Path to write evaluation outputs")
    parser.add_argument("--cweval_dir", type=str, default="CWEval", help="Path to cloned CWEval directory")
    parser.add_argument("--docker", type=str, default="False", choices=["True", "False"], help="Run evaluation inside Docker")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated random seeds used in training")
    parser.add_argument("--skip_merge", action="store_true", help="Skip the PEFT merge step")
    parser.add_argument("--bootstrap_samples", type=int, default=1000, help="Number of bootstrap resamples for CI")
    parser.add_argument("--api_base", type=str, default="http://localhost:1234/v1", help="Local model server API base url")
    parser.add_argument("--api_key", type=str, default="sk-local-research", help="API key for inference server")
    return parser.parse_args()

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

def start_local_server(model_id_or_path, api_base):
    # Parse port from api_base (e.g. http://localhost:1234/v1)
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

def merge_peft_adapters(base_model_id, adapter_dir, output_dir, seeds):
    """
    Loads base model and merges PEFT adapters for each seed, saving full models.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Clean the base_model_id (strip openai/ if present)
    base_model_id_clean = base_model_id
    if base_model_id_clean.startswith("openai/"):
        base_model_id_clean = base_model_id_clean[len("openai/"):]

    for seed in seeds:
        seed_adapter = Path(adapter_dir) / f"seed_{seed}" / "best_model"
        seed_merged = Path(output_dir) / f"seed_{seed}"
        
        if not seed_adapter.exists():
            print(f"Adapter not found for seed {seed} at {seed_adapter}. Skipping.")
            continue
            
        if seed_merged.exists():
            print(f"Merged model for seed {seed} already exists at {seed_merged}. Skipping merge.")
            continue

        print(f"\nMerging adapter for Seed {seed}...")
        print(f"Loading base model {base_model_id_clean}...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id_clean,
            dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id_clean)

        print(f"Loading adapter from {seed_adapter}...")
        model = PeftModel.from_pretrained(base_model, seed_adapter)

        print("Merging weights...")
        merged_model = model.merge_and_unload()

        print(f"Saving merged model to {seed_merged}...")
        seed_merged.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(seed_merged))
        tokenizer.save_pretrained(str(seed_merged))
        
        # Clean up memory
        del base_model
        del model
        del merged_model
        import gc
        gc.collect()
        
    print("\nPEFT adapters merged successfully.")

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

def bootstrap_ci(data, num_bootstraps=1000, ci=95):
    """
    Computes bootstrap confidence intervals (lower and upper percentiles) for a binary list.
    """
    if not data:
        return 0.0, 0.0
    scores = []
    n = len(data)
    for _ in range(num_bootstraps):
        sample = [random.choice(data) for _ in range(n)]
        scores.append(sum(sample) / n)
    lower = np.percentile(scores, (100 - ci) / 2) * 100.0
    upper = np.percentile(scores, 100 - (100 - ci) / 2) * 100.0
    return lower, upper

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
        
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # 1. Merge adapters if not skipped
    if not args.skip_merge:
        try:
            merge_peft_adapters(args.model_id, args.adapter_dir, args.merged_dir, seeds)
        except ImportError:
            print("PEFT / transformers not available for merging locally. Skipping merge step. Assuming merged models are already in place.")

    # 2. Identify the training CWEs from dataset summary
    summary_path = Path(args.dataset_dir) / "dataset_summary.json"
    known_cwes = set()
    if summary_path.exists():
        with open(summary_path, "r") as f:
            meta = json.load(f)
        for task_id in meta.get("task_distribution", {}).keys():
            # task_id structure: "python/cwe-XXX/task_YYY"
            parts = task_id.split("/")
            if len(parts) >= 2:
                known_cwes.add(parts[1])
    else:
        print(f"Warning: Dataset summary not found at {summary_path}. Novel vs Known CWE analysis will be skipped.")

    # 3. Read baseline flags to identify "at risk" languages
    baseline_summary_path = Path("results/baseline_summary.json")
    at_risk_langs = {}
    if baseline_summary_path.exists():
        with open(baseline_summary_path, "r") as f:
            base_meta = json.load(f)
        for lang, item in base_meta.items():
            if "RISK" in item.get("risk_status", ""):
                at_risk_langs[lang] = item["risk_status"]
    
    # 4. Evaluation Loop
    # We will evaluate the base model and each seed model
    models_to_eval = {"base": args.model_id}
    for seed in seeds:
        models_to_eval[f"seed_{seed}"] = "openai/" + str(Path(args.merged_dir) / f"seed_{seed}")

    eval_results = {}

    for name, model_path in models_to_eval.items():
        print(f"\n==========================================")
        print(f"Evaluating Model Configuration: {name}")
        print(f"==========================================")
        
        server_proc = start_local_server(model_path, args.api_base)
        try:
            eval_path = (Path(args.eval_base_dir) / name).resolve()
            # We don't delete the directory here so that run_generation.py can skip/resume already generated samples.
            
            gen_script = str(Path("run_generation.py").resolve())
            eval_script = str(Path(args.cweval_dir).resolve() / "cweval" / "evaluate.py")

            # Run generation
            gen_cmd = [
                sys.executable, gen_script, "gen",
                "--model", model_path,
                "--n", "1",
                "--temperature", "0.0",
                "--eval_path", str(eval_path),
                "--api_base", args.api_base,
                "--api_key", args.api_key,
                "--num_proc", str(args.num_proc)
            ]
            run_command(gen_cmd, cwd=args.cweval_dir)

            # Run evaluation pipeline
            if args.docker == "True":
                from cweval_orchestrator import run_evaluation_in_docker
                run_evaluation_in_docker(str(eval_path), num_proc=args.num_proc)
            else:
                eval_cmd = [
                    sys.executable, eval_script, "pipeline",
                    "--eval_path", str(eval_path),
                    "--num_proc", str(args.num_proc),
                    "--docker", "False"
                ]
                run_command(eval_cmd, cwd=args.cweval_dir)
        finally:
            print("Stopping local model server...")
            server_proc.terminate()
            server_proc.wait()

        # Load res_all.json
        res_file = eval_path / "res_all.json"
        if not res_file.exists():
            print(f"Error: res_all.json not found for {name} at {res_file}", file=sys.stderr)
            continue
            
        with open(res_file, "r") as f:
            res_data = json.load(f)

        # Structure metrics
        # Group tasks by language
        lang_data = {}
        for task_id, attempts in res_data.items():
            parts = task_id.split("/")
            if len(parts) < 3:
                continue
            lang, cwe_id, task_name = parts[0], parts[1], parts[2]
            
            if lang not in lang_data:
                lang_data[lang] = []
                
            # Since n=1, there is only one attempt
            for attempt in attempts:
                is_func, is_sec = check_func_and_sec(attempt)
                lang_data[lang].append({
                    "task_id": task_id,
                    "cwe": cwe_id,
                    "functional": is_func,
                    "secure": is_sec
                })

        # Calculate metrics per language
        eval_results[name] = {}
        for lang, tasks in lang_data.items():
            funcs = [t["functional"] for t in tasks]
            
            # Security rates are computed ONLY on correct completions
            sec_given_func = [t["secure"] for t in tasks if t["functional"]]

            pass1 = (sum(funcs) / len(funcs)) * 100.0 if funcs else 0.0
            sec_rate = (sum(sec_given_func) / len(sec_given_func)) * 100.0 if sec_given_func else 0.0

            # Bootstrap CIs
            pass1_ci = bootstrap_ci(funcs, args.bootstrap_samples)
            sec_ci = bootstrap_ci(sec_given_func, args.bootstrap_samples)

            # Split into known CWE vs novel CWE (for held-out languages)
            known_cwe_tasks = [t for t in tasks if t["cwe"] in known_cwes]
            novel_cwe_tasks = [t for t in tasks if t["cwe"] not in known_cwes]

            known_funcs = [t["functional"] for t in known_cwe_tasks]
            known_sec = [t["secure"] for t in known_cwe_tasks if t["functional"]]
            
            novel_funcs = [t["functional"] for t in novel_cwe_tasks]
            novel_sec = [t["secure"] for t in novel_cwe_tasks if t["functional"]]

            eval_results[name][lang] = {
                "total_tasks": len(tasks),
                "pass1_rate": pass1,
                "pass1_ci": pass1_ci,
                "security_rate": sec_rate,
                "security_ci": sec_ci,
                "at_risk": lang in at_risk_langs,
                "risk_message": at_risk_langs.get(lang, "OK"),
                "known_cwe": {
                    "total_tasks": len(known_cwe_tasks),
                    "pass1_rate": (sum(known_funcs) / len(known_funcs)) * 100.0 if known_funcs else 0.0,
                    "security_rate": (sum(known_sec) / len(known_sec)) * 100.0 if known_sec else 0.0
                },
                "novel_cwe": {
                    "total_tasks": len(novel_cwe_tasks),
                    "pass1_rate": (sum(novel_funcs) / len(novel_funcs)) * 100.0 if novel_funcs else 0.0,
                    "security_rate": (sum(novel_sec) / len(novel_sec)) * 100.0 if novel_sec else 0.0
                }
            }

    # 5. Aggregate metrics across seeds (compute mean, std, variance)
    languages = list(eval_results["base"].keys()) if "base" in eval_results else []
    aggregated_results = {
        "base": eval_results.get("base", {}),
        "seeds": {f"seed_{seed}": eval_results.get(f"seed_{seed}", {}) for seed in seeds},
        "aggregate": {}
    }

    for lang in languages:
        seed_pass1s = []
        seed_sec_rates = []
        seed_known_secs = []
        seed_novel_secs = []

        for seed in seeds:
            seed_name = f"seed_{seed}"
            if seed_name in eval_results and lang in eval_results[seed_name]:
                seed_pass1s.append(eval_results[seed_name][lang]["pass1_rate"])
                seed_sec_rates.append(eval_results[seed_name][lang]["security_rate"])
                seed_known_secs.append(eval_results[seed_name][lang]["known_cwe"]["security_rate"])
                seed_novel_secs.append(eval_results[seed_name][lang]["novel_cwe"]["security_rate"])

        if seed_pass1s:
            aggregated_results["aggregate"][lang] = {
                "pass1": {
                    "mean": float(np.mean(seed_pass1s)),
                    "std": float(np.std(seed_pass1s)),
                    "variance": float(np.var(seed_pass1s))
                },
                "security_rate": {
                    "mean": float(np.mean(seed_sec_rates)),
                    "std": float(np.std(seed_sec_rates)),
                    "variance": float(np.var(seed_sec_rates))
                },
                "known_cwe_security_rate": {
                    "mean": float(np.mean(seed_known_secs)) if seed_known_secs else 0.0,
                    "std": float(np.std(seed_known_secs)) if seed_known_secs else 0.0
                },
                "novel_cwe_security_rate": {
                    "mean": float(np.mean(seed_novel_secs)) if seed_novel_secs else 0.0,
                    "std": float(np.std(seed_novel_secs)) if seed_novel_secs else 0.0
                }
            }

    # Save summary json
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(aggregated_results, f, indent=2)

    print(f"\nEvaluation summary written to {summary_file}")

if __name__ == "__main__":
    main()
