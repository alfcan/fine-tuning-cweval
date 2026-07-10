#!/usr/bin/env python3
"""
run_evaluation.py - Phase 5: Evaluation
Performs weight merging for trained PEFT adapters.
Automates evaluation of base and tuned seed models on Python, JS, C, C++, Go.
Calculates pass@1 (func@1), func-sec@1, and conditional security rate,
splits by CWE category (known vs novel), computes bootstrap confidence intervals,
compares base vs tuned via paired task-level McNemar/binomial flip checks,
and aggregates variance across seeds.
"""

import os
import json
import argparse
import subprocess
import sys
import random
import time
import re
import math
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
    parser.add_argument("--only_parse", action="store_true", help="Only parse already existing evaluation results without running active generation/eval")
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
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("PEFT / transformers not available for merging locally. Skipping merge step. Assuming merged models are already in place.")
        return

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
        
        del base_model
        del model
        del merged_model
        import gc
        gc.collect()
        
    print("\nPEFT adapters merged successfully.")

def filename_to_lang(path: str) -> str:
    normalized_path = path.replace("\\", "/")
    categories = ["core/c/", "core/cpp/", "core/go/", "core/py/", "core/js/", "lang/c"]
    for cat in categories:
        if cat in normalized_path:
            if "py" in cat: return "python"
            if "js" in cat: return "js"
            if "cpp" in cat: return "cpp"
            if "go" in cat: return "go"
            if "c/" in cat or "lang/c" in cat: return "c"
    
    # Fallback to name/extension check
    filename = os.path.splitext(os.path.basename(path))[0].lower()
    if "_py_" in filename or filename.endswith(".py"): return "python"
    if "_js_" in filename or filename.endswith(".js"): return "js"
    if "_go_" in filename or filename.endswith(".go"): return "go"
    if "_cpp_" in filename or filename.endswith(".cpp") or filename.endswith(".cc"): return "cpp"
    if "_c_" in filename or filename.endswith(".c"): return "c"
    return "unknown"

def extract_cwe_id(path: str) -> str:
    match = re.search(r'(cwe[_-]?\d+)', path, re.IGNORECASE)
    if match:
        return match.group(1).lower().replace("-", "_")
    return "unknown"

def normalize_task_id(task_id: str) -> str:
    path = task_id.replace("\\", "/")
    parts = path.split("/")
    for idx, part in enumerate(parts):
        if part.startswith("generated_"):
            return "/".join(parts[idx + 1:])
    return path

def exact_binomial_test(b, c):
    """
    Performs a two-tailed exact binomial test on discordant pairs b (degraded) and c (improved).
    Returns the p-value.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_val = 0.0
    for i in range(k + 1):
        p_val += math.comb(n, i) * (0.5 ** n)
    p_val = min(1.0, p_val * 2)
    return p_val

def bootstrap_ci(data, num_bootstraps=1000, ci=95):
    """
    Computes bootstrap confidence intervals (lower and upper percentiles) for a binary list.
    Returns values in percentage points [0.0, 100.0].
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
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    
    if args.api_base:
        os.environ["OPENAI_API_BASE"] = args.api_base
        print(f"Set environment OPENAI_API_BASE={args.api_base}")
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        print(f"Set environment OPENAI_API_KEY={args.api_key}")
        
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # 1. Merge adapters if active and not only_parse
    if not args.skip_merge and not args.only_parse:
        merge_peft_adapters(args.model_id, args.adapter_dir, args.merged_dir, seeds)

    # 2. Identify the training CWEs from dataset summary
    summary_path = Path(args.dataset_dir) / "dataset_summary.json"
    known_cwes = set()
    if summary_path.exists():
        with open(summary_path, "r") as f:
            meta = json.load(f)
        for task_id in meta.get("task_distribution", {}).keys():
            cwe = extract_cwe_id(task_id)
            if cwe != "unknown":
                known_cwes.add(cwe)
        print(f"Loaded {len(known_cwes)} known CWEs from training set: {sorted(list(known_cwes))}")
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
    
    # 4. Evaluation / Parsing Loop
    models_to_eval = {"base": args.model_id}
    for seed in seeds:
        models_to_eval[f"seed_{seed}"] = "openai/" + str(Path(args.merged_dir) / f"seed_{seed}")

    eval_results = {}

    for name, model_path in models_to_eval.items():
        eval_path = (Path(args.eval_base_dir) / name).resolve()
        res_file = eval_path / "res_all.json"
        
        if args.only_parse:
            print(f"\n[Only-Parse Mode] Checking existing results for: {name}")
            if not res_file.exists():
                print(f"res_all.json not found for {name} at {res_file}. Skipping.")
                continue
        else:
            print(f"\n==========================================")
            print(f"Evaluating Model Configuration: {name}")
            print(f"==========================================")
            
            server_proc = start_local_server(model_path, args.api_base)
            try:
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
        if not res_file.exists():
            print(f"Error: res_all.json not found for {name} at {res_file}", file=sys.stderr)
            continue
            
        with open(res_file, "r") as f:
            res_data = json.load(f)

        # Structure metrics and group tasks by language
        lang_data = {}
        for task_id, attempts in res_data.items():
            lang = filename_to_lang(task_id)
            cwe_id = extract_cwe_id(task_id)
            
            if lang not in lang_data:
                lang_data[lang] = []
                
            # Attempts contain "functional" and "secure" lists of booleans (size n=1)
            is_func = bool(attempts.get("functional", [False])[0])
            is_sec = bool(attempts.get("secure", [False])[0])
            
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
            func_secs = [t["functional"] and t["secure"] for t in tasks]
            sec_given_func = [t["secure"] for t in tasks if t["functional"]]

            pass1_rate = (sum(funcs) / len(funcs)) * 100.0 if funcs else 0.0
            func_secure_rate = (sum(func_secs) / len(func_secs)) * 100.0 if func_secs else 0.0
            security_rate = (sum(sec_given_func) / len(sec_given_func)) * 100.0 if sec_given_func else 0.0

            # Bootstrap CIs (95%)
            pass1_ci = bootstrap_ci(funcs, args.bootstrap_samples)
            func_sec_ci = bootstrap_ci(func_secs, args.bootstrap_samples)
            sec_ci = bootstrap_ci(sec_given_func, args.bootstrap_samples)

            # Splits known CWE vs novel CWE
            known_cwe_tasks = [t for t in tasks if t["cwe"] in known_cwes]
            novel_cwe_tasks = [t for t in tasks if t["cwe"] not in known_cwes]

            known_funcs = [t["functional"] for t in known_cwe_tasks]
            known_func_secs = [t["functional"] and t["secure"] for t in known_cwe_tasks]
            known_sec = [t["secure"] for t in known_cwe_tasks if t["functional"]]
            
            novel_funcs = [t["functional"] for t in novel_cwe_tasks]
            novel_func_secs = [t["functional"] and t["secure"] for t in novel_cwe_tasks]
            novel_sec = [t["secure"] for t in novel_cwe_tasks if t["functional"]]

            eval_results[name][lang] = {
                "total_tasks": len(tasks),
                "func_correct": sum(funcs),
                "func_secure": sum(func_secs),
                "pass1_rate": pass1_rate,
                "pass1_ci": pass1_ci,
                "func_secure_rate": func_secure_rate,
                "func_secure_ci": func_sec_ci,
                "security_rate": security_rate,
                "security_ci": sec_ci,
                "at_risk": lang in at_risk_langs,
                "risk_message": at_risk_langs.get(lang, "OK"),
                "known_cwe": {
                    "total_tasks": len(known_cwe_tasks),
                    "func_correct": sum(known_funcs),
                    "func_secure": sum(known_func_secs),
                    "pass1_rate": (sum(known_funcs) / len(known_funcs)) * 100.0 if known_funcs else 0.0,
                    "func_secure_rate": (sum(known_func_secs) / len(known_func_secs)) * 100.0 if known_func_secs else 0.0,
                    "security_rate": (sum(known_sec) / len(known_sec)) * 100.0 if known_sec else 0.0
                },
                "novel_cwe": {
                    "total_tasks": len(novel_cwe_tasks),
                    "func_correct": sum(novel_funcs),
                    "func_secure": sum(novel_func_secs),
                    "pass1_rate": (sum(novel_funcs) / len(novel_funcs)) * 100.0 if novel_funcs else 0.0,
                    "func_secure_rate": (sum(novel_func_secs) / len(novel_func_secs)) * 100.0 if novel_func_secs else 0.0,
                    "security_rate": (sum(novel_sec) / len(novel_sec)) * 100.0 if novel_sec else 0.0
                },
                "tasks_list": tasks
            }

    # 5. Paired Comparisons (Tuned vs Base)
    if "base" in eval_results:
        for seed in seeds:
            seed_name = f"seed_{seed}"
            if seed_name not in eval_results:
                continue
            for lang in eval_results[seed_name].keys():
                if lang == "tasks_list":
                    continue
                
                base_tasks = eval_results["base"].get(lang, {}).get("tasks_list", [])
                base_dict = {normalize_task_id(t["task_id"]): t for t in base_tasks}
                
                tuned_tasks = eval_results[seed_name][lang].get("tasks_list", [])
                
                improved_func = 0
                degraded_func = 0
                unchanged_func = 0
                
                improved_func_sec = 0
                degraded_func_sec = 0
                unchanged_func_sec = 0
                
                for t_tuned in tuned_tasks:
                    norm_id = normalize_task_id(t_tuned["task_id"])
                    if norm_id in base_dict:
                        t_base = base_dict[norm_id]
                        
                        bf = t_base["functional"]
                        tf = t_tuned["functional"]
                        if not bf and tf:
                            improved_func += 1
                        elif bf and not tf:
                            degraded_func += 1
                        else:
                            unchanged_func += 1
                            
                        bfs = t_base["functional"] and t_base["secure"]
                        tfs = t_tuned["functional"] and t_tuned["secure"]
                        if not bfs and tfs:
                            improved_func_sec += 1
                        elif bfs and not tfs:
                            degraded_func_sec += 1
                        else:
                            unchanged_func_sec += 1
                
                p_val_func = exact_binomial_test(degraded_func, improved_func)
                p_val_func_sec = exact_binomial_test(degraded_func_sec, improved_func_sec)
                
                eval_results[seed_name][lang]["comparison"] = {
                    "functional": {
                        "improved": improved_func,
                        "degraded": degraded_func,
                        "unchanged": unchanged_func,
                        "p_value": p_val_func
                    },
                    "func_secure": {
                        "improved": improved_func_sec,
                        "degraded": degraded_func_sec,
                        "unchanged": unchanged_func_sec,
                        "p_value": p_val_func_sec
                    }
                }

    # 6. Aggregate metrics across available seeds
    languages = list(eval_results["base"].keys()) if "base" in eval_results else []
    languages = [l for l in languages if l != "tasks_list"]
    
    aggregated_results = {
        "base": {},
        "seeds": {},
        "aggregate": {}
    }
    
    for lang in languages:
        base_info = eval_results["base"][lang].copy()
        if "tasks_list" in base_info:
            del base_info["tasks_list"]
        aggregated_results["base"][lang] = base_info
        
    for seed in seeds:
        seed_name = f"seed_{seed}"
        if seed_name in eval_results:
            aggregated_results["seeds"][seed_name] = {}
            for lang in eval_results[seed_name].keys():
                seed_info = eval_results[seed_name][lang].copy()
                if "tasks_list" in seed_info:
                    del seed_info["tasks_list"]
                aggregated_results["seeds"][seed_name][lang] = seed_info

    for lang in languages:
        seed_pass1s = []
        seed_func_secs = []
        seed_sec_rates = []
        
        seed_known_pass1s = []
        seed_known_func_secs = []
        seed_known_sec_rates = []
        
        seed_novel_pass1s = []
        seed_novel_func_secs = []
        seed_novel_sec_rates = []
        
        seed_improved_func = []
        seed_degraded_func = []
        seed_improved_func_sec = []
        seed_degraded_func_sec = []
        
        for seed in seeds:
            seed_name = f"seed_{seed}"
            if seed_name in eval_results and lang in eval_results[seed_name]:
                lang_res = eval_results[seed_name][lang]
                seed_pass1s.append(lang_res["pass1_rate"])
                seed_func_secs.append(lang_res["func_secure_rate"])
                seed_sec_rates.append(lang_res["security_rate"])
                
                seed_known_pass1s.append(lang_res["known_cwe"]["pass1_rate"])
                seed_known_func_secs.append(lang_res["known_cwe"]["func_secure_rate"])
                seed_known_sec_rates.append(lang_res["known_cwe"]["security_rate"])
                
                seed_novel_pass1s.append(lang_res["novel_cwe"]["pass1_rate"])
                seed_novel_func_secs.append(lang_res["novel_cwe"]["func_secure_rate"])
                seed_novel_sec_rates.append(lang_res["novel_cwe"]["security_rate"])
                
                comp = lang_res.get("comparison", {})
                if comp:
                    seed_improved_func.append(comp["functional"]["improved"])
                    seed_degraded_func.append(comp["functional"]["degraded"])
                    seed_improved_func_sec.append(comp["func_secure"]["improved"])
                    seed_degraded_func_sec.append(comp["func_secure"]["degraded"])
                    
        if seed_pass1s:
            aggregated_results["aggregate"][lang] = {
                "pass1": {
                    "mean": float(np.mean(seed_pass1s)),
                    "std": float(np.std(seed_pass1s)) if len(seed_pass1s) > 1 else 0.0,
                    "min": float(np.min(seed_pass1s)),
                    "max": float(np.max(seed_pass1s))
                },
                "func_secure_rate": {
                    "mean": float(np.mean(seed_func_secs)),
                    "std": float(np.std(seed_func_secs)) if len(seed_func_secs) > 1 else 0.0,
                    "min": float(np.min(seed_func_secs)),
                    "max": float(np.max(seed_func_secs))
                },
                "security_rate": {
                    "mean": float(np.mean(seed_sec_rates)),
                    "std": float(np.std(seed_sec_rates)) if len(seed_sec_rates) > 1 else 0.0,
                    "min": float(np.min(seed_sec_rates)),
                    "max": float(np.max(seed_sec_rates))
                },
                "known_cwe": {
                    "pass1": {"mean": float(np.mean(seed_known_pass1s)), "std": float(np.std(seed_known_pass1s)) if len(seed_known_pass1s) > 1 else 0.0},
                    "func_secure_rate": {"mean": float(np.mean(seed_known_func_secs)), "std": float(np.std(seed_known_func_secs)) if len(seed_known_func_secs) > 1 else 0.0},
                    "security_rate": {"mean": float(np.mean(seed_known_sec_rates)), "std": float(np.std(seed_known_sec_rates)) if len(seed_known_sec_rates) > 1 else 0.0}
                },
                "novel_cwe": {
                    "pass1": {"mean": float(np.mean(seed_novel_pass1s)), "std": float(np.std(seed_novel_pass1s)) if len(seed_novel_pass1s) > 1 else 0.0},
                    "func_secure_rate": {"mean": float(np.mean(seed_novel_func_secs)), "std": float(np.std(seed_novel_func_secs)) if len(seed_novel_func_secs) > 1 else 0.0},
                    "security_rate": {"mean": float(np.mean(seed_novel_sec_rates)), "std": float(np.std(seed_novel_sec_rates)) if len(seed_novel_sec_rates) > 1 else 0.0}
                },
                "comparison": {
                    "functional": {
                        "improved_mean": float(np.mean(seed_improved_func)) if seed_improved_func else 0.0,
                        "degraded_mean": float(np.mean(seed_degraded_func)) if seed_degraded_func else 0.0
                    },
                    "func_secure": {
                        "improved_mean": float(np.mean(seed_improved_func_sec)) if seed_improved_func_sec else 0.0,
                        "degraded_mean": float(np.mean(seed_degraded_func_sec)) if seed_degraded_func_sec else 0.0
                    }
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
