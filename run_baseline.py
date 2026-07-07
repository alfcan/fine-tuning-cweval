#!/usr/bin/env python3
"""
run_baseline.py - Phase 1: Baseline Check
Evaluates the untouched base model on all 5 CWEval language subsets.
Computes functional correctness (pass@1) and security-rate.
Flags "at risk" languages with degenerate competency.
"""

import os
import json
import argparse
import subprocess
import sys
import time
import requests
import urllib.parse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline check on CWEval")
    parser.add_argument("--model", type=str, default="openai/Qwen/Qwen3.5-2B", help="Model to evaluate")
    parser.add_argument("--eval_path", type=str, default="results/baseline", help="Directory to save evaluation outputs")
    parser.add_argument("--docker", type=str, default="True", choices=["True", "False"], help="Run CWEval evaluation inside Docker")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel processes for evaluation")
    parser.add_argument("--cweval_dir", type=str, default="CWEval", help="Path to cloned CWEval directory")
    parser.add_argument("--api_base", type=str, default="http://localhost:1234/v1", help="Local model server API base url")
    parser.add_argument("--api_key", type=str, default="sk-local-research", help="API key for inference server")
    return parser.parse_args()

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

def check_func_and_sec(attempt):
    """
    Robustly parses an evaluation attempt dict to extract functionality and security outcomes.
    """
    if isinstance(attempt, dict):
        # Determine functional correctness
        func = None
        for k in ["func", "func_correct", "functional", "correct", "passed", "pass", "functional_correctness"]:
            if k in attempt:
                func = bool(attempt[k])
                break
        if func is None:
            # Fallback
            func = attempt.get("status") == "success" or attempt.get("exit_code") == 0

        # Determine security correctness
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
        
        # If still None, default to False/True conservatively or extract first available bool
        if func is None: func = False
        if sec is None: sec = False
        return func, sec

    elif isinstance(attempt, (list, tuple)):
        if len(attempt) >= 2:
            return bool(attempt[0]), bool(attempt[1])
        elif len(attempt) == 1:
            return bool(attempt[0]), True
    elif isinstance(attempt, bool):
        return attempt, True
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
    
    # Verify CWEval directory
    cweval_path = Path(args.cweval_dir).resolve()
    if not cweval_path.exists():
        print(f"CWEval repository not found at {cweval_path}. Please run setup_env.sh first.", file=sys.stderr)
        sys.exit(1)

    print(f"=== Starting Baseline Evaluation for model: {args.model} ===")
    
    # Start the local model server
    server_proc = start_local_server(args.model, args.api_base)
    
    try:
        # CWEval scripts use relative paths internally (BENCHMARK_DIR = 'benchmark'),
        # so all subprocesses must run with cwd=CWEval. We use absolute eval_path to
        # ensure outputs land in the right place regardless of cwd.
        eval_path_abs = str(Path(args.eval_path).resolve())
        
        # Clear directory to prevent generate.py from prompting for overwrite
        eval_dir = Path(eval_path_abs)
        if eval_dir.exists():
            import shutil
            print(f"Clearing existing evaluation directory: {eval_path_abs}")
            shutil.rmtree(eval_dir)
        
        # 1. Run generation using our wrapper script (cwd must be CWEval for benchmark discovery)
        gen_script = str(Path("run_generation.py").resolve())
        gen_cmd = [
            sys.executable, gen_script, "gen",
            "--model", args.model,
            "--n", "1",
            "--eval_path", eval_path_abs,
            "--temperature", "0.0",
            "--api_base", args.api_base,
            "--api_key", args.api_key,
            "--num_proc", str(args.num_proc)
        ]
        run_command(gen_cmd, cwd=str(cweval_path))
        
        # 2. Run evaluation pipeline
        if args.docker == "True":
            from cweval_orchestrator import run_evaluation_in_docker
            run_evaluation_in_docker(eval_path_abs, num_proc=args.num_proc)
        else:
            eval_script = str(cweval_path / "cweval" / "evaluate.py")
            eval_cmd = [
                sys.executable, eval_script, "pipeline",
                "--eval_path", eval_path_abs,
                "--num_proc", str(args.num_proc),
                "--docker", "False"
            ]
            run_command(eval_cmd, cwd=str(cweval_path))
    finally:
        print("Stopping local model server...")
        server_proc.terminate()
        server_proc.wait()

    # 3. Parse res_all.json
    res_file = Path(args.eval_path) / "res_all.json"
    if not res_file.exists():
        print(f"Error: Evaluation output file {res_file} not found.", file=sys.stderr)
        sys.exit(1)
        
    with open(res_file, "r") as f:
        res_data = json.load(f)

    # Group tasks by language/category matching benchmark definitions
    metrics_by_lang = {}
    
    def filename_to_lang(path: str) -> str:
        # Normalize slashes to forward slashes
        normalized_path = path.replace("\\", "/")
        # These match standard benchmark category filters in evaluate.py
        categories = ["core/c/", "core/cpp/", "core/go/", "core/py/", "core/js/", "lang/c"]
        for cat in categories:
            if cat in normalized_path:
                return cat
        
        # Fallback to basename extraction
        filename = os.path.splitext(os.path.basename(path))[0]
        lang = filename.split('_')[-2]
        if lang.isdigit():
            return 'py'
        return lang

    for task_id, attempts in res_data.items():
        # Identify language from task_id
        lang = filename_to_lang(task_id)
            
        if lang not in metrics_by_lang:
            metrics_by_lang[lang] = {
                "total_tasks": 0,
                "functional_correct": 0,
                "secure_given_correct": 0,
                "total_secure": 0,
                "vulnerable_correct": 0
            }
            
        # We did n=1 sampling for baseline
        n_samples = len(attempts.get("functional", []))
        metrics_by_lang[lang]["total_tasks"] += 1
        
        for idx in range(n_samples):
            is_func = attempts["functional"][idx]
            is_sec = attempts["secure"][idx]
            if is_func:
                metrics_by_lang[lang]["functional_correct"] += 1
                if is_sec:
                    metrics_by_lang[lang]["secure_given_correct"] += 1
                else:
                    metrics_by_lang[lang]["vulnerable_correct"] += 1
            if is_sec:
                metrics_by_lang[lang]["total_secure"] += 1

    # Also compute overall/all metrics
    if metrics_by_lang:
        total_tasks = sum(m["total_tasks"] for m in metrics_by_lang.values())
        functional_correct = sum(m["functional_correct"] for m in metrics_by_lang.values())
        secure_given_correct = sum(m["secure_given_correct"] for m in metrics_by_lang.values())
        total_secure = sum(m["total_secure"] for m in metrics_by_lang.values())
        vulnerable_correct = sum(m["vulnerable_correct"] for m in metrics_by_lang.values())
        metrics_by_lang["all"] = {
            "total_tasks": total_tasks,
            "functional_correct": functional_correct,
            "secure_given_correct": secure_given_correct,
            "total_secure": total_secure,
            "vulnerable_correct": vulnerable_correct
        }

    # Print summary and compute percentages
    summary = {}
    print("\n=== Baseline Results Summary ===")
    for lang, metrics in metrics_by_lang.items():
        total = metrics["total_tasks"]
        func = metrics["functional_correct"]
        sec_given_func = metrics["secure_given_correct"]
        
        pass1_rate = (func / total) * 100.0 if total > 0 else 0.0
        sec_rate = (sec_given_func / func) * 100.0 if func > 0 else 0.0
        
        # Check for degenerate pass@1 (noisy signals)
        risk = "OK"
        if pass1_rate < 10.0:
            risk = "RISK (Near-Floor Functional Competency)"
        elif pass1_rate > 95.0:
            risk = "RISK (Near-Ceiling Functional Competency)"
            
        summary[lang] = {
            "total_tasks": total,
            "functional_correct": func,
            "pass@1_rate": pass1_rate,
            "secure_given_correct": sec_given_func,
            "security_rate_of_correct": sec_rate,
            "risk_status": risk
        }
        
        print(f"Language: {lang.upper()}")
        print(f"  Total Tasks: {total}")
        print(f"  Pass@1: {pass1_rate:.2f}% ({func}/{total})")
        print(f"  Security-rate (on correct completions): {sec_rate:.2f}% ({sec_given_func}/{func})")
        print(f"  Status: {risk}")
        print("-" * 40)

    # Save summary report
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Baseline summary written to {summary_path}")

if __name__ == "__main__":
    main()
