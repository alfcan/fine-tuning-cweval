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
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline check on CWEval")
    parser.add_argument("--model", type=str, default="qwen/qwen3-coder-30b", help="Model to evaluate")
    parser.add_argument("--eval_path", type=str, default="results/baseline", help="Directory to save evaluation outputs")
    parser.add_argument("--docker", type=str, default="False", choices=["True", "False"], help="Run CWEval evaluation inside Docker")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel processes for evaluation")
    parser.add_argument("--cweval_dir", type=str, default="CWEval", help="Path to cloned CWEval directory")
    return parser.parse_args()

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
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
        print(f"STDOUT:\n{result.stdout}", file=sys.stderr)
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout

def main():
    args = parse_args()
    
    # Verify CWEval directory
    cweval_path = Path(args.cweval_dir)
    if not cweval_path.exists():
        print(f"CWEval repository not found at {cweval_path}. Please run setup_env.sh first.", file=sys.stderr)
        sys.exit(1)

    print(f"=== Starting Baseline Evaluation for model: {args.model} ===")
    
    # 1. Run generation using CWEval's script
    # cweval/generate.py gen --model <model> --n 1 --eval_path <eval_path> --temperature 0.0
    gen_script = str(cweval_path / "cweval" / "generate.py")
    gen_cmd = [
        sys.executable, gen_script, "gen",
        "--model", args.model,
        "--n", "1",
        "--eval_path", args.eval_path,
        "--temperature", "0.0"
    ]
    run_command(gen_cmd)
    
    # 2. Run evaluation pipeline using CWEval's script
    eval_script = str(cweval_path / "cweval" / "evaluate.py")
    eval_cmd = [
        sys.executable, eval_script, "pipeline",
        "--eval_path", args.eval_path,
        "--num_proc", str(args.num_proc),
        "--docker", args.docker
    ]
    run_command(eval_cmd)

    # 3. Parse res_all.json
    res_file = Path(args.eval_path) / "res_all.json"
    if not res_file.exists():
        print(f"Error: Evaluation output file {res_file} not found.", file=sys.stderr)
        sys.exit(1)
        
    with open(res_file, "r") as f:
        res_data = json.load(f)

    # Group tasks by language
    # Typically, task_id looks like: "python/cwe-022/task_0"
    metrics_by_lang = {}
    
    for task_id, attempts in res_data.items():
        # Identify language from task_id
        parts = task_id.split("/")
        if len(parts) >= 1:
            lang = parts[0]
        else:
            lang = "unknown"
            
        if lang not in metrics_by_lang:
            metrics_by_lang[lang] = {
                "total_tasks": 0,
                "functional_correct": 0,
                "secure_given_correct": 0,
                "total_secure": 0,
                "vulnerable_correct": 0
            }
            
        metrics_by_lang[lang]["total_tasks"] += 1
        
        # We did n=1 sampling for baseline
        for attempt in attempts:
            is_func, is_sec = check_func_and_sec(attempt)
            if is_func:
                metrics_by_lang[lang]["functional_correct"] += 1
                if is_sec:
                    metrics_by_lang[lang]["secure_given_correct"] += 1
                else:
                    metrics_by_lang[lang]["vulnerable_correct"] += 1
            if is_sec:
                metrics_by_lang[lang]["total_secure"] += 1

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
