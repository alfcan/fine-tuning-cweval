# Implementation Plan: Preference-based Security Alignment via IPO on CWEval

## Research Question
Do security preferences learned via IPO on Python tasks from CWEval transfer to unseen languages (JS, C, C++, Go)? Does transfer depend on whether the CWE category was seen during training or is entirely novel?

---

## Phase 0 — Environment Setup
- [ ] Pull and run the official CWEval Docker image (`co1lin/cweval`) — handles compilation/execution uniformly across all 5 languages.
- [ ] Set up target model:  Qwen3 Coder (qwen/qwen3-coder-30b via LM Studio) + LoRA/PEFT.
- [ ] Install TRL (for `DPOTrainer` with `loss_type="ipo"`), plus standard HF stack (transformers, peft, accelerate, bitsandbytes if quantizing).
- [ ] Clone CWEval repo, verify `evaluate.py pipeline` runs end-to-end on a sample task per language (sanity check before scaling).
- [ ] Set up a sandboxed execution wrapper (subprocess w/ timeout, resource limits) if anything needs to run outside the Docker harness (e.g., ad-hoc sampling generation scripts).

## Phase 1 — Baseline Check (before building any dataset)
- [ ] Run the **untouched base model** on all 5 CWEval language subsets (Python, JS, C, C++, Go).
- [ ] Compute pass@1 (functional correctness) per language.
- [ ] Compute baseline security-rate per language (using CWEval's dual oracle) as a reference point for later comparison.
- [ ] Flag any language with degenerate pass@1 (near-floor or near-ceiling) as "at risk" for noisy signal — do not exclude yet, just note for later interpretation.
- [ ] Save all baseline outputs/logs — needed later for the base-vs-tuned comparison.

## Phase 2 — Preference Pair Construction (Python subset, 25 tasks only)
- [ ] For each Python task, generate N completions via **on-policy temperature sampling** using the same base model that will later be fine-tuned. Use multiple temperatures (e.g., 0.4, 0.6, 0.8, 1.0) rather than a single value, to increase real diversity.
- [ ] If a task fails to yield both a secure and a vulnerable *functionally correct* completion after standard sampling, retry with 2-3 paraphrased variants of the task prompt (same target CWE, different phrasing) before giving up on it.
- [ ] Label every completion with CWEval's dual oracle: functional (pass/fail) and security (secure/vulnerable).
- [ ] Construct pairs **only** between functionally correct completions of the same task: chosen = secure, rejected = vulnerable. Do not pair against broken/non-compiling code.
- [ ] Apply a **per-task cap** (e.g., max 8-10 pairs/task) so that a few "easy" tasks don't dominate the dataset.
- [ ] **Deduplicate** near-identical pairs (normalize whitespace/formatting, drop exact/near-exact duplicates).
- [ ] Log final pair count, and per-task pair distribution (to confirm the cap worked as intended and check balance).
- Expected output: ~150-250 balanced, deduplicated preference pairs.

## Phase 3 — Data Splitting
- [ ] Split the Python pairs into **train / validation** (~80/20), **splitting by task**, not randomly at the pair level, to avoid leakage between train and validation from the same underlying task.
- [ ] Keep this validation set strictly for monitoring training (not the final held-out evaluation).

## Phase 4 — Training
- [ ] Configure LoRA with a **low rank** (r=4-8) to limit degrees of freedom given the small dataset.
- [ ] Use `DPOTrainer` (TRL) with `loss_type="ipo"`.
- [ ] Use a conservative learning rate, 1-3 epochs max.
- [ ] Monitor validation loss (Phase 3 split) during training; apply **early stopping** if it diverges or plateaus while training loss keeps decreasing (signal that the dataset is too small/homogeneous for that config).
- [ ] Train with **3-5 independent seeds** to distinguish a real signal from noise — this is not optional given the small dataset size.
- [ ] Save all checkpoints + training/validation loss curves per seed.

## Phase 5 — Evaluation
- [ ] Run both **base model** and each **IPO-tuned seed model** on:
  - Python (in-distribution sanity check — did training break anything or overfit visibly?)
  - JS, C, C++, Go (held-out, never seen in training)
- [ ] Compute per language: pass@1 (functional) and security-rate (via CWEval's dual oracle), for both base and tuned models.
- [ ] Report bootstrap confidence intervals given the small task counts per language.
- [ ] For each held-out language, split results into two groups based on task metadata (CWE type):
  - CWEs also present in the Python training set (**known-pattern transfer**)
  - CWEs never seen in any language during training (**true generalization**)
- [ ] Aggregate variance across the 3-5 training seeds as an indicator of result stability (IPO should show tighter variance than vanilla DPO would — worth noting if true).
- [ ] Cross-reference with Phase 1 baseline flags — treat results on "at risk" languages (degenerate baseline pass@1) with explicit caution in the write-up.

## Phase 6 — Analysis and Write-up
- [ ] Build a summary table: per held-out language — baseline pass@1 (Phase 1), security-rate base vs. tuned, split by CWE-known vs. CWE-novel.
- [ ] Interpret results distinguishing "transfer of a known pattern to new syntax" from "genuine generalization to unseen vulnerability categories."
- [ ] Explicitly state limitations:
  - Small training set (~150-250 pairs), all derived from only 25 underlying Python tasks.
  - Result quality depends on the base model's exploration diversity during sampling (Phase 2).
  - Only one base model tested.
  - Possible noise on held-out languages with degenerate baseline competence (Phase 1 flags).
- [ ] Include an explicit methodological note: standard DPO/IPO literature assumes datasets of thousands of pairs; this study tests whether a measurable security-alignment signal emerges even at much smaller scale, motivated by prior work (SafeCoder, ~465 examples via SFT) showing that narrow security fine-tuning can work with modest datasets.
- [ ] Add a "future work" note explicitly framing verified data-synthesis pipelines (Secure-Instruct, HexaCoder — generating new tasks per CWE with verified oracles) as the natural next step to scale beyond this pilot's data limitation — this directly addresses the professor's suggestion as a follow-up, not a competing approach.

---

## Deliverables Checklist
- [ ] Baseline results (Phase 1) — table + raw logs.
- [ ] Preference pair dataset (Phase 2-3) — with per-task pair counts and dedup stats.
- [ ] Trained model checkpoints (Phase 4) — 3-5 seeds, with loss curves.
- [ ] Evaluation results (Phase 5) — per-language, per-CWE-group tables with confidence intervals.
- [ ] Final write-up (Phase 6) — results, limitations, future work.