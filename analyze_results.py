#!/usr/bin/env python3
"""
analyze_results.py - Phase 6: Analysis and Report Generator
Parses evaluation summary results and compiles a detailed Markdown report
summarizing baseline correctness, base vs tuned security-rates, known vs novel CWE splits,
task-level paired comparison flips (improved vs degraded) with exact binomial test significance,
regression analysis, risk warnings, and methodological notes.
"""

import os
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6: Result Analysis and Report compiler")
    parser.add_argument("--eval_summary", type=str, default="results/eval_summary.json", help="Path to evaluation summary JSON")
    parser.add_argument("--output_report", type=str, default="results/report.md", help="Path to output markdown report")
    return parser.parse_args()

def main():
    args = parse_args()
    summary_path = Path(args.eval_summary)
    
    if not summary_path.exists():
        print(f"Error: Summary file {summary_path} not found. Please run run_evaluation.py first.")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    base_data = data.get("base", {})
    seeds_data = data.get("seeds", {})
    agg_data = data.get("aggregate", {})
    
    available_seeds = sorted(list(seeds_data.keys()))
    print(f"Found available seeds for report generation: {available_seeds}")

    # 1. Main Quantitative Results Table
    table_lines = [
        "| Language | Model | Func@1 (Pass@1) (%) | Func-Sec@1 (%) | Sec Rate (Cond.) (%) |",
        "| :--- | :--- | :---: | :---: | :---: |"
    ]
    
    languages = sorted(list(base_data.keys()))
    
    for lang in languages:
        b_info = base_data.get(lang, {})
        # Base Model row
        b_p1 = b_info.get("pass1_rate", 0.0)
        b_p1_ci = b_info.get("pass1_ci", [0.0, 0.0])
        b_fs = b_info.get("func_secure_rate", 0.0)
        b_fs_ci = b_info.get("func_secure_ci", [0.0, 0.0])
        b_sec = b_info.get("security_rate", 0.0)
        b_sec_ci = b_info.get("security_ci", [0.0, 0.0])
        
        table_lines.append(
            f"| **{lang.upper()}** | Base | {b_p1:.1f}% [CI: {b_p1_ci[0]:.1f}-{b_p1_ci[1]:.1f}] | {b_fs:.1f}% [CI: {b_fs_ci[0]:.1f}-{b_fs_ci[1]:.1f}] | {b_sec:.1f}% [CI: {b_sec_ci[0]:.1f}-{b_sec_ci[1]:.1f}] |"
        )
        
        # Individual Seeds rows
        for seed_name in available_seeds:
            s_info = seeds_data[seed_name].get(lang, {})
            s_p1 = s_info.get("pass1_rate", 0.0)
            s_p1_ci = s_info.get("pass1_ci", [0.0, 0.0])
            s_fs = s_info.get("func_secure_rate", 0.0)
            s_fs_ci = s_info.get("func_secure_ci", [0.0, 0.0])
            s_sec = s_info.get("security_rate", 0.0)
            s_sec_ci = s_info.get("security_ci", [0.0, 0.0])
            
            table_lines.append(
                f"| | Tuned ({seed_name}) | {s_p1:.1f}% [CI: {s_p1_ci[0]:.1f}-{s_p1_ci[1]:.1f}] | {s_fs:.1f}% [CI: {s_fs_ci[0]:.1f}-{s_fs_ci[1]:.1f}] | {s_sec:.1f}% [CI: {s_sec_ci[0]:.1f}-{s_sec_ci[1]:.1f}] |"
            )
            
        # Aggregate row (if multiple seeds are available)
        if len(available_seeds) > 1:
            a_info = agg_data.get(lang, {})
            a_p1_mean = a_info.get("pass1", {}).get("mean", 0.0)
            a_p1_std = a_info.get("pass1", {}).get("std", 0.0)
            a_fs_mean = a_info.get("func_secure_rate", {}).get("mean", 0.0)
            a_fs_std = a_info.get("func_secure_rate", {}).get("std", 0.0)
            a_sec_mean = a_info.get("security_rate", {}).get("mean", 0.0)
            a_sec_std = a_info.get("security_rate", {}).get("std", 0.0)
            
            table_lines.append(
                f"| | *Tuned (Mean±Std)* | *{a_p1_mean:.1f}±{a_p1_std:.1f}%* | *{a_fs_mean:.1f}±{a_fs_std:.1f}%* | *{a_sec_mean:.1f}±{a_sec_std:.1f}%* |"
            )
            
    main_results_table = "\n".join(table_lines)

    # 2. CWE-Known vs CWE-Novel Splits Table (for held-out languages)
    held_out_langs = [l for l in languages if l != "python"]
    split_lines = [
        "| Language | Model | CWE Split | Total Tasks | Func@1 (Pass@1) (%) | Func-Sec@1 (%) | Sec Rate (Cond.) (%) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    
    for lang in held_out_langs:
        # Base model known vs novel
        b_info = base_data.get(lang, {})
        b_k = b_info.get("known_cwe", {})
        b_n = b_info.get("novel_cwe", {})
        
        split_lines.append(
            f"| **{lang.upper()}** | Base | Known CWE | {b_k.get('total_tasks', 0)} | {b_k.get('pass1_rate', 0.0):.1f}% | {b_k.get('func_secure_rate', 0.0):.1f}% | {b_k.get('security_rate', 0.0):.1f}% |"
        )
        split_lines.append(
            f"| | Base | Novel CWE | {b_n.get('total_tasks', 0)} | {b_n.get('pass1_rate', 0.0):.1f}% | {b_n.get('func_secure_rate', 0.0):.1f}% | {b_n.get('security_rate', 0.0):.1f}% |"
        )
        
        # Individual Seeds known vs novel
        for seed_name in available_seeds:
            s_info = seeds_data[seed_name].get(lang, {})
            s_k = s_info.get("known_cwe", {})
            s_n = s_info.get("novel_cwe", {})
            
            split_lines.append(
                f"| | Tuned ({seed_name}) | Known CWE | {s_k.get('total_tasks', 0)} | {s_k.get('pass1_rate', 0.0):.1f}% | {s_k.get('func_secure_rate', 0.0):.1f}% | {s_k.get('security_rate', 0.0):.1f}% |"
            )
            split_lines.append(
                f"| | Tuned ({seed_name}) | Novel CWE | {s_n.get('total_tasks', 0)} | {s_n.get('pass1_rate', 0.0):.1f}% | {s_n.get('func_secure_rate', 0.0):.1f}% | {s_n.get('security_rate', 0.0):.1f}% |"
            )
            
        # Aggregate splits
        if len(available_seeds) > 1:
            a_info = agg_data.get(lang, {})
            ak_p1 = a_info.get("known_cwe", {}).get("pass1", {}).get("mean", 0.0)
            ak_fs = a_info.get("known_cwe", {}).get("func_secure_rate", {}).get("mean", 0.0)
            ak_sec = a_info.get("known_cwe", {}).get("security_rate", {}).get("mean", 0.0)
            
            an_p1 = a_info.get("novel_cwe", {}).get("pass1", {}).get("mean", 0.0)
            an_fs = a_info.get("novel_cwe", {}).get("func_secure_rate", {}).get("mean", 0.0)
            an_sec = a_info.get("novel_cwe", {}).get("security_rate", {}).get("mean", 0.0)
            
            split_lines.append(
                f"| | *Tuned (Mean)* | *Known CWE* | - | *{ak_p1:.1f}%* | *{ak_fs:.1f}%* | *{ak_sec:.1f}%* |"
            )
            split_lines.append(
                f"| | *Tuned (Mean)* | *Novel CWE* | - | *{an_p1:.1f}%* | *{an_fs:.1f}%* | *{an_sec:.1f}%* |"
            )
            
    splits_table = "\n".join(split_lines)

    # 3. Paired Task Comparison Table
    paired_lines = [
        "| Seed | Language | Metric Type | Improved Flips (0→1) | Degraded Flips (1→0) | Unchanged | McNemar/Binomial p-value |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    
    for seed_name in available_seeds:
        for lang in languages:
            s_info = seeds_data[seed_name].get(lang, {})
            comp = s_info.get("comparison", {})
            if not comp:
                continue
            
            # Correctness
            c_imp = comp["functional"]["improved"]
            c_deg = comp["functional"]["degraded"]
            c_unc = comp["functional"]["unchanged"]
            c_p = comp["functional"]["p_value"]
            
            # Correctness + Security
            s_imp = comp["func_secure"]["improved"]
            s_deg = comp["func_secure"]["degraded"]
            s_unc = comp["func_secure"]["unchanged"]
            s_p = comp["func_secure"]["p_value"]
            
            # Formatting significance stars
            c_stars = " (p < 0.05) *" if c_p < 0.05 else ""
            s_stars = " (p < 0.05) *" if s_p < 0.05 else ""
            
            paired_lines.append(
                f"| {seed_name} | **{lang.upper()}** | Func@1 (Correctness) | {c_imp} | {c_deg} | {c_unc} | {c_p:.4f}{c_stars} |"
            )
            paired_lines.append(
                f"| | | Func-Sec@1 (Corr + Sec) | {s_imp} | {s_deg} | {s_unc} | {s_p:.4f}{s_stars} |"
            )
            
    paired_table = "\n".join(paired_lines)

    # 4. Regression Analysis section
    regression_items = []
    for seed_name in available_seeds:
        for lang in languages:
            s_info = seeds_data[seed_name].get(lang, {})
            comp = s_info.get("comparison", {})
            if not comp:
                continue
            
            c_deg = comp["functional"]["degraded"]
            s_deg = comp["func_secure"]["degraded"]
            
            if c_deg > 0 or s_deg > 0:
                regression_items.append(
                    f"- **{seed_name}** su **{lang.upper()}**: ha causato la regressione funzionale di {c_deg} task (che il modello base risolveva correttamente, ma il tuned ha sbagliato) e la regressione di sicurezza di {s_deg} task (che prima erano corretti+sicuri e ora non lo sono più)."
                )
    
    if not regression_items:
        regression_analysis_md = "Nessuna regressione significativa rilevata nei seed disponibili rispetto al modello base."
    else:
        regression_analysis_md = "\n".join(regression_items)

    # 5. Risk Flag warnings
    risk_warnings = []
    for lang in languages:
        b_info = base_data.get(lang, {})
        b_p1 = b_info.get("pass1_rate", 0.0)
        
        if b_p1 == 0.0:
            risk_warnings.append(
                f"> [!WARNING]\n"
                f"> **Linguaggio a Rischio Elevato: {lang.upper()}**\n"
                f"> Il modello base ha mostrato una correttezza funzionale `Func@1 = 0.00%` (0 su {b_info.get('total_tasks', 0)} task risolti). "
                f"Ciò significa che il modello non è in grado di produrre codice Go funzionante per questo benchmark. "
                f"Qualsiasi metrica di sicurezza calcolata su questo linguaggio è priva di valore statistico e non deve essere utilizzata per trarre conclusioni sull'efficacia dell'allineamento."
            )
        elif b_p1 < 10.0:
            risk_warnings.append(
                f"> [!NOTE]\n"
                f"> **Linguaggio a Basso Competency: {lang.upper()}**\n"
                f"> Il modello ha una correttezza funzionale estremamente bassa (`{b_p1:.1f}%`). I calcoli sulla sicurezza condizionata (`Sec Rate (Cond.)`) sono soggetti a forte rumore statistico dovuto a un campione molto ridotto."
            )
            
    risk_section_md = "\n\n".join(risk_warnings) if risk_warnings else "Nessun avviso di rischio critico identificato per i linguaggi valutati."

    # Build the report
    report_content = f"""# Relazione di Valutazione: Allineamento di Sicurezza Preference-based (IPO) su CWEval

Questo report analizza il trasferimento di preferenze di sicurezza apprese tramite Identity Preference Optimization (IPO) da un sottoinsieme di addestramento in Python verso linguaggi di programmazione non visti durante il tuning (JavaScript, C, C++, Go) sul benchmark CWEval.

---

## 1. Risultati Quantitativi Globali

Le metriche principali comprendono:
* **Func@1 (Pass@1)**: la correttezza funzionale (codice che compila ed esegue con successo).
* **Func-Sec@1**: la percentuale di codici che sono **sia** funzionalmente corretti **sia** privi di vulnerabilità (sicuri).
* **Sec Rate (Cond.)**: il tasso di sicurezza condizionato al fatto che il codice funzioni (`Func-Sec@1 / Func@1`).
* **CIs**: Intervallo di confidenza bootstrap al 95%.

{main_results_table}

---

## 2. Generalizzazione a CWE Note vs CWE Nuove (Held-out)

Per comprendere se il modello apprende pattern di vulnerabilità generici o una sintassi difensiva specifica per determinati CWE, i task dei linguaggi held-out sono stati suddivisi in:
* **CWE-Known**: categorie presenti nel training set in Python (es. `cwe_020`, `cwe_022`).
* **CWE-Novel**: categorie mai viste dal modello durante il fine-tuning.

{splits_table}

---

## 3. Analisi Statistica Appaiata (Task-level Flips)

Il confronto tra percentuali aggregate può mascherare i reali cambiamenti microscopici. Questa tabella riporta quanti task singoli hanno cambiato stato per ciascun seed rispetto al modello base, applicando il test binomiale esatto sulle coppie discordanti.

{paired_table}

*Nota: `*` indica che la differenza è statisticamente significativa con p < 0.05.*

---

## 4. Controlli di Regressione e Degrado

Il tuning focalizzato sulla sicurezza comporta il rischio reale di degradare le capacità funzionali generali o introdurre insicurezze su task precedentemente stabili.

{regression_analysis_md}

---

## 5. Linguaggi a Rischio e Limiti Metodologici

{risk_section_md}

### Limiti Principali
1. **Dimensione Campionaria**: La ristrettezza dei task per ciascun linguaggio (19-25 task) rende i tassi molto sensibili a variazioni minime.
2. **Effetto del Seed**: La variabilità tra i seed dimostra come l'ottimizzazione con segnali deboli (dataset piccolo) risenta delle condizioni iniziali del tuning.
"""

    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)

    print(f"\nReport generated and written to {output_path}")

if __name__ == "__main__":
    main()
