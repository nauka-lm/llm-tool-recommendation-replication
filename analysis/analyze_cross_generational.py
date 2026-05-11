"""
Cross-Generational Analysis Script for Paper
Loads all 4 phases (Gen1 standard/thinking + Gen2 standard/thinking)
and produces comprehensive comparison tables for the paper.

Usage:
    python analyze_cross_generational.py
"""

import json
import glob
import os
import sys
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..", "data", "results")

PHASES = {
    "gen1_standard": os.path.join(BASE, "phase1"),
    "gen1_thinking": os.path.join(BASE, "phase2"),
    "gen2_standard": os.path.join(BASE, "phase3a"),
    "gen2_thinking": os.path.join(BASE, "phase3b"),
}

# Display names
MODEL_NAMES = {
    ("openai", "standard", "gen1"): "GPT-4.1",
    ("openai", "thinking", "gen1"): "o4-mini",
    ("anthropic", "standard", "gen1"): "Claude Sonnet 4.5",
    ("anthropic", "thinking", "gen1"): "Claude Sonnet 4.5 (think)",
    ("google", "standard", "gen1"): "Gemini 2.5 Flash-Lite",
    ("google", "thinking", "gen1"): "Gemini 2.5 Flash",
    ("openai", "standard", "gen2"): "GPT-5.2",
    ("openai", "thinking", "gen2"): "GPT-5.2 (thinking)",
    ("anthropic", "standard", "gen2"): "Claude Sonnet 4.6",
    ("anthropic", "thinking", "gen2"): "Claude Sonnet 4.6 (thinking)",
    ("google", "standard", "gen2"): "Gemini 3.1 Flash-Lite",
    ("google", "thinking", "gen2"): "Gemini 3.1 Flash-Lite (thinking)",
}

PROVIDERS = ["openai", "anthropic", "google"]
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
DOMAINS = ["D1", "D2", "D3", "D4"]


def load_results(results_dir):
    results = {}
    if not os.path.exists(results_dir):
        return results
    for f in sorted(glob.glob(os.path.join(results_dir, "results_*.json"))):
        name = os.path.basename(f).replace("results_", "").replace(".json", "")
        with open(f) as fh:
            data = json.load(fh)
        results[name] = data
    return results


def compute_metrics(queries):
    valid = [q for q in queries if not q.get("Error")]
    if not valid:
        return None

    total_tools = sum(len(q.get("MentionedToolNames", [])) for q in valid)
    total_halluc = sum(len(q.get("HallucinatedToolNames", [])) for q in valid)
    total_grounded = sum(len(q.get("GroundedToolNames", [])) for q in valid)

    hr = (total_halluc / total_tools * 100) if total_tools > 0 else 0
    gr = (total_grounded / total_tools * 100) if total_tools > 0 else 0

    all_stages = {"Selection", "Simulation", "Calculation", "Verification", "Manufacturing", "Prototyping"}
    avg_wc = sum(
        len(set(q.get("WorkflowStages", [])) & all_stages) / len(all_stages)
        for q in valid
    ) / len(valid) * 100

    avg_input = sum(q.get("PromptTokens", 0) for q in valid) / len(valid)
    avg_output = sum(q.get("CompletionTokens", 0) for q in valid) / len(valid)
    avg_think = sum(q.get("ThinkingTokens", 0) for q in valid) / len(valid)
    avg_latency = sum(q.get("LatencyMs", 0) for q in valid) / len(valid)
    avg_tools = total_tools / len(valid)

    json_valid = sum(1 for q in valid if q.get("IsValidJson", False))
    json_rate = json_valid / len(valid) * 100

    return {
        "valid_queries": len(valid),
        "errors": len(queries) - len(valid),
        "total_tools": total_tools,
        "total_halluc": total_halluc,
        "total_grounded": total_grounded,
        "hr": hr,
        "gr": gr,
        "wc": avg_wc,
        "avg_tools_per_query": avg_tools,
        "avg_input_tokens": avg_input,
        "avg_output_tokens": avg_output,
        "avg_thinking_tokens": avg_think,
        "avg_latency_ms": avg_latency,
        "json_valid_rate": json_rate,
    }


def parse_key(key):
    parts = key.split("_")
    return parts[0], parts[1], parts[2], parts[3]


def p(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def get_hr(metrics, provider, category, gen, config, domains=DOMAINS):
    """Get average HR across domains for a given model+config."""
    phase_key = f"{gen}_{category}"
    hrs = []
    for d in domains:
        key = f"{provider}_{category}_{d}_{config}"
        if phase_key in all_metrics and key in all_metrics[phase_key]:
            hrs.append(all_metrics[phase_key][key]["hr"])
    return sum(hrs) / len(hrs) if hrs else None


def get_metric_avg(metrics, provider, category, gen, config, field, domains=DOMAINS):
    """Get average of any metric field across domains."""
    phase_key = f"{gen}_{category}"
    vals = []
    for d in domains:
        key = f"{provider}_{category}_{d}_{config}"
        if phase_key in all_metrics and key in all_metrics[phase_key]:
            vals.append(all_metrics[phase_key][key][field])
    return sum(vals) / len(vals) if vals else None


# ── Load all data ────────────────────────────────────────────────────────
all_results = {}
all_metrics = {}

for phase_name, phase_dir in PHASES.items():
    results = load_results(phase_dir)
    all_results[phase_name] = results
    phase_metrics = {}
    for key, queries in results.items():
        m = compute_metrics(queries)
        if m:
            phase_metrics[key] = m
    all_metrics[phase_name] = phase_metrics


# ── Data Quality Report ──────────────────────────────────────────────────
p("DATA QUALITY REPORT")
total_files = 0
total_queries = 0
total_errors = 0
for phase_name, results in all_results.items():
    n_files = len(results)
    n_queries = sum(len(q) for q in results.values())
    n_errors = sum(sum(1 for r in q if r.get("Error")) for q in results.values())
    n_valid = n_queries - n_errors
    total_files += n_files
    total_queries += n_queries
    total_errors += n_errors
    err_str = f" ({n_errors} errors)" if n_errors > 0 else ""
    print(f"  {phase_name:<20s}: {n_files:3d} files, {n_valid:5d} valid queries{err_str}")

print(f"  {'TOTAL':<20s}: {total_files:3d} files, {total_queries - total_errors:5d} valid, {total_errors} errors")

# Flag files with high error rates
high_error_files = []
for phase_name, results in all_results.items():
    for key, queries in results.items():
        errors = sum(1 for q in queries if q.get("Error"))
        if errors > 5:
            high_error_files.append((phase_name, key, len(queries), errors))

if high_error_files:
    print(f"\n  WARNING: {len(high_error_files)} files with >5 errors:")
    for phase, key, total, errors in high_error_files:
        print(f"    {phase}/{key}: {errors}/{total} errors")


# ══════════════════════════════════════════════════════════════════════════
# TABLE: CROSS-GENERATIONAL C5 COMPARISON (key paper table)
# ══════════════════════════════════════════════════════════════════════════
p("TABLE: CROSS-GENERATIONAL C5 HALLUCINATION RATES")
print(f"  {'Provider':<12s} {'Gen1 Std':>10s} {'Gen2 Std':>10s} {'Delta':>8s}  {'Gen1 Think':>12s} {'Gen2 Think':>12s} {'Delta':>8s}")
print(f"  {'-' * 78}")

for prov in PROVIDERS:
    g1s = get_hr(all_metrics, prov, "standard", "gen1", "C5")
    g2s = get_hr(all_metrics, prov, "standard", "gen2", "C5")
    g1t = get_hr(all_metrics, prov, "thinking", "gen1", "C5")
    g2t = get_hr(all_metrics, prov, "thinking", "gen2", "C5")

    g1s_str = f"{g1s:.1f}%" if g1s is not None else "---"
    g2s_str = f"{g2s:.1f}%" if g2s is not None else "---"
    g1t_str = f"{g1t:.1f}%" if g1t is not None else "---"
    g2t_str = f"{g2t:.1f}%" if g2t is not None else "---"

    ds = f"{g2s - g1s:+.1f}pp" if g1s and g2s else "---"
    dt = f"{g2t - g1t:+.1f}pp" if g1t and g2t else "---"

    name1 = MODEL_NAMES.get((prov, "standard", "gen1"), prov)
    print(f"  {prov:<12s} {g1s_str:>10s} {g2s_str:>10s} {ds:>8s}  {g1t_str:>12s} {g2t_str:>12s} {dt:>8s}")

# Cross-provider averages
avg_g1s = [get_hr(all_metrics, p, "standard", "gen1", "C5") for p in PROVIDERS]
avg_g2s = [get_hr(all_metrics, p, "standard", "gen2", "C5") for p in PROVIDERS]
avg_g1t = [get_hr(all_metrics, p, "thinking", "gen1", "C5") for p in PROVIDERS]
avg_g2t = [get_hr(all_metrics, p, "thinking", "gen2", "C5") for p in PROVIDERS]

avg_g1s = sum(v for v in avg_g1s if v) / sum(1 for v in avg_g1s if v) if any(avg_g1s) else None
avg_g2s = sum(v for v in avg_g2s if v) / sum(1 for v in avg_g2s if v) if any(avg_g2s) else None
avg_g1t = sum(v for v in avg_g1t if v) / sum(1 for v in avg_g1t if v) if any(avg_g1t) else None
avg_g2t = sum(v for v in avg_g2t if v) / sum(1 for v in avg_g2t if v) if any(avg_g2t) else None

print(f"  {'-' * 78}")
ds = f"{avg_g2s - avg_g1s:+.1f}pp" if avg_g1s and avg_g2s else "---"
dt = f"{avg_g2t - avg_g1t:+.1f}pp" if avg_g1t and avg_g2t else "---"
print(f"  {'AVERAGE':<12s} {avg_g1s:.1f}%{'':<4s} {avg_g2s:.1f}%{'':<4s} {ds:>8s}  {avg_g1t:.1f}%{'':<6s} {avg_g2t:.1f}%{'':<6s} {dt:>8s}")


# ══════════════════════════════════════════════════════════════════════════
# TABLE: CROSS-GENERATIONAL C0 BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════
p("TABLE: CROSS-GENERATIONAL C0 BASELINE RATES")
print(f"  {'Provider':<12s} {'Gen1 Std':>10s} {'Gen2 Std':>10s} {'Delta':>8s}  {'Gen1 Think':>12s} {'Gen2 Think':>12s} {'Delta':>8s}")
print(f"  {'-' * 78}")

for prov in PROVIDERS:
    g1s = get_hr(all_metrics, prov, "standard", "gen1", "C0")
    g2s = get_hr(all_metrics, prov, "standard", "gen2", "C0")
    g1t = get_hr(all_metrics, prov, "thinking", "gen1", "C0")
    g2t = get_hr(all_metrics, prov, "thinking", "gen2", "C0")

    g1s_str = f"{g1s:.1f}%" if g1s else "---"
    g2s_str = f"{g2s:.1f}%" if g2s else "---"
    g1t_str = f"{g1t:.1f}%" if g1t else "---"
    g2t_str = f"{g2t:.1f}%" if g2t else "---"

    ds = f"{g2s - g1s:+.1f}pp" if g1s and g2s else "---"
    dt = f"{g2t - g1t:+.1f}pp" if g1t and g2t else "---"

    print(f"  {prov:<12s} {g1s_str:>10s} {g2s_str:>10s} {ds:>8s}  {g1t_str:>12s} {g2t_str:>12s} {dt:>8s}")


# ══════════════════════════════════════════════════════════════════════════
# TABLE: GEN2 FULL ABLATION (C0-C5) — standard models
# ══════════════════════════════════════════════════════════════════════════
p("TABLE: GEN2 ABLATION STUDY (STANDARD MODELS)")
print(f"  {'Config':<8s}", end="")
for prov in PROVIDERS:
    name = MODEL_NAMES.get((prov, "standard", "gen2"), prov)
    print(f" {name:>18s}", end="")
print(f" {'Cross-Avg':>12s}")
print(f"  {'-' * 62}")

for cfg in CONFIGS:
    print(f"  {cfg:<8s}", end="")
    hrs = []
    for prov in PROVIDERS:
        hr = get_hr(all_metrics, prov, "standard", "gen2", cfg)
        if hr is not None:
            hrs.append(hr)
            print(f" {hr:>17.1f}%", end="")
        else:
            print(f" {'---':>18s}", end="")
    avg = sum(hrs) / len(hrs) if hrs else None
    print(f" {avg:>11.1f}%" if avg else f" {'---':>12s}")


# ══════════════════════════════════════════════════════════════════════════
# TABLE: GEN1 vs GEN2 ABLATION COMPARISON
# ══════════════════════════════════════════════════════════════════════════
p("TABLE: ABLATION COMPARISON GEN1 vs GEN2 (CROSS-PROVIDER AVERAGES)")
print(f"  {'Config':<8s} {'Gen1 Avg':>10s} {'Gen2 Avg':>10s} {'Delta':>8s} {'Improvement':>12s}")
print(f"  {'-' * 52}")

for cfg in CONFIGS:
    g1_hrs = [get_hr(all_metrics, p, "standard", "gen1", cfg) for p in PROVIDERS]
    g2_hrs = [get_hr(all_metrics, p, "standard", "gen2", cfg) for p in PROVIDERS]

    g1_avg = sum(v for v in g1_hrs if v) / sum(1 for v in g1_hrs if v) if any(g1_hrs) else None
    g2_avg = sum(v for v in g2_hrs if v) / sum(1 for v in g2_hrs if v) if any(g2_hrs) else None

    if g1_avg is not None and g2_avg is not None:
        delta = g2_avg - g1_avg
        improv = ((g1_avg - g2_avg) / g1_avg * 100) if g1_avg > 0 else 0
        print(f"  {cfg:<8s} {g1_avg:>9.1f}% {g2_avg:>9.1f}% {delta:>+7.1f}pp {improv:>11.1f}%")
    else:
        print(f"  {cfg:<8s} {'---':>10s} {'---':>10s}")


# ══════════════════════════════════════════════════════════════════════════
# TABLE: GEN2 STANDARD vs THINKING
# ══════════════════════════════════════════════════════════════════════════
p("TABLE: GEN2 STANDARD vs THINKING COMPARISON")
print(f"  {'Provider':<12s} {'Std C0':>8s} {'Think C0':>10s} {'Std C5':>8s} {'Think C5':>10s} {'Std Red.':>10s} {'Think Red.':>12s}")
print(f"  {'-' * 74}")

for prov in PROVIDERS:
    sc0 = get_hr(all_metrics, prov, "standard", "gen2", "C0")
    tc0 = get_hr(all_metrics, prov, "thinking", "gen2", "C0")
    sc5 = get_hr(all_metrics, prov, "standard", "gen2", "C5")
    tc5 = get_hr(all_metrics, prov, "thinking", "gen2", "C5")

    sr = ((sc0 - sc5) / sc0 * 100) if sc0 and sc5 else None
    tr = ((tc0 - tc5) / tc0 * 100) if tc0 and tc5 else None

    print(f"  {prov:<12s}"
          f" {sc0:.1f}%" if sc0 else "   ---", end="")
    print(f"    {tc0:.1f}%" if tc0 else "       ---", end="")
    print(f"  {sc5:.1f}%" if sc5 else "     ---", end="")
    print(f"      {tc5:.1f}%" if tc5 else "         ---", end="")
    print(f"    {sr:.1f}%" if sr else "       ---", end="")
    print(f"        {tr:.1f}%" if tr else "           ---")


# ══════════════════════════════════════════════════════════════════════════
# TABLE: GEN2 RESPONSE CHARACTERISTICS (C5)
# ══════════════════════════════════════════════════════════════════════════
p("TABLE: GEN2 RESPONSE CHARACTERISTICS (C5)")
print(f"  {'Model':<30s} {'Tools/Q':>8s} {'OutTok':>8s} {'Latency':>10s} {'ThinkTok':>10s} {'JSON%':>8s}")
print(f"  {'-' * 78}")

for gen in ["gen2"]:
    for cat in ["standard", "thinking"]:
        phase_key = f"{gen}_{cat}"
        if phase_key not in all_metrics:
            continue
        for prov in PROVIDERS:
            c5_metrics = [m for k, m in all_metrics[phase_key].items()
                         if k.startswith(f"{prov}_{cat}") and "_C5" in k]
            if not c5_metrics:
                continue
            avg_tools = sum(m["avg_tools_per_query"] for m in c5_metrics) / len(c5_metrics)
            avg_out = sum(m["avg_output_tokens"] for m in c5_metrics) / len(c5_metrics)
            avg_lat = sum(m["avg_latency_ms"] for m in c5_metrics) / len(c5_metrics)
            avg_think = sum(m["avg_thinking_tokens"] for m in c5_metrics) / len(c5_metrics)
            avg_json = sum(m["json_valid_rate"] for m in c5_metrics) / len(c5_metrics)

            name = MODEL_NAMES.get((prov, cat, gen), f"{prov} ({cat})")
            print(f"  {name:<30s}"
                  f"  {avg_tools:6.1f}"
                  f"  {avg_out:6.0f}"
                  f"  {avg_lat/1000:8.1f}s"
                  f"  {avg_think:8.0f}"
                  f"  {avg_json:6.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# TABLE: GEN2 DOMAIN-SPECIFIC C5 RATES
# ══════════════════════════════════════════════════════════════════════════
p("TABLE: GEN2 DOMAIN-SPECIFIC C5 HALLUCINATION RATES (STANDARD)")
print(f"  {'Provider':<12s} {'D1':>8s} {'D2':>8s} {'D3':>8s} {'D4':>8s} {'Avg':>8s}")
print(f"  {'-' * 50}")

for prov in PROVIDERS:
    name = MODEL_NAMES.get((prov, "standard", "gen2"), prov)
    row = f"  {name:<30s}"
    hrs = []
    for d in DOMAINS:
        key = f"{prov}_standard_{d}_C5"
        if "gen2_standard" in all_metrics and key in all_metrics["gen2_standard"]:
            hr = all_metrics["gen2_standard"][key]["hr"]
            hrs.append(hr)
            row += f"  {hr:6.1f}%"
        else:
            row += "     ---"
    avg = sum(hrs) / len(hrs) if hrs else None
    row += f"  {avg:6.1f}%" if avg else "     ---"
    print(row)


# ══════════════════════════════════════════════════════════════════════════
# TABLE: C3 ANOMALY COMPARISON ACROSS GENERATIONS
# ══════════════════════════════════════════════════════════════════════════
p("TABLE: C3 ANOMALY (JSON-ONLY) CROSS-GENERATIONAL")
print(f"  {'Provider':<12s} {'Gen1 C0':>9s} {'Gen1 C3':>9s} {'Increase':>10s}  {'Gen2 C0':>9s} {'Gen2 C3':>9s} {'Increase':>10s}")
print(f"  {'-' * 66}")

for prov in PROVIDERS:
    g1c0 = get_hr(all_metrics, prov, "standard", "gen1", "C0")
    g1c3 = get_hr(all_metrics, prov, "standard", "gen1", "C3")
    g2c0 = get_hr(all_metrics, prov, "standard", "gen2", "C0")
    g2c3 = get_hr(all_metrics, prov, "standard", "gen2", "C3")

    g1inc = f"+{g1c3 - g1c0:.1f}pp" if g1c0 and g1c3 else "---"
    g2inc = f"+{g2c3 - g2c0:.1f}pp" if g2c0 and g2c3 else "---"

    print(f"  {prov:<12s}"
          f" {g1c0:.1f}%" if g1c0 else "      ---", end="")
    print(f" {g1c3:.1f}%" if g1c3 else "      ---", end="")
    print(f" {g1inc:>10s}", end="")
    print(f"  {g2c0:.1f}%" if g2c0 else "      ---", end="")
    print(f" {g2c3:.1f}%" if g2c3 else "      ---", end="")
    print(f" {g2inc:>10s}")


# ══════════════════════════════════════════════════════════════════════════
# TABLE: MECHANISM CONTRIBUTION ANALYSIS (GEN2)
# ══════════════════════════════════════════════════════════════════════════
p("TABLE: GEN2 MECHANISM CONTRIBUTION ANALYSIS")
print("  Cross-provider averages for Gen2 standard models:")
print(f"  {'Mechanism':<40s} {'Avg HR':>8s} {'Reduction':>10s} {'Capture%':>10s}")
print(f"  {'-' * 72}")

c0_avg = sum(get_hr(all_metrics, p, "standard", "gen2", "C0") or 0 for p in PROVIDERS) / 3
c5_avg = sum(get_hr(all_metrics, p, "standard", "gen2", "C5") or 0 for p in PROVIDERS) / 3
full_reduction = c0_avg - c5_avg

for cfg, label in [("C0", "No mechanisms (baseline)"),
                   ("C1", "M1 only (context)"),
                   ("C2", "M2 only (vocabulary)"),
                   ("C3", "M3 only (JSON) — ANOMALY"),
                   ("C4", "M1+M2 (context + vocabulary)"),
                   ("C5", "Full architecture (M1+M2+M3)")]:
    avg = sum(get_hr(all_metrics, p, "standard", "gen2", cfg) or 0 for p in PROVIDERS) / 3
    red = ((c0_avg - avg) / c0_avg * 100) if c0_avg > 0 else 0
    capture = ((c0_avg - avg) / full_reduction * 100) if full_reduction > 0 else 0
    print(f"  {label:<40s} {avg:>7.1f}% {red:>9.1f}% {capture:>9.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# LATEX TABLE SNIPPETS (copy-paste ready)
# ══════════════════════════════════════════════════════════════════════════
p("LATEX: CROSS-GENERATIONAL C5 TABLE")
print(r"""
\begin{table}[H]
\centering
\caption{Cross-generational C5 hallucination rates (\%). Lower is better.}
\label{tab:cross_gen_c5}
\begin{tabular}{llcccc}
\toprule
\textbf{Provider} & \textbf{Mode} & \textbf{Gen1} & \textbf{Gen2} & \textbf{$\Delta$} & \textbf{Direction} \\
\midrule""")

for prov in PROVIDERS:
    for cat in ["standard", "thinking"]:
        g1 = get_hr(all_metrics, prov, cat, "gen1", "C5")
        g2 = get_hr(all_metrics, prov, cat, "gen2", "C5")
        name1 = MODEL_NAMES.get((prov, cat, "gen1"), "?")
        name2 = MODEL_NAMES.get((prov, cat, "gen2"), "?")
        if g1 is not None and g2 is not None:
            delta = g2 - g1
            direction = r"$\downarrow$ Improved" if delta < -0.5 else (r"$\uparrow$ Regressed" if delta > 0.5 else r"$\approx$ Stable")
            better = r"\textbf" if abs(delta) > 0.5 and delta < 0 else ""
            g2_fmt = f"\\textbf{{{g2:.1f}}}" if delta < -0.5 else f"{g2:.1f}"
            print(f"{name1} & {cat} & {g1:.1f} & {g2_fmt} & {delta:+.1f} & {direction} \\\\")

print(r"""\midrule""")
# Averages
for cat in ["standard", "thinking"]:
    g1_vals = [get_hr(all_metrics, p, cat, "gen1", "C5") for p in PROVIDERS]
    g2_vals = [get_hr(all_metrics, p, cat, "gen2", "C5") for p in PROVIDERS]
    g1_avg = sum(v for v in g1_vals if v) / sum(1 for v in g1_vals if v) if any(g1_vals) else 0
    g2_avg = sum(v for v in g2_vals if v) / sum(1 for v in g2_vals if v) if any(g2_vals) else 0
    delta = g2_avg - g1_avg
    direction = r"$\downarrow$ Improved" if delta < -0.5 else (r"$\uparrow$ Regressed" if delta > 0.5 else r"$\approx$ Stable")
    print(f"\\textit{{Average}} & {cat} & {g1_avg:.1f} & {g2_avg:.1f} & {delta:+.1f} & {direction} \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}""")


# ══════════════════════════════════════════════════════════════════════════
# GRAND SUMMARY
# ══════════════════════════════════════════════════════════════════════════
p("GRAND SUMMARY FOR PAPER")
print(f"  Total API calls (valid): {total_queries - total_errors}")
print(f"  Total API calls (incl. errors): {total_queries}")
print(f"  Total result files: {total_files}")

print(f"\n  Gen1 Standard C5 (cross-provider avg): {avg_g1s:.1f}%")
print(f"  Gen2 Standard C5 (cross-provider avg): {avg_g2s:.1f}%")
print(f"  Generational improvement (standard): {avg_g1s - avg_g2s:+.1f}pp")

print(f"\n  Gen1 Thinking C5 (cross-provider avg): {avg_g1t:.1f}%")
print(f"  Gen2 Thinking C5 (cross-provider avg): {avg_g2t:.1f}%")
print(f"  Generational improvement (thinking): {avg_g1t - avg_g2t:+.1f}pp")

# Best model overall
all_c5 = []
for gen in ["gen1", "gen2"]:
    for cat in ["standard", "thinking"]:
        for prov in PROVIDERS:
            hr = get_hr(all_metrics, prov, cat, gen, "C5")
            if hr is not None:
                name = MODEL_NAMES.get((prov, cat, gen), f"{prov} {cat} {gen}")
                all_c5.append((hr, name))

all_c5.sort()
print(f"\n  Best C5 model: {all_c5[0][1]} ({all_c5[0][0]:.1f}%)")
print(f"  Worst C5 model: {all_c5[-1][1]} ({all_c5[-1][0]:.1f}%)")
print(f"  Range factor: {all_c5[-1][0] / all_c5[0][0]:.1f}x")

# C3 anomaly universality
print(f"\n  C3 Anomaly (Gen2):")
for prov in PROVIDERS:
    c0 = get_hr(all_metrics, prov, "standard", "gen2", "C0")
    c3 = get_hr(all_metrics, prov, "standard", "gen2", "C3")
    if c0 and c3:
        print(f"    {prov}: C0={c0:.1f}% -> C3={c3:.1f}% ({c3-c0:+.1f}pp) {'ANOMALY' if c3 > c0 else 'OK'}")

# M1+M2 capture percentage
c4_avg_g2 = sum(get_hr(all_metrics, p, "standard", "gen2", "C4") or 0 for p in PROVIDERS) / 3
c5_avg_g2 = sum(get_hr(all_metrics, p, "standard", "gen2", "C5") or 0 for p in PROVIDERS) / 3
c0_avg_g2 = sum(get_hr(all_metrics, p, "standard", "gen2", "C0") or 0 for p in PROVIDERS) / 3
capture_g2 = ((c0_avg_g2 - c4_avg_g2) / (c0_avg_g2 - c5_avg_g2) * 100) if (c0_avg_g2 - c5_avg_g2) > 0 else 0
print(f"\n  Gen2 M1+M2 (C4) capture: {capture_g2:.1f}% of full C5 benefit")
print(f"  Gen2 C4 avg: {c4_avg_g2:.1f}%, C5 avg: {c5_avg_g2:.1f}%")
