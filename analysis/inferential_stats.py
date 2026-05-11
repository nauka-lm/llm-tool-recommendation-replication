#!/usr/bin/env python3
"""
Inferential statistical tests for the cross-generational tier-aligned evaluation.

Tests pre-registered in the analysis plan:
  1. Cross-generational HR comparison (paired by domain/prompt/rep, under C5):
       - Sonnet 4.5 vs Sonnet 4.6 (mid-tier, both gens)            -> Wilcoxon
       - Flash-Lite 2.5 vs Flash-Lite 3.1 (entry-tier, both gens)  -> Wilcoxon
       - GPT-4.1 vs GPT-5.2 (flagship, both gens)                  -> Wilcoxon
  2. Standard vs thinking within each Gen2 model (paired, under C5)  -> Wilcoxon
  3. Ablation rank order (Friedman across C2/C4/C5 within model)    -> Friedman + planned contrasts
  4. Effect sizes: matched-pairs rank-biserial correlation, Cliff's delta.
  5. Multiple-comparison correction: Bonferroni for the planned-contrast family (k=3).

Outputs:
    inferential_stats_results.json  -- structured machine-readable
    inferential_stats_report.md     -- human-readable report

Requirements:
    Python 3.10+, scipy, numpy.  Pin versions in requirements.txt.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from scipy.stats import wilcoxon, mannwhitneyu, friedmanchisquare
except ImportError as exc:
    sys.stderr.write(
        "Missing dependency: install scipy and numpy.\n"
        "    pip install -r requirements.txt\n"
    )
    raise SystemExit(1) from exc


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results"

DOMAINS = ["D1", "D2", "D3", "D4"]
PROMPTS = list(range(12))
REPS = [1, 2, 3]

# Cross-generational pairs (mid-tier and entry-tier are the new tier-aligned cells)
PAIRS_CROSS_GEN = [
    {
        "label": "Sonnet 4.5 (Gen1) vs Sonnet 4.6 (Gen2), C5 standard",
        "a": ("phase1", "claude-sonnet-4-5-20250929", "standard", "C5"),
        "b": ("phase3a", "claude-sonnet-4-6", "standard", "C5"),
    },
    {
        "label": "Flash-Lite 2.5 (Gen1) vs Flash-Lite 3.1 (Gen2), C5 standard",
        "a": ("phase1", "gemini-2.5-flash-lite", "standard", "C5"),
        "b": ("phase3a", "gemini-3.1-flash-lite-preview", "standard", "C5"),
    },
    {
        "label": "GPT-4.1 (Gen1) vs GPT-5.2 (Gen2), C5 standard",
        "a": ("phase1", "gpt-4.1", "standard", "C5"),
        "b": ("phase3a", "gpt-5.2", "standard", "C5"),
    },
]

# Standard vs thinking pairs within each Gen2 provider (under C5)
PAIRS_STD_VS_THINK_GEN2 = [
    {
        "label": "GPT-5.2 standard vs thinking, C5",
        "a": ("phase3a", "gpt-5.2", "standard", "C5"),
        "b": ("phase3b", "gpt-5.2", "thinking", "C5"),
    },
    {
        "label": "Sonnet 4.6 standard vs thinking, C5",
        "a": ("phase3a", "claude-sonnet-4-6", "standard", "C5"),
        "b": ("phase3b", "claude-sonnet-4-6", "thinking", "C5"),
    },
    {
        "label": "Flash-Lite 3.1 standard vs thinking, C5",
        "a": ("phase3a", "gemini-3.1-flash-lite-preview", "standard", "C5"),
        "b": ("phase3b", "gemini-3.1-flash-lite-preview", "thinking", "C5"),
    },
]

# Friedman ablation rank-ordering: per-model, compare HR distributions across C2, C4, C5.
# Each "block" is one query (12 prompts x 3 reps x 4 domains = 144 blocks per model).
FRIEDMAN_MODELS_GEN2 = [
    ("phase3a", "gpt-5.2", "standard", "GPT-5.2"),
    ("phase3a", "claude-sonnet-4-6", "standard", "Claude Sonnet 4.6"),
    ("phase3a", "gemini-3.1-flash-lite-preview", "standard", "Gemini 3.1 Flash-Lite"),
]
FRIEDMAN_CONFIGS = ["C2", "C4", "C5"]

# Multiple-comparison correction
BONFERRONI_K = 3
ALPHA = 0.05
ALPHA_ADJ = ALPHA / BONFERRONI_K  # 0.01667 for k=3


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def load_block(phase: str, model_id: str, category: str, config: str) -> list[dict[str, Any]]:
    """Load all queries for (phase, model, category, config) across the 4 domains.

    Returns a list of QueryResult dicts in deterministic order:
    sorted by (DomainId, PromptIndex, Repetition).
    """
    rows: list[dict[str, Any]] = []
    for d in DOMAINS:
        provider = _provider_from_model(model_id)
        path = RESULTS_DIR / phase / f"results_{provider}_{category}_{d}_{config}.json"
        if not path.exists():
            sys.stderr.write(f"  WARN: missing {path}\n")
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for r in data:
            if r.get("Error"):
                continue
            if r.get("ModelId") != model_id:
                # Defensive: skip rows that don't match the requested model
                continue
            rows.append(r)

    rows.sort(key=lambda x: (x["DomainId"], x["PromptIndex"], x["Repetition"]))
    return rows


def _provider_from_model(model_id: str) -> str:
    if model_id.startswith("gpt") or model_id.startswith("o4"):
        return "openai"
    if model_id.startswith("claude"):
        return "anthropic"
    if model_id.startswith("gemini"):
        return "google"
    raise ValueError(f"Cannot infer provider from model_id={model_id}")


def hr_per_query(rows: list[dict[str, Any]]) -> list[float]:
    """Per-query hallucination rate = |hallucinated| / |mentioned| (0 if no tools mentioned)."""
    out: list[float] = []
    for r in rows:
        mentioned = len(r.get("MentionedToolNames") or [])
        halluc = len(r.get("HallucinatedToolNames") or [])
        out.append(halluc / mentioned if mentioned > 0 else 0.0)
    return out


# ----------------------------------------------------------------------------
# Effect sizes
# ----------------------------------------------------------------------------

def matched_pairs_rank_biserial(diffs: list[float]) -> float:
    """Matched-pairs rank-biserial correlation (Wilcoxon signed-rank effect size).

    r = (W_pos - W_neg) / (W_pos + W_neg)
    where W_pos and W_neg are the sums of positive and negative signed ranks.
    Range: [-1, 1].  Sign indicates direction of effect.
    """
    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        return 0.0
    abs_diffs = [abs(d) for d in nonzero]
    # Rank with ties averaged
    ranked_pairs = sorted(enumerate(abs_diffs), key=lambda t: t[1])
    ranks = [0.0] * len(abs_diffs)
    i = 0
    while i < len(ranked_pairs):
        j = i
        while j + 1 < len(ranked_pairs) and ranked_pairs[j + 1][1] == ranked_pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            idx = ranked_pairs[k][0]
            ranks[idx] = avg_rank
        i = j + 1
    w_pos = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    if w_pos + w_neg == 0:
        return 0.0
    return (w_pos - w_neg) / (w_pos + w_neg)


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta for two independent samples.

    delta = (#(a > b) - #(a < b)) / (n_a * n_b)
    Range: [-1, 1].  Magnitude conventions: |delta| < 0.147 negligible,
    < 0.33 small, < 0.474 medium, otherwise large.
    """
    if not a or not b:
        return 0.0
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    gt = sum(int((a_arr > x).sum()) for x in b_arr)
    lt = sum(int((a_arr < x).sum()) for x in b_arr)
    return (gt - lt) / (len(a) * len(b))


def cliffs_delta_label(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def _summary(arr: list[float]) -> dict[str, float]:
    if not arr:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    a = np.asarray(arr)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


def run_wilcoxon_paired(spec: dict[str, Any]) -> dict[str, Any]:
    a_rows = load_block(*spec["a"])
    b_rows = load_block(*spec["b"])

    # Pair by (DomainId, PromptIndex, Repetition)
    a_index = {(r["DomainId"], r["PromptIndex"], r["Repetition"]): r for r in a_rows}
    b_index = {(r["DomainId"], r["PromptIndex"], r["Repetition"]): r for r in b_rows}
    keys = sorted(set(a_index) & set(b_index))
    a_hr = [hr_per_query([a_index[k]])[0] for k in keys]
    b_hr = [hr_per_query([b_index[k]])[0] for k in keys]

    diffs = [b - a for a, b in zip(a_hr, b_hr)]
    n_pairs = len(diffs)
    n_nonzero = sum(1 for d in diffs if d != 0.0)

    if n_nonzero >= 1:
        try:
            stat, pvalue = wilcoxon(a_hr, b_hr, zero_method="wilcox", alternative="two-sided")
            stat_value = float(stat)
            p = float(pvalue)
        except ValueError:
            stat_value = float("nan")
            p = float("nan")
    else:
        stat_value = float("nan")
        p = float("nan")

    rb = matched_pairs_rank_biserial(diffs)

    return {
        "label": spec["label"],
        "a": _spec_to_label(spec["a"]),
        "b": _spec_to_label(spec["b"]),
        "n_pairs": n_pairs,
        "n_nonzero": n_nonzero,
        "a_summary": _summary(a_hr),
        "b_summary": _summary(b_hr),
        "diff_summary": _summary(diffs),
        "wilcoxon_W": stat_value,
        "p_value": p,
        "rank_biserial": rb,
        "significant_at_alpha_0_05": (p < ALPHA) if not math.isnan(p) else None,
    }


def run_mannwhitney(spec: dict[str, Any]) -> dict[str, Any]:
    a_rows = load_block(*spec["a"])
    b_rows = load_block(*spec["b"])
    a_hr = hr_per_query(a_rows)
    b_hr = hr_per_query(b_rows)

    if a_hr and b_hr:
        try:
            stat, p = mannwhitneyu(a_hr, b_hr, alternative="two-sided")
            stat_value, p_value = float(stat), float(p)
        except ValueError:
            stat_value, p_value = float("nan"), float("nan")
    else:
        stat_value, p_value = float("nan"), float("nan")

    delta = cliffs_delta(a_hr, b_hr)

    return {
        "label": spec["label"],
        "a": _spec_to_label(spec["a"]),
        "b": _spec_to_label(spec["b"]),
        "n_a": len(a_hr),
        "n_b": len(b_hr),
        "a_summary": _summary(a_hr),
        "b_summary": _summary(b_hr),
        "mannwhitney_U": stat_value,
        "p_value": p_value,
        "cliffs_delta": delta,
        "cliffs_delta_label": cliffs_delta_label(delta),
        "significant_at_alpha_0_05": (p_value < ALPHA) if not math.isnan(p_value) else None,
    }


def _spec_to_label(spec: tuple[str, str, str, str]) -> str:
    phase, model, cat, cfg = spec
    return f"{phase}/{model}/{cat}/{cfg}"


def run_friedman(model_spec: tuple[str, str, str, str]) -> dict[str, Any]:
    phase, model_id, category, model_label = model_spec
    columns: list[list[float]] = []
    keys: list[tuple[Any, ...]] | None = None

    for cfg in FRIEDMAN_CONFIGS:
        rows = load_block(phase, model_id, category, cfg)
        idx = {(r["DomainId"], r["PromptIndex"], r["Repetition"]): hr_per_query([r])[0] for r in rows}
        if keys is None:
            keys = sorted(idx.keys())
        else:
            keys = sorted(set(keys) & set(idx))
        columns.append([idx[k] for k in keys] if keys else [])

    # Re-extract aligned columns
    aligned_columns = []
    for cfg, rows in zip(FRIEDMAN_CONFIGS, columns):
        aligned_columns.append(rows)

    # All columns must have the same length
    if not aligned_columns or len(set(len(c) for c in aligned_columns)) != 1 or len(aligned_columns[0]) < 2:
        return {
            "model": model_label,
            "configs": FRIEDMAN_CONFIGS,
            "n_blocks": 0,
            "friedman_chi2": float("nan"),
            "p_value": float("nan"),
            "note": "insufficient paired data",
        }

    chi2, p = friedmanchisquare(*aligned_columns)

    # Planned pairwise contrasts (k=3): C2 vs C4, C2 vs C5, C4 vs C5
    pairs = [(0, 1), (0, 2), (1, 2)]
    contrasts = []
    for i, j in pairs:
        a, b = aligned_columns[i], aligned_columns[j]
        try:
            stat, pp = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            stat_val, p_val = float(stat), float(pp)
        except ValueError:
            stat_val, p_val = float("nan"), float("nan")
        diffs = [bb - aa for aa, bb in zip(a, b)]
        contrasts.append({
            "pair": f"{FRIEDMAN_CONFIGS[i]} vs {FRIEDMAN_CONFIGS[j]}",
            "wilcoxon_W": stat_val,
            "p_value": p_val,
            "p_value_bonferroni_adjusted": min(1.0, p_val * BONFERRONI_K) if not math.isnan(p_val) else float("nan"),
            "rank_biserial": matched_pairs_rank_biserial(diffs),
            "significant_after_bonferroni": (p_val < ALPHA_ADJ) if not math.isnan(p_val) else None,
        })

    return {
        "model": model_label,
        "configs": FRIEDMAN_CONFIGS,
        "n_blocks": len(aligned_columns[0]),
        "friedman_chi2": float(chi2),
        "p_value": float(p),
        "significant_at_alpha_0_05": float(p) < ALPHA,
        "planned_contrasts": contrasts,
        "alpha_adj": ALPHA_ADJ,
        "bonferroni_k": BONFERRONI_K,
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def render_markdown(results: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("# Inferential statistical tests — cross-generational tier-aligned evaluation")
    out.append("")
    out.append(f"Generated by `analysis/inferential_stats.py`. Pre-registered tests with Bonferroni adjustment for k={BONFERRONI_K} planned contrasts (alpha_adj = {ALPHA_ADJ:.4f}).")
    out.append("")

    out.append("## 1. Cross-generational paired comparisons (Wilcoxon signed-rank, C5)")
    out.append("")
    out.append("Paired by (Domain, Prompt, Repetition). Effect size: matched-pairs rank-biserial correlation.")
    out.append("")
    out.append("| Comparison | n | mean HR (a) | mean HR (b) | mean diff (b-a) | W | p | r_rb | sig at 0.05 |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for r in results["cross_gen"]:
        sig = "yes" if r["significant_at_alpha_0_05"] else "no"
        out.append(
            f"| {r['label']} | {r['n_pairs']} | {r['a_summary']['mean']:.3f} | {r['b_summary']['mean']:.3f} | "
            f"{r['diff_summary']['mean']:+.3f} | {r['wilcoxon_W']:.0f} | {r['p_value']:.4g} | {r['rank_biserial']:+.3f} | {sig} |"
        )
    out.append("")

    out.append("## 2. Standard vs thinking within each Gen2 model (Wilcoxon signed-rank, C5)")
    out.append("")
    out.append("| Comparison | n | mean HR (std) | mean HR (think) | mean diff (think-std) | W | p | r_rb | sig at 0.05 |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for r in results["std_vs_think"]:
        sig = "yes" if r["significant_at_alpha_0_05"] else "no"
        out.append(
            f"| {r['label']} | {r['n_pairs']} | {r['a_summary']['mean']:.3f} | {r['b_summary']['mean']:.3f} | "
            f"{r['diff_summary']['mean']:+.3f} | {r['wilcoxon_W']:.0f} | {r['p_value']:.4g} | {r['rank_biserial']:+.3f} | {sig} |"
        )
    out.append("")

    out.append("## 3. Ablation rank ordering — Friedman test + planned pairwise contrasts (Wilcoxon)")
    out.append("")
    out.append(f"Configs compared: {', '.join(FRIEDMAN_CONFIGS)}. Bonferroni-adjusted alpha for k={BONFERRONI_K} planned contrasts: {ALPHA_ADJ:.4f}.")
    out.append("")
    for r in results["friedman"]:
        out.append(f"### {r['model']}")
        out.append("")
        out.append(f"- Friedman chi^2 = {r['friedman_chi2']:.3f}, p = {r['p_value']:.4g}, n = {r['n_blocks']} blocks")
        out.append("")
        out.append("| Contrast | W | p (raw) | p (Bonferroni) | r_rb | sig after Bonferroni |")
        out.append("| --- | ---: | ---: | ---: | ---: | :---: |")
        for c in r["planned_contrasts"]:
            sig = "yes" if c["significant_after_bonferroni"] else "no"
            out.append(
                f"| {c['pair']} | {c['wilcoxon_W']:.0f} | {c['p_value']:.4g} | "
                f"{c['p_value_bonferroni_adjusted']:.4g} | {c['rank_biserial']:+.3f} | {sig} |"
            )
        out.append("")

    out.append("---")
    out.append("")
    out.append("Notes:")
    out.append("")
    out.append("- HR is computed per query as |HallucinatedToolNames| / |MentionedToolNames|, with HR = 0 for queries that mention zero tools.")
    out.append("- Wilcoxon test uses `zero_method='wilcox'` (zero-difference pairs dropped).")
    out.append("- Effect sizes: matched-pairs rank-biserial correlation (Wilcoxon) and Cliff's delta (Mann-Whitney).")
    out.append("- Magnitude conventions for Cliff's delta and rank-biserial: |r| < 0.147 negligible, < 0.33 small, < 0.474 medium, otherwise large.")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    sys.stdout.write("Running inferential statistical tests...\n\n")

    cross_gen = [run_wilcoxon_paired(spec) for spec in PAIRS_CROSS_GEN]
    sys.stdout.write(f"  Cross-generational paired comparisons: {len(cross_gen)} done\n")

    std_vs_think = [run_wilcoxon_paired(spec) for spec in PAIRS_STD_VS_THINK_GEN2]
    sys.stdout.write(f"  Standard vs thinking comparisons: {len(std_vs_think)} done\n")

    friedman = [run_friedman(m) for m in FRIEDMAN_MODELS_GEN2]
    sys.stdout.write(f"  Friedman + planned contrasts: {len(friedman)} models done\n")

    results = {
        "alpha": ALPHA,
        "alpha_adj": ALPHA_ADJ,
        "bonferroni_k": BONFERRONI_K,
        "cross_gen": cross_gen,
        "std_vs_think": std_vs_think,
        "friedman": friedman,
    }

    out_dir = Path(__file__).resolve().parent
    json_path = out_dir / "inferential_stats_results.json"
    md_path = out_dir / "inferential_stats_report.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(results))

    sys.stdout.write(f"\nWrote {json_path}\n")
    sys.stdout.write(f"Wrote {md_path}\n")


if __name__ == "__main__":
    main()
