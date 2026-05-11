#!/usr/bin/env python3
"""
Statistical Analysis for Journal Paper
Cross-Provider Evaluation of Anti-Hallucination Architectures

Computes:
  A1.1 - 95% Confidence Intervals for all key HR values
  A1.2 - C0 vs C5 significance tests (architecture effectiveness)
  A1.3 - Standard vs Thinking significance tests
  A1.4 - C3 anomaly significance tests
  A1.5 - Cross-generational stability tests
  A1.6 - Ablation ordering significance (Friedman test)
  B1   - Verbosity-hallucination correlation
  C1   - Per-tool-mention hallucination rates
"""

import json
import glob
import math
import os
import random
import sys
from collections import defaultdict
from itertools import combinations

# ===========================================================================
# Utility functions (no scipy dependency - pure Python implementations)
# ===========================================================================

def mean(data):
    return sum(data) / len(data) if data else 0

def std(data, ddof=1):
    if len(data) <= ddof:
        return 0
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / (len(data) - ddof))

def se(data):
    return std(data) / math.sqrt(len(data)) if data else 0

def ci95(data):
    """95% confidence interval using t-distribution approximation."""
    n = len(data)
    if n < 2:
        return (mean(data), mean(data))
    m = mean(data)
    # t critical value for 95% CI (two-tailed)
    # For n >= 30, use 1.96; for smaller n, use approximate t values
    t_crit = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 15: 2.145, 20: 2.093, 25: 2.064, 30: 2.045,
        40: 2.023, 50: 2.010, 60: 2.001, 80: 1.990, 100: 1.984,
        120: 1.980, 144: 1.977, 200: 1.972, 288: 1.968, 500: 1.965
    }
    # Find closest t value
    df = n - 1
    t = 1.96  # default for large n
    for k in sorted(t_crit.keys()):
        if k >= df:
            t = t_crit[k]
            break
    margin = t * se(data)
    return (m - margin, m + margin)

def bootstrap_ci95(data, n_resamples=10000, seed=42):
    """95% bootstrap confidence interval (percentile method, 10,000 resamples).
    Uses fixed seed for reproducibility."""
    n = len(data)
    if n < 2:
        return (mean(data), mean(data))
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(0.025 * n_resamples)
    hi_idx = int(0.975 * n_resamples) - 1
    return (means[lo_idx], means[hi_idx])

def cohens_d(group1, group2):
    """Cohen's d effect size (pooled standard deviation)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)
    pooled_s = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_s == 0:
        return 0
    return (m1 - m2) / pooled_s

def cliffs_delta(x, y):
    """Cliff's delta effect size (nonparametric).
    Ranges from -1 to +1. |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, >= 0.474 large."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0
    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    return (more - less) / (nx * ny)

def cliffs_delta_label(d):
    """Interpret Cliff's delta magnitude."""
    d = abs(d)
    if d < 0.147: return "negligible"
    if d < 0.33: return "small"
    if d < 0.474: return "medium"
    return "large"

def rank_biserial_r(x, y):
    """Matched-pairs rank-biserial correlation for Wilcoxon signed-rank test.
    r = (W+ - W-) / (W+ + W-), ranges from -1 to +1.
    Positive r means x > y on average; negative means x < y."""
    diffs = [xi - yi for xi, yi in zip(x, y)]
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return 0.0
    # Rank absolute differences (same logic as wilcoxon_signed_rank)
    abs_diffs = [(abs(d), i) for i, d in enumerate(diffs)]
    abs_diffs.sort(key=lambda t: t[0])
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    total = w_plus + w_minus  # = n*(n+1)/2
    if total == 0:
        return 0.0
    return (w_plus - w_minus) / total

def mann_whitney_u(x, y):
    """Mann-Whitney U test (two-sided). Returns U statistic, z-score, and approximate p-value."""
    nx, ny = len(x), len(y)
    # Combine and rank
    combined = [(v, 'x', i) for i, v in enumerate(x)] + [(v, 'y', i) for i, v in enumerate(y)]
    combined.sort(key=lambda t: t[0])

    # Assign ranks with tie handling
    ranks = [0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average rank
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum ranks for x
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 'x')
    u1 = r1 - nx * (nx + 1) / 2
    u2 = nx * ny - u1
    u = min(u1, u2)

    # Normal approximation for p-value
    mu = nx * ny / 2
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12)
    if sigma == 0:
        return u, 0, 1.0
    z = (u - mu) / sigma
    # Two-sided p-value approximation using error function
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u, z, p

def wilcoxon_signed_rank(x, y, alternative='two-sided'):
    """Wilcoxon signed-rank test. Returns T statistic, z-score, and approximate p-value."""
    diffs = [xi - yi for xi, yi in zip(x, y)]
    # Remove zeros
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return 0, 0, 1.0

    # Rank absolute differences
    abs_diffs = [(abs(d), i) for i, d in enumerate(diffs)]
    abs_diffs.sort(key=lambda t: t[0])

    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j

    # Sum positive and negative ranks
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)

    if alternative == 'two-sided':
        t_stat = min(w_plus, w_minus)
    elif alternative == 'greater':
        t_stat = w_minus  # small w_minus means x > y
    elif alternative == 'less':
        t_stat = w_plus

    # Normal approximation
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return t_stat, 0, 1.0
    z = (t_stat - mu) / sigma

    if alternative == 'two-sided':
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    else:
        # One-sided: left-tail CDF Phi(z). When data is in expected direction,
        # z is very negative -> p near 0. When wrong direction, z > 0 -> p near 1.
        p = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    return t_stat, z, p

def friedman_test(groups):
    """Friedman test for k related samples. groups is a list of lists (each list = one treatment)."""
    k = len(groups)
    n = len(groups[0])

    # Rank within each block (row)
    rank_sums = [0.0] * k
    for i in range(n):
        row = [(groups[j][i], j) for j in range(k)]
        row.sort(key=lambda t: t[0])
        # Assign ranks with ties
        idx = 0
        while idx < k:
            jdx = idx
            while jdx < k and row[jdx][0] == row[idx][0]:
                jdx += 1
            avg_rank = (idx + jdx + 1) / 2
            for m in range(idx, jdx):
                rank_sums[row[m][1]] += avg_rank
            idx = jdx

    # Friedman statistic
    ss = sum(r**2 for r in rank_sums)
    chi2 = (12 / (n * k * (k + 1))) * ss - 3 * n * (k + 1)

    # Chi-squared p-value approximation (df = k-1)
    df = k - 1
    # Using Wilson-Hilferty approximation for chi-squared CDF
    if chi2 <= 0:
        p = 1.0
    else:
        z_approx = (((chi2 / df) ** (1/3)) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        p = 1 - 0.5 * (1 + math.erf(z_approx / math.sqrt(2)))

    return chi2, df, p, rank_sums

def pearson_r(x, y):
    """Pearson correlation coefficient with p-value."""
    n = len(x)
    if n < 3:
        return 0, 1.0
    mx, my = mean(x), mean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    sx, sy = std(x), std(y)
    if sx == 0 or sy == 0:
        return 0, 1.0
    r = cov / (sx * sy)
    # t-test for significance
    if abs(r) >= 1:
        return r, 0
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r**2)
    # Two-tailed p-value (approximate)
    df = n - 2
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))  # rough approximation
    return r, p

def effect_size_label(d):
    """Interpret Cohen's d."""
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."

# ===========================================================================
# Data Loading
# ===========================================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "results")

PHASES = {
    'phase1': ('Gen1', 'standard'),
    'phase2': ('Gen2_err', 'thinking'),  # will be fixed below
    'phase3a': ('Gen2', 'standard'),
    'phase3b': ('Gen2', 'thinking'),
}
# Fix phase2 label
PHASES['phase2'] = ('Gen1', 'thinking')

def load_all_data():
    """Load all result files into a structured dict.
    Returns:
        data: (phase, provider, config) -> list of per-query HR values
        tool_data: (phase, provider, config) -> list of (halluc, grounded, total)
        paired_data: (phase, provider, config) -> {(domain, prompt_idx, rep): hr}
            Enables explicit query-identity pairing for paired statistical tests.
    """
    data = defaultdict(list)
    tool_data = defaultdict(list)
    paired_data = defaultdict(dict)  # explicit pairing by query identity

    for phase_name in ['phase1', 'phase2', 'phase3a', 'phase3b']:
        phase_dir = os.path.join(BASE_DIR, phase_name)
        if not os.path.exists(phase_dir):
            print(f"WARNING: {phase_dir} not found")
            continue

        files = sorted(glob.glob(os.path.join(phase_dir, 'results_*.json')))
        for f in files:
            with open(f) as fh:
                results = json.load(fh)
            for item in results:
                provider = item.get('ProviderId', '?')
                config = item.get('ConfigId', '?')
                domain = item.get('DomainId', '?')
                h_tools = item.get('HallucinatedToolNames', [])
                g_tools = item.get('GroundedToolNames', [])
                h = len(h_tools)
                g = len(g_tools)
                total = h + g
                hr = h / total if total > 0 else 0

                data[(phase_name, provider, config)].append(hr)
                tool_data[(phase_name, provider, config)].append((h, g, total))
                # Also store by domain
                data[(phase_name, provider, config, domain)].append(hr)
                tool_data[(phase_name, provider, config, domain)].append((h, g, total))

                # Explicit pairing key: (domain, prompt_index, repetition)
                pairing_key = (domain, item.get('PromptIndex', 0), item.get('Repetition', 1))
                paired_data[(phase_name, provider, config)][pairing_key] = hr

    return data, tool_data, paired_data

def get_paired_arrays(paired_data, key1, key2):
    """Get two arrays explicitly paired by query identity (domain, prompt, repetition).
    Only includes queries present in BOTH conditions."""
    d1 = paired_data.get(key1, {})
    d2 = paired_data.get(key2, {})
    common_keys = sorted(set(d1.keys()) & set(d2.keys()))
    return [d1[k] for k in common_keys], [d2[k] for k in common_keys]

def get_multi_paired_arrays(paired_data, keys):
    """Get aligned arrays for multiple keys, paired by query identity.
    Used for Friedman test where all k conditions must share the same queries."""
    dicts = [paired_data.get(k, {}) for k in keys]
    if not all(dicts):
        return []
    common_keys = sorted(set.intersection(*[set(d.keys()) for d in dicts]))
    return [[d[k] for k in common_keys] for d in dicts]

# ===========================================================================
# Analysis Functions
# ===========================================================================

def analyze_confidence_intervals(data, tool_data):
    """A1.1: 95% CIs for all key HR values (bootstrap primary, parametric secondary)."""
    print("=" * 80)
    print("  A1.1: 95% CONFIDENCE INTERVALS FOR KEY HR VALUES")
    print("  (Bootstrap: 10,000 resamples, percentile method; Parametric: t-distribution)")
    print("=" * 80)

    for phase_label, phase_names in [
        ("GEN1 STANDARD (Phase 1)", ['phase1']),
        ("GEN1 THINKING (Phase 2)", ['phase2']),
        ("GEN2 STANDARD (Phase 3a)", ['phase3a']),
        ("GEN2 THINKING (Phase 3b)", ['phase3b']),
    ]:
        print(f"\n  --- {phase_label} ---")
        print(f"  {'Provider':<12} {'Config':<6} {'N':>4} {'Mean HR':>8} {'Bootstrap 95% CI':>24} {'Parametric 95% CI':>24} {'Std':>8}")
        print(f"  {'-'*92}")

        for phase_name in phase_names:
            configs = ['C0', 'C5'] if 'phase2' in phase_name or 'phase3b' in phase_name else ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
            for provider in ['openai', 'anthropic', 'google']:
                for config in configs:
                    key = (phase_name, provider, config)
                    if key not in data:
                        continue
                    vals = data[key]
                    n = len(vals)
                    m = mean(vals) * 100
                    blo, bhi = bootstrap_ci95(vals)
                    plo, phi = ci95(vals)
                    s = std(vals) * 100
                    print(f"  {provider:<12} {config:<6} {n:>4} {m:>7.1f}% [{blo*100:>5.1f}%, {bhi*100:>5.1f}%] [{plo*100:>5.1f}%, {phi*100:>5.1f}%] {s:>7.1f}%")

def analyze_c0_vs_c5(data, paired_data):
    """A1.2: Architecture effectiveness - C0 vs C5 significance tests.
    Uses Wilcoxon signed-rank (paired) for within-model comparisons,
    with explicit query-identity pairing by (domain, prompt, repetition)."""
    print("\n" + "=" * 80)
    print("  A1.2: C0 vs C5 SIGNIFICANCE TESTS (Wilcoxon Signed-Rank, Paired)")
    print("=" * 80)
    print(f"  {'Phase':<10} {'Provider':<12} {'N':>4} {'C0 Mean':>8} {'C5 Mean':>8} {'Diff':>8} {'T stat':>8} {'z':>7} {'p-value':>10} {'Sig':>5} {'r_rb':>7} {'Cliff d':>8} {'Cl.Eff':>10} {'Cohen d':>8} {'Co.Eff':>10}")
    print(f"  {'-'*144}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                      ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
        for provider in ['openai', 'anthropic', 'google']:
            c0, c5 = get_paired_arrays(paired_data,
                                       (phase_name, provider, 'C0'),
                                       (phase_name, provider, 'C5'))
            if not c0:
                continue

            m0 = mean(c0) * 100
            m5 = mean(c5) * 100
            t_stat, z, p = wilcoxon_signed_rank(c0, c5)
            r_rb = rank_biserial_r(c0, c5)
            cd = cliffs_delta(c0, c5)
            d = cohens_d(c0, c5)
            print(f"  {phase_label:<10} {provider:<12} {len(c0):>4} {m0:>7.1f}% {m5:>7.1f}% {m5-m0:>+7.1f}pp {t_stat:>8.0f} {z:>7.2f} {p:>10.6f} {sig_label(p):>5} {r_rb:>+7.3f} {cd:>+7.3f}  {cliffs_delta_label(cd):>10} {d:>+7.2f}  {effect_size_label(d):>10}")

def analyze_std_vs_thinking(data, paired_data):
    """A1.3: Standard vs Thinking significance tests.
    Primary: Paired Wilcoxon signed-rank (same 144 queries across modes).
    Robustness: Mann-Whitney U (unpaired, independent samples).
    Convention: thinking-first (positive r_rb/delta = thinking worse than standard)."""
    print("\n" + "=" * 80)
    print("  A1.3: STANDARD vs THINKING SIGNIFICANCE TESTS (C5 only)")
    print("  Primary: Paired Wilcoxon signed-rank | Robustness: Mann-Whitney U")
    print("  Convention: diffs = thinking - standard; positive r_rb = thinking worse")
    print("=" * 80)

    # --- Per-provider paired Wilcoxon (primary) ---
    print(f"\n  --- PRIMARY: Paired Wilcoxon signed-rank ---")
    print(f"  {'Gen':<6} {'Provider':<12} {'N':>4} {'Std C5':>8} {'Thn C5':>8} {'Diff':>8} {'T stat':>8} {'z':>7} {'p-value':>10} {'Sig':>5} {'r_rb':>7} {'Cliff d':>8} {'Cl.Eff':>10} {'Cohen d':>8} {'Co.Eff':>10}")
    print(f"  {'-'*140}")

    for gen_label, std_phase, thn_phase in [('Gen1', 'phase1', 'phase2'), ('Gen2', 'phase3a', 'phase3b')]:
        # Collect pooled arrays for aggregate test
        all_std = []
        all_thn = []
        for provider in ['openai', 'anthropic', 'google']:
            std_arr, thn_arr = get_paired_arrays(paired_data,
                                                  (std_phase, provider, 'C5'),
                                                  (thn_phase, provider, 'C5'))
            if not std_arr:
                continue

            ms = mean(std_arr) * 100
            mt = mean(thn_arr) * 100
            t_stat, z, p = wilcoxon_signed_rank(thn_arr, std_arr)
            r_rb = rank_biserial_r(thn_arr, std_arr)
            cd = cliffs_delta(thn_arr, std_arr)
            d = cohens_d(thn_arr, std_arr)
            print(f"  {gen_label:<6} {provider:<12} {len(std_arr):>4} {ms:>7.1f}% {mt:>7.1f}% {mt-ms:>+7.1f}pp {t_stat:>8.0f} {z:>7.2f} {p:>10.6f} {sig_label(p):>5} {r_rb:>+7.3f} {cd:>+7.3f}  {cliffs_delta_label(cd):>10} {d:>+7.2f}  {effect_size_label(d):>10}")
            all_std.extend(std_arr)
            all_thn.extend(thn_arr)

        # Aggregate across providers
        if all_std:
            ms = mean(all_std) * 100
            mt = mean(all_thn) * 100
            t_stat, z, p = wilcoxon_signed_rank(all_thn, all_std)
            r_rb = rank_biserial_r(all_thn, all_std)
            cd = cliffs_delta(all_thn, all_std)
            d = cohens_d(all_thn, all_std)
            print(f"  {gen_label:<6} {'AGGREGATE':<12} {len(all_std):>4} {ms:>7.1f}% {mt:>7.1f}% {mt-ms:>+7.1f}pp {t_stat:>8.0f} {z:>7.2f} {p:>10.6f} {sig_label(p):>5} {r_rb:>+7.3f} {cd:>+7.3f}  {cliffs_delta_label(cd):>10} {d:>+7.2f}  {effect_size_label(d):>10}")

    # --- Per-provider Mann-Whitney U (robustness check) ---
    print(f"\n  --- ROBUSTNESS CHECK: Mann-Whitney U (unpaired) ---")
    print(f"  {'Gen':<6} {'Provider':<12} {'Std C5':>8} {'Thn C5':>8} {'Diff':>8} {'U stat':>8} {'z':>7} {'p-value':>10} {'Sig':>5} {'Cliff d':>8} {'Cl.Eff':>10} {'Cohen d':>8} {'Co.Eff':>10}")
    print(f"  {'-'*130}")

    for gen_label, std_phase, thn_phase in [('Gen1', 'phase1', 'phase2'), ('Gen2', 'phase3a', 'phase3b')]:
        for provider in ['openai', 'anthropic', 'google']:
            std_c5 = data.get((std_phase, provider, 'C5'), [])
            thn_c5 = data.get((thn_phase, provider, 'C5'), [])
            if not std_c5 or not thn_c5:
                continue

            ms = mean(std_c5) * 100
            mt = mean(thn_c5) * 100
            u, z, p = mann_whitney_u(thn_c5, std_c5)
            cd = cliffs_delta(thn_c5, std_c5)
            d = cohens_d(thn_c5, std_c5)
            print(f"  {gen_label:<6} {provider:<12} {ms:>7.1f}% {mt:>7.1f}% {mt-ms:>+7.1f}pp {u:>8.0f} {z:>7.2f} {p:>10.6f} {sig_label(p):>5} {cd:>+7.3f}  {cliffs_delta_label(cd):>10} {d:>+7.2f}  {effect_size_label(d):>10}")

def analyze_c3_anomaly(data, paired_data):
    """A1.4: C3 anomaly significance tests (C3 > C0).
    Uses Wilcoxon signed-rank (paired) with explicit query-identity pairing."""
    print("\n" + "=" * 80)
    print("  A1.4: C3 ANOMALY SIGNIFICANCE TESTS (C3 > C0, one-sided, paired)")
    print("=" * 80)
    print(f"  {'Phase':<10} {'Provider':<12} {'N':>4} {'C0 Mean':>8} {'C3 Mean':>8} {'Diff':>8} {'T stat':>8} {'z':>7} {'p-value':>10} {'Sig':>5} {'r_rb':>7} {'Cliff d':>8} {'Cl.Eff':>10} {'Cohen d':>8}")
    print(f"  {'-'*128}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase3a', 'Gen2 Std')]:
        for provider in ['openai', 'anthropic', 'google']:
            c0, c3 = get_paired_arrays(paired_data,
                                       (phase_name, provider, 'C0'),
                                       (phase_name, provider, 'C3'))
            if not c0:
                continue

            m0 = mean(c0) * 100
            m3 = mean(c3) * 100
            t_stat, z, p = wilcoxon_signed_rank(c3, c0, alternative='greater')
            r_rb = rank_biserial_r(c3, c0)
            cd = cliffs_delta(c3, c0)
            d = cohens_d(c3, c0)
            print(f"  {phase_label:<10} {provider:<12} {len(c0):>4} {m0:>7.1f}% {m3:>7.1f}% {m3-m0:>+7.1f}pp {t_stat:>8.0f} {z:>7.2f} {p:>10.6f} {sig_label(p):>5} {r_rb:>+7.3f} {cd:>+7.3f}  {cliffs_delta_label(cd):>10} {d:>+7.2f}")

def analyze_cross_generational(data):
    """A1.5: Cross-generational stability tests (Gen1 C5 vs Gen2 C5)."""
    print("\n" + "=" * 80)
    print("  A1.5: CROSS-GENERATIONAL STABILITY TESTS (Gen1 C5 vs Gen2 C5)")
    print("=" * 80)
    print(f"  {'Mode':<10} {'Provider':<12} {'Gen1 C5':>8} {'Gen2 C5':>8} {'Diff':>8} {'U stat':>8} {'z':>7} {'p-value':>10} {'Sig':>5} {'Cliff d':>8} {'Cl.Eff':>10} {'Cohen d':>8} {'Co.Eff':>10}")
    print(f"  {'-'*135}")

    for mode_label, gen1_phase, gen2_phase in [('Standard', 'phase1', 'phase3a'), ('Thinking', 'phase2', 'phase3b')]:
        for provider in ['openai', 'anthropic', 'google']:
            g1 = data.get((gen1_phase, provider, 'C5'), [])
            g2 = data.get((gen2_phase, provider, 'C5'), [])
            if not g1 or not g2:
                continue

            m1 = mean(g1) * 100
            m2 = mean(g2) * 100
            u, z, p = mann_whitney_u(g1, g2)
            cd = cliffs_delta(g1, g2)
            d = cohens_d(g1, g2)
            print(f"  {mode_label:<10} {provider:<12} {m1:>7.1f}% {m2:>7.1f}% {m2-m1:>+7.1f}pp {u:>8.0f} {z:>7.2f} {p:>10.6f} {sig_label(p):>5} {cd:>+7.3f}  {cliffs_delta_label(cd):>10} {d:>+7.2f}  {effect_size_label(d):>10}")

def analyze_ablation_ordering(data, paired_data):
    """A1.6: Friedman test for ablation ordering.
    Uses explicit query-identity pairing across all 6 configurations."""
    print("\n" + "=" * 80)
    print("  A1.6: ABLATION ORDERING SIGNIFICANCE (Friedman Test)")
    print("=" * 80)

    for phase_name, phase_label in [('phase1', 'Gen1 Standard'), ('phase3a', 'Gen2 Standard')]:
        print(f"\n  --- {phase_label} ---")
        for provider in ['openai', 'anthropic', 'google']:
            configs = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
            keys = [(phase_name, provider, cfg) for cfg in configs]
            groups = get_multi_paired_arrays(paired_data, keys)

            if len(groups) != 6 or not groups[0]:
                continue

            n_paired = len(groups[0])
            chi2, df, p, rank_sums = friedman_test(groups)

            print(f"  {provider} (N={n_paired} paired queries): chi2={chi2:.2f}, df={df}, p={p:.6f} {sig_label(p)}")
            print(f"    Rank sums: {', '.join(f'{configs[i]}={rank_sums[i]:.1f}' for i in range(6))}")

            # Planned pairwise Wilcoxon signed-rank (paired, consistent with Friedman)
            # C0 vs C3 tested separately in analyze_c3_anomaly (one-sided)
            print(f"    Planned pairwise comparisons (Wilcoxon signed-rank, Bonferroni k=3):")
            pairs = [('C2', 'C5'), ('C4', 'C5'), ('C2', 'C4')]
            for cfg_a, cfg_b in pairs:
                ia, ib = configs.index(cfg_a), configs.index(cfg_b)
                t_stat, z, p_pair = wilcoxon_signed_rank(groups[ia], groups[ib])
                # Bonferroni correction (3 planned contrasts)
                p_adj = min(p_pair * 3, 1.0)
                m_a = mean(groups[ia]) * 100
                m_b = mean(groups[ib]) * 100
                r_rb = rank_biserial_r(groups[ia], groups[ib])
                cd = cliffs_delta(groups[ia], groups[ib])
                d = cohens_d(groups[ia], groups[ib])
                print(f"      {cfg_a} ({m_a:.1f}%) vs {cfg_b} ({m_b:.1f}%): p={p_pair:.4f}, p_adj={p_adj:.4f} {sig_label(p_adj)}, r_rb={r_rb:+.3f}, delta={cd:+.3f} ({cliffs_delta_label(cd)}), d={d:+.2f}")

def analyze_verbosity_correlation(data, tool_data):
    """B1: Verbosity-hallucination correlation analysis."""
    print("\n" + "=" * 80)
    print("  B1: VERBOSITY-HALLUCINATION CORRELATION")
    print("=" * 80)

    models = []
    tools_per_q = []
    hr_values = []

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                      ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
        for provider in ['openai', 'anthropic', 'google']:
            key = (phase_name, provider, 'C5')
            if key not in tool_data:
                continue

            td = tool_data[key]
            total_h = sum(t[0] for t in td)
            total_g = sum(t[1] for t in td)
            total_t = sum(t[2] for t in td)
            n_queries = len(td)

            tpq = total_t / n_queries if n_queries > 0 else 0
            hr = total_h / total_t * 100 if total_t > 0 else 0
            per_mention = total_h / total_t * 100 if total_t > 0 else 0

            models.append(f"{phase_label} {provider}")
            tools_per_q.append(tpq)
            hr_values.append(hr)

    print(f"\n  {'Model':<25} {'Tools/Q':>8} {'HR%':>8}")
    print(f"  {'-'*43}")
    for i in range(len(models)):
        print(f"  {models[i]:<25} {tools_per_q[i]:>7.1f} {hr_values[i]:>7.1f}%")

    r, p = pearson_r(tools_per_q, hr_values)
    print(f"\n  Pearson r = {r:.3f}, p = {p:.4f} {sig_label(p)}")
    print(f"  N = {len(models)} model configurations")

    if p >= 0.05:
        print(f"  >>> CORRELATION IS NOT STATISTICALLY SIGNIFICANT")
        print(f"  >>> Paper claim about verbosity-hallucination correlation needs revision")

def analyze_per_tool_mention(data, tool_data):
    """C1: Per-tool-mention hallucination rates."""
    print("\n" + "=" * 80)
    print("  C1: PER-TOOL-MENTION HALLUCINATION RATES (C5)")
    print("=" * 80)
    print(f"  {'Phase':<10} {'Provider':<12} {'Total H':>8} {'Total T':>8} {'Per-Mention':>12} {'Tools/Q':>8} {'Agg HR':>8}")
    print(f"  {'-'*72}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                      ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
        for provider in ['openai', 'anthropic', 'google']:
            key = (phase_name, provider, 'C5')
            if key not in tool_data:
                continue

            td = tool_data[key]
            total_h = sum(t[0] for t in td)
            total_t = sum(t[2] for t in td)
            n = len(td)

            tpq = total_t / n if n > 0 else 0
            per_mention = total_h / total_t * 100 if total_t > 0 else 0
            agg_hr = mean(data[(phase_name, provider, 'C5')]) * 100

            print(f"  {phase_label:<10} {provider:<12} {total_h:>8} {total_t:>8} {per_mention:>10.2f}% {tpq:>7.1f} {agg_hr:>7.1f}%")

def analyze_gen2_domain_detail(data, tool_data):
    """C2: Gen2 domain-level detail for missing analysis."""
    print("\n" + "=" * 80)
    print("  C2: GEN2 DOMAIN-SPECIFIC C5 HALLUCINATION RATES")
    print("=" * 80)

    for phase_name, phase_label in [('phase3a', 'Gen2 Standard'), ('phase3b', 'Gen2 Thinking')]:
        print(f"\n  --- {phase_label} ---")
        print(f"  {'Provider':<12} {'D1':>8} {'D2':>8} {'D3':>8} {'D4':>8} {'Avg':>8}")
        print(f"  {'-'*50}")

        for provider in ['openai', 'anthropic', 'google']:
            domain_hrs = []
            for domain in ['D1', 'D2', 'D3', 'D4']:
                key = (phase_name, provider, 'C5', domain)
                if key in tool_data:
                    td = tool_data[key]
                    total_h = sum(t[0] for t in td)
                    total_t = sum(t[2] for t in td)
                    hr = total_h / total_t * 100 if total_t > 0 else 0
                    domain_hrs.append(hr)
                else:
                    domain_hrs.append(0)
            avg = mean(domain_hrs) if domain_hrs else 0
            print(f"  {provider:<12} {domain_hrs[0]:>7.1f}% {domain_hrs[1]:>7.1f}% {domain_hrs[2]:>7.1f}% {domain_hrs[3]:>7.1f}% {avg:>7.1f}%")

def summary_for_paper(data, tool_data, paired_data):
    """Generate LaTeX-ready summary of statistical results."""
    print("\n" + "=" * 80)
    print("  SUMMARY: KEY STATISTICS FOR PAPER")
    print("=" * 80)

    print("\n  --- C5 HR with 95% CI (bootstrap primary, parametric secondary) ---")
    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase3a', 'Gen2 Std')]:
        for provider in ['openai', 'anthropic', 'google']:
            vals = data.get((phase_name, provider, 'C5'), [])
            if vals:
                m = mean(vals) * 100
                blo, bhi = bootstrap_ci95(vals)
                plo, phi = ci95(vals)
                print(f"  {phase_label} {provider}: {m:.1f}% boot[{blo*100:.1f}, {bhi*100:.1f}] param[{plo*100:.1f}, {phi*100:.1f}]")

    print("\n  --- Standard vs Thinking summary ---")
    for gen, sp, tp in [('Gen1', 'phase1', 'phase2'), ('Gen2', 'phase3a', 'phase3b')]:
        all_std = []
        all_thn = []
        for prov in ['openai', 'anthropic', 'google']:
            all_std.extend(data.get((sp, prov, 'C5'), []))
            all_thn.extend(data.get((tp, prov, 'C5'), []))
        u, z, p = mann_whitney_u(all_std, all_thn)
        cd = cliffs_delta(all_std, all_thn)
        d = cohens_d(all_std, all_thn)
        print(f"  {gen} cross-provider: std={mean(all_std)*100:.1f}%, thn={mean(all_thn)*100:.1f}%, p={p:.4f} {sig_label(p)}, delta={cd:+.3f} ({cliffs_delta_label(cd)}), d={d:+.2f} ({effect_size_label(d)})")

    print("\n  --- C3 anomaly summary (paired by query identity) ---")
    for phase_name, phase_label in [('phase1', 'Gen1'), ('phase3a', 'Gen2')]:
        all_c0 = []
        all_c3 = []
        for prov in ['openai', 'anthropic', 'google']:
            c0, c3 = get_paired_arrays(paired_data,
                                       (phase_name, prov, 'C0'),
                                       (phase_name, prov, 'C3'))
            all_c0.extend(c0)
            all_c3.extend(c3)
        t, z, p = wilcoxon_signed_rank(all_c3, all_c0, alternative='greater')
        r_rb = rank_biserial_r(all_c3, all_c0)
        cd = cliffs_delta(all_c3, all_c0)
        d = cohens_d(all_c3, all_c0)
        print(f"  {phase_label}: C0={mean(all_c0)*100:.1f}%, C3={mean(all_c3)*100:.1f}%, N={len(all_c0)} pairs, p={p:.6f} {sig_label(p)}, r_rb={r_rb:+.3f}, delta={cd:+.3f} ({cliffs_delta_label(cd)}), d={d:+.2f} ({effect_size_label(d)})")

# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print("Loading data from:", BASE_DIR)
    data, tool_data, paired_data = load_all_data()

    total = sum(len(v) for k, v in data.items() if len(k) == 3)
    print(f"Loaded {total} query-level results")

    # Sanity check: total must equal expected experiment size
    EXPECTED_TOTAL = 6912  # 2592 (phase1) + 864 (phase2) + 2592 (phase3a) + 864 (phase3b)
    if total != EXPECTED_TOTAL:
        print(f"  *** ERROR: Expected {EXPECTED_TOTAL} entries, got {total}. "
              f"Results directory may contain extra/missing files. ***")
        sys.exit(1)
    else:
        print(f"  Sanity check PASSED: {total} == {EXPECTED_TOTAL} expected entries")

    # Per-phase sanity checks
    expected_per_phase = {'phase1': 2592, 'phase2': 864, 'phase3a': 2592, 'phase3b': 864}
    for phase, expected_n in expected_per_phase.items():
        actual_n = sum(len(v) for k, v in data.items() if len(k) == 3 and k[0] == phase)
        if actual_n != expected_n:
            print(f"  *** WARNING: {phase} has {actual_n} entries, expected {expected_n} ***")
        else:
            print(f"  {phase}: {actual_n} entries OK")
    print()

    analyze_confidence_intervals(data, tool_data)
    analyze_c0_vs_c5(data, paired_data)
    analyze_std_vs_thinking(data, paired_data)
    analyze_c3_anomaly(data, paired_data)
    analyze_cross_generational(data)
    analyze_ablation_ordering(data, paired_data)
    analyze_verbosity_correlation(data, tool_data)
    analyze_per_tool_mention(data, tool_data)
    analyze_gen2_domain_detail(data, tool_data)
    summary_for_paper(data, tool_data, paired_data)

    # =======================================================================
    # Optional scipy verification for borderline p-values (MF-3)
    # =======================================================================
    try:
        from scipy import stats as sp_stats
        print("\n" + "=" * 80)
        print("  SCIPY VERIFICATION (independent check of key p-values)")
        print("=" * 80)

        # Verify borderline Google Gen2 C2 vs C5 (p_adj=0.027 in paper)
        for phase_name, phase_label in [('phase3a', 'Gen2')]:
            for provider in ['openai', 'anthropic', 'google']:
                configs = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
                pairs = [('C2', 'C5'), ('C4', 'C5'), ('C2', 'C4')]
                for cfg_a, cfg_b in pairs:
                    a, b = get_paired_arrays(paired_data,
                                             (phase_name, provider, cfg_a),
                                             (phase_name, provider, cfg_b))
                    if not a:
                        continue
                    sp_result = sp_stats.wilcoxon(a, b, alternative='two-sided')
                    custom_t, custom_z, custom_p = wilcoxon_signed_rank(a, b)
                    p_adj_sp = min(sp_result.pvalue * 3, 1.0)
                    p_adj_custom = min(custom_p * 3, 1.0)
                    match = "OK" if abs(p_adj_sp - p_adj_custom) < 0.005 else "DIFFERS"
                    print(f"  {phase_label} {provider} {cfg_a} vs {cfg_b}: "
                          f"scipy p_adj={p_adj_sp:.4f}, custom p_adj={p_adj_custom:.4f} [{match}]")

        # Verify all C0 vs C5 paired tests
        print("\n  --- C0 vs C5 verification (two-sided) ---")
        for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                          ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
            for provider in ['openai', 'anthropic', 'google']:
                c0, c5 = get_paired_arrays(paired_data,
                                           (phase_name, provider, 'C0'),
                                           (phase_name, provider, 'C5'))
                if not c0:
                    continue
                sp_result = sp_stats.wilcoxon(c0, c5, alternative='two-sided')
                _, _, custom_p = wilcoxon_signed_rank(c0, c5)
                match = "OK" if abs(sp_result.pvalue - custom_p) < 0.005 else "DIFFERS"
                print(f"  {phase_label} {provider}: scipy p={sp_result.pvalue:.6f}, custom p={custom_p:.6f} [{match}]")

        # Verify one-sided C3 anomaly tests (C3 > C0)
        print("\n  --- C3 anomaly verification (one-sided: greater) ---")
        for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase3a', 'Gen2 Std')]:
            for provider in ['openai', 'anthropic', 'google']:
                c0, c3 = get_paired_arrays(paired_data,
                                           (phase_name, provider, 'C0'),
                                           (phase_name, provider, 'C3'))
                if not c0:
                    continue
                sp_result = sp_stats.wilcoxon(c3, c0, alternative='greater')
                _, _, custom_p = wilcoxon_signed_rank(c3, c0, alternative='greater')
                match = "OK" if abs(sp_result.pvalue - custom_p) < 0.005 else "DIFFERS"
                print(f"  {phase_label} {provider}: scipy p={sp_result.pvalue:.6f}, custom p={custom_p:.6f} [{match}]")

    except ImportError:
        print("\n  [scipy not installed -- install with 'pip install scipy' for independent p-value verification]")
