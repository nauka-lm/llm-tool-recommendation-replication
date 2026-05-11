#!/usr/bin/env python3
"""
Compute Zero-Difference Pair Percentages for Wilcoxon Signed-Rank Tests

For the journal paper, this script computes the percentage of paired
observations that have identical hallucination rates (zero difference) across
all paired Wilcoxon signed-rank comparisons used in the statistical analysis.

The Wilcoxon signed-rank test discards zero-difference pairs before ranking.
We report the maximum percentage across all comparison cells so the paper
can state: "zero-difference pairs represent <= X% of observations across cells."

Paired comparisons using Wilcoxon signed-rank:
  A1.2: C0 vs C5 (4 phases x 3 providers = 12 cells)
  A1.4: C3 anomaly C0 vs C3 (2 phases x 3 providers = 6 cells)
  A1.6: Ablation pairwise C2-C5, C4-C5, C2-C4 (2 phases x 3 providers x 3 pairs = 18 cells)

Note: A1.3 Standard vs Thinking uses Mann-Whitney U (unpaired), so excluded.
"""

import json
import glob
import os
from collections import defaultdict

# ===========================================================================
# Configuration
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "data", "results")

PROVIDERS = ['openai', 'anthropic', 'google']

# ===========================================================================
# Data Loading (same logic as statistical_analysis.py)
# ===========================================================================

def load_paired_data():
    """Load all result files and build paired_data dict.
    Returns:
        paired_data: (phase, provider, config) -> {(domain, prompt_idx, rep): hr}
    """
    paired_data = defaultdict(dict)

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

                pairing_key = (domain, item.get('PromptIndex', 0), item.get('Repetition', 1))
                paired_data[(phase_name, provider, config)][pairing_key] = hr

    return paired_data


def get_paired_arrays(paired_data, key1, key2):
    """Get two arrays explicitly paired by query identity (domain, prompt, repetition).
    Only includes queries present in BOTH conditions."""
    d1 = paired_data.get(key1, {})
    d2 = paired_data.get(key2, {})
    common_keys = sorted(set(d1.keys()) & set(d2.keys()))
    return [d1[k] for k in common_keys], [d2[k] for k in common_keys]


def compute_zero_diff(arr_a, arr_b):
    """Compute zero-difference statistics for a paired comparison."""
    total_pairs = len(arr_a)
    zero_pairs = sum(1 for a, b in zip(arr_a, arr_b) if a == b)
    pct = (zero_pairs / total_pairs * 100) if total_pairs > 0 else 0
    return total_pairs, zero_pairs, pct


# ===========================================================================
# Main Analysis
# ===========================================================================

def main():
    print("=" * 90)
    print("  ZERO-DIFFERENCE PAIR ANALYSIS FOR WILCOXON SIGNED-RANK TESTS")
    print("  (Pairs where both conditions have identical hallucination rates)")
    print("=" * 90)
    print(f"\n  Data directory: {os.path.abspath(BASE_DIR)}\n")

    paired_data = load_paired_data()

    all_results = []

    # -----------------------------------------------------------------------
    # A1.2: C0 vs C5 (all 4 phases x 3 providers)
    # -----------------------------------------------------------------------
    print("-" * 90)
    print("  A1.2: C0 vs C5 PAIRED COMPARISONS (Wilcoxon Signed-Rank)")
    print("-" * 90)
    print(f"  {'Phase':<12} {'Provider':<12} {'Total Pairs':>12} {'Zero-Diff':>10} {'Pct':>8}")
    print(f"  {'-'*54}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                     ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
        for provider in PROVIDERS:
            a, b = get_paired_arrays(paired_data,
                                     (phase_name, provider, 'C0'),
                                     (phase_name, provider, 'C5'))
            if not a:
                continue
            total, zeros, pct = compute_zero_diff(a, b)
            all_results.append(('A1.2 C0vC5', phase_label, provider, total, zeros, pct))
            print(f"  {phase_label:<12} {provider:<12} {total:>12} {zeros:>10} {pct:>7.1f}%")

    # -----------------------------------------------------------------------
    # A1.4: C3 Anomaly - C0 vs C3 (phase1, phase3a x 3 providers)
    # -----------------------------------------------------------------------
    print(f"\n{'-' * 90}")
    print("  A1.4: C0 vs C3 ANOMALY PAIRED COMPARISONS (Wilcoxon Signed-Rank)")
    print("-" * 90)
    print(f"  {'Phase':<12} {'Provider':<12} {'Total Pairs':>12} {'Zero-Diff':>10} {'Pct':>8}")
    print(f"  {'-'*54}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase3a', 'Gen2 Std')]:
        for provider in PROVIDERS:
            a, b = get_paired_arrays(paired_data,
                                     (phase_name, provider, 'C0'),
                                     (phase_name, provider, 'C3'))
            if not a:
                continue
            total, zeros, pct = compute_zero_diff(a, b)
            all_results.append(('A1.4 C0vC3', phase_label, provider, total, zeros, pct))
            print(f"  {phase_label:<12} {provider:<12} {total:>12} {zeros:>10} {pct:>7.1f}%")

    # -----------------------------------------------------------------------
    # A1.6: Ablation Pairwise - C2vC5, C4vC5, C2vC4 (phase1, phase3a x 3 providers)
    # -----------------------------------------------------------------------
    print(f"\n{'-' * 90}")
    print("  A1.6: ABLATION PAIRWISE COMPARISONS (Wilcoxon Signed-Rank)")
    print("-" * 90)
    print(f"  {'Phase':<12} {'Provider':<12} {'Pair':<8} {'Total Pairs':>12} {'Zero-Diff':>10} {'Pct':>8}")
    print(f"  {'-'*62}")

    ablation_pairs = [('C2', 'C5'), ('C4', 'C5'), ('C2', 'C4')]
    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase3a', 'Gen2 Std')]:
        for provider in PROVIDERS:
            for cfg_a, cfg_b in ablation_pairs:
                a, b = get_paired_arrays(paired_data,
                                         (phase_name, provider, cfg_a),
                                         (phase_name, provider, cfg_b))
                if not a:
                    continue
                total, zeros, pct = compute_zero_diff(a, b)
                pair_label = f"{cfg_a}v{cfg_b}"
                all_results.append((f'A1.6 {pair_label}', phase_label, provider, total, zeros, pct))
                print(f"  {phase_label:<12} {provider:<12} {pair_label:<8} {total:>12} {zeros:>10} {pct:>7.1f}%")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print("=" * 90)

    total_comparisons = len(all_results)
    total_pairs_all = sum(r[3] for r in all_results)
    total_zeros_all = sum(r[4] for r in all_results)
    overall_pct = (total_zeros_all / total_pairs_all * 100) if total_pairs_all > 0 else 0

    max_pct = max(r[5] for r in all_results) if all_results else 0
    max_entry = [r for r in all_results if r[5] == max_pct][0] if all_results else None
    min_pct = min(r[5] for r in all_results) if all_results else 0
    min_entry = [r for r in all_results if r[5] == min_pct][0] if all_results else None

    print(f"\n  Total Wilcoxon comparison cells: {total_comparisons}")
    print(f"  Total paired observations across all cells: {total_pairs_all}")
    print(f"  Total zero-difference pairs across all cells: {total_zeros_all}")
    print(f"  Overall zero-difference percentage: {overall_pct:.1f}%")
    print(f"\n  MAXIMUM zero-diff %: {max_pct:.1f}%  ({max_entry[0]}, {max_entry[1]}, {max_entry[2]})")
    print(f"  MINIMUM zero-diff %: {min_pct:.1f}%  ({min_entry[0]}, {min_entry[1]}, {min_entry[2]})")

    # Also show distribution
    pcts = [r[5] for r in all_results]
    pcts.sort()
    avg_pct = sum(pcts) / len(pcts) if pcts else 0
    median_idx = len(pcts) // 2
    median_pct = pcts[median_idx] if pcts else 0

    print(f"\n  Average zero-diff % across cells: {avg_pct:.1f}%")
    print(f"  Median zero-diff % across cells: {median_pct:.1f}%")

    # Show counts at various thresholds
    print(f"\n  Distribution of zero-diff percentages:")
    for threshold in [0, 5, 10, 15, 20, 25, 30, 40, 50]:
        count = sum(1 for p in pcts if p <= threshold)
        print(f"    Cells with <= {threshold:>2}% zero-diff: {count}/{total_comparisons}")

    # Paper-ready statement
    print(f"\n  {'=' * 70}")
    print(f"  PAPER-READY STATEMENT:")
    print(f"  \"Zero-difference pairs (discarded by the Wilcoxon signed-rank test)")
    print(f"   represent at most {max_pct:.1f}% of observations in any comparison cell")
    print(f"   (mean {avg_pct:.1f}%, median {median_pct:.1f}% across {total_comparisons} cells).\"")
    print(f"  {'=' * 70}")


if __name__ == '__main__':
    main()
