"""
Computation 1: Paired Wilcoxon signed-rank tests for standard vs thinking (C5 only)
Computation 2: E[H per query] for all 12 model configurations under C5 and C0

Uses experiment results from ../data/results/ directory.
"""

import sys
import json
import glob
import os
import numpy as np
from scipy import stats
from collections import defaultdict

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "results")

# ── Model configuration ──────────────────────────────────────────────────────

# Gen1 standard (phase1) and thinking (phase2)
GEN1_STANDARD = {
    'openai': ('phase1', 'gpt-4.1', 'standard'),
    'anthropic': ('phase1', 'claude-sonnet-4-5-20250929', 'standard'),
    'google': ('phase1', 'gemini-2.5-flash-lite', 'standard'),
}
GEN1_THINKING = {
    'openai': ('phase2', 'o4-mini', 'thinking'),
    'anthropic': ('phase2', 'claude-sonnet-4-5-20250929', 'thinking'),
    'google': ('phase2', 'gemini-2.5-flash', 'thinking'),
}

# Gen2 standard (phase3a) and thinking (phase3b)
GEN2_STANDARD = {
    'openai': ('phase3a', 'gpt-5.2', 'standard'),
    'anthropic': ('phase3a', 'claude-sonnet-4-6', 'standard'),
    'google': ('phase3a', 'gemini-3.1-flash-lite-preview', 'standard'),
}
GEN2_THINKING = {
    'openai': ('phase3b', 'gpt-5.2', 'thinking'),
    'anthropic': ('phase3b', 'claude-sonnet-4-6', 'thinking'),
    'google': ('phase3b', 'gemini-3.1-flash-lite-preview', 'thinking'),
}

# Display names
MODEL_DISPLAY = {
    ('phase1', 'gpt-4.1', 'standard'): 'GPT-4.1',
    ('phase2', 'o4-mini', 'thinking'): 'o4-mini',
    ('phase1', 'claude-sonnet-4-5-20250929', 'standard'): 'Claude Sonnet 4.5',
    ('phase2', 'claude-sonnet-4-5-20250929', 'thinking'): 'Claude Sonnet 4.5 (thinking)',
    ('phase1', 'gemini-2.5-flash-lite', 'standard'): 'Gemini 2.5 Flash-Lite',
    ('phase2', 'gemini-2.5-flash', 'thinking'): 'Gemini 2.5 Flash',
    ('phase3a', 'gpt-5.2', 'standard'): 'GPT-5.2',
    ('phase3b', 'gpt-5.2', 'thinking'): 'GPT-5.2 (thinking)',
    ('phase3a', 'claude-sonnet-4-6', 'standard'): 'Claude Sonnet 4.6',
    ('phase3b', 'claude-sonnet-4-6', 'thinking'): 'Claude Sonnet 4.6 (thinking)',
    ('phase3a', 'gemini-3.1-flash-lite-preview', 'standard'): 'Gemini 3.1 Flash-Lite',
    ('phase3b', 'gemini-3.1-flash-lite-preview', 'thinking'): 'Gemini 3.1 Flash-Lite (thinking)',
}


def load_results(phase, model_id, model_category, config_id):
    """Load all results for a given phase/model/config, return dict keyed by (Domain, PromptIndex, Repetition)."""
    results = {}
    pattern = os.path.join(BASE, phase, f'results_*_{config_id}.json')
    for fpath in glob.glob(pattern):
        with open(fpath, 'r') as f:
            data = json.load(f)
        for r in data:
            if r['ModelId'] != model_id or r['ModelCategory'] != model_category:
                continue
            if r['ConfigId'] != config_id:
                continue
            key = (r['DomainId'], r['PromptIndex'], r['Repetition'])
            # Compute hallucination rate
            total = len(r['MentionedToolNames'])
            hallucinated = len(r['HallucinatedToolNames'])
            h_rate = hallucinated / total if total > 0 else 0.0
            results[key] = {
                'h_rate': h_rate,
                'hallucinated_count': hallucinated,
                'total_mentioned': total,
                'grounded_count': len(r['GroundedToolNames']),
            }
    return results


def compute_wilcoxon(standard_data, thinking_data):
    """
    Compute paired Wilcoxon signed-rank test on hallucination rates.
    Returns dict with p-value, direction, concordant/discordant/tied counts, and rank-biserial r.
    """
    # Get matched keys
    common_keys = sorted(set(standard_data.keys()) & set(thinking_data.keys()))
    n = len(common_keys)

    std_rates = np.array([standard_data[k]['h_rate'] for k in common_keys])
    think_rates = np.array([thinking_data[k]['h_rate'] for k in common_keys])

    diffs = std_rates - think_rates  # positive = standard worse (higher hallucination)

    # Count concordant/discordant/tied
    n_std_worse = np.sum(diffs > 0)   # standard has higher h_rate
    n_think_worse = np.sum(diffs < 0)  # thinking has higher h_rate
    n_tied = np.sum(diffs == 0)

    # Wilcoxon signed-rank test (two-sided)
    # Remove zero-differences for the test
    nonzero_diffs = diffs[diffs != 0]

    if len(nonzero_diffs) == 0:
        return {
            'n_matched': n,
            'n_std_worse': int(n_std_worse),
            'n_think_worse': int(n_think_worse),
            'n_tied': int(n_tied),
            'p_value': 1.0,
            'statistic': 0.0,
            'direction': 'tied (all pairs identical)',
            'r_rb': 0.0,
            'mean_std': float(np.mean(std_rates)),
            'mean_think': float(np.mean(think_rates)),
        }

    try:
        stat, p_val = stats.wilcoxon(std_rates, think_rates, alternative='two-sided')
    except ValueError as e:
        return {
            'n_matched': n,
            'n_std_worse': int(n_std_worse),
            'n_think_worse': int(n_think_worse),
            'n_tied': int(n_tied),
            'p_value': float('nan'),
            'statistic': float('nan'),
            'direction': f'Error: {e}',
            'r_rb': float('nan'),
            'mean_std': float(np.mean(std_rates)),
            'mean_think': float(np.mean(think_rates)),
        }

    # Rank-biserial correlation r_rb
    # r_rb = 1 - (2*T) / (n*(n+1)/2)  where n = number of non-zero diffs
    n_eff = len(nonzero_diffs)
    r_rb = 1 - (2 * stat) / (n_eff * (n_eff + 1) / 2)

    # Direction
    mean_std = float(np.mean(std_rates))
    mean_think = float(np.mean(think_rates))
    if mean_std < mean_think:
        direction = 'standard better (lower H-rate)'
    elif mean_std > mean_think:
        direction = 'thinking better (lower H-rate)'
    else:
        direction = 'identical means'

    return {
        'n_matched': n,
        'n_std_worse': int(n_std_worse),
        'n_think_worse': int(n_think_worse),
        'n_tied': int(n_tied),
        'p_value': float(p_val),
        'statistic': float(stat),
        'direction': direction,
        'r_rb': float(r_rb),
        'mean_std': mean_std,
        'mean_think': mean_think,
    }


def compute_expected_h(data):
    """Compute E[H per query] = total hallucinated tools / number of queries."""
    total_hallucinated = sum(v['hallucinated_count'] for v in data.values())
    n_queries = len(data)
    return total_hallucinated / n_queries if n_queries > 0 else 0.0, total_hallucinated, n_queries


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTATION 1: Wilcoxon signed-rank tests
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 100)
print("COMPUTATION 1: Paired Wilcoxon Signed-Rank Tests (Standard vs Thinking, C5)")
print("=" * 100)

for gen_label, std_config, think_config in [
    ("Gen1", GEN1_STANDARD, GEN1_THINKING),
    ("Gen2", GEN2_STANDARD, GEN2_THINKING),
]:
    print(f"\n{'─' * 100}")
    print(f"  {gen_label} — Standard vs Thinking (C5)")
    print(f"{'─' * 100}")

    # Collect all standard/thinking rates for cross-provider aggregate
    all_std_data = {}
    all_think_data = {}

    for provider in ['openai', 'anthropic', 'google']:
        phase_std, model_std, cat_std = std_config[provider]
        phase_think, model_think, cat_think = think_config[provider]

        display_std = MODEL_DISPLAY[(phase_std, model_std, cat_std)]
        display_think = MODEL_DISPLAY[(phase_think, model_think, cat_think)]

        std_data = load_results(phase_std, model_std, cat_std, 'C5')
        think_data = load_results(phase_think, model_think, cat_think, 'C5')

        # Add provider prefix to keys for aggregate
        for k, v in std_data.items():
            all_std_data[(provider, *k)] = v
        for k, v in think_data.items():
            all_think_data[(provider, *k)] = v

        result = compute_wilcoxon(std_data, think_data)

        print(f"\n  {provider.upper()}: {display_std} vs {display_think}")
        print(f"    Matched pairs:     {result['n_matched']}")
        print(f"    Mean H-rate (std): {result['mean_std']:.4f}  |  Mean H-rate (think): {result['mean_think']:.4f}")
        print(f"    Direction:         {result['direction']}")
        print(f"    Pairs std>think:   {result['n_std_worse']}  |  think>std: {result['n_think_worse']}  |  tied: {result['n_tied']}")
        print(f"    Wilcoxon T:        {result['statistic']:.1f}")
        print(f"    p-value:           {result['p_value']:.6f}  {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'n.s.'}")
        print(f"    Rank-biserial r:   {result['r_rb']:.4f}")

    # Cross-provider aggregate
    result_agg = compute_wilcoxon(all_std_data, all_think_data)
    print(f"\n  CROSS-PROVIDER AGGREGATE ({gen_label})")
    print(f"    Matched pairs:     {result_agg['n_matched']}")
    print(f"    Mean H-rate (std): {result_agg['mean_std']:.4f}  |  Mean H-rate (think): {result_agg['mean_think']:.4f}")
    print(f"    Direction:         {result_agg['direction']}")
    print(f"    Pairs std>think:   {result_agg['n_std_worse']}  |  think>std: {result_agg['n_think_worse']}  |  tied: {result_agg['n_tied']}")
    print(f"    Wilcoxon T:        {result_agg['statistic']:.1f}")
    print(f"    p-value:           {result_agg['p_value']:.6f}  {'***' if result_agg['p_value'] < 0.001 else '**' if result_agg['p_value'] < 0.01 else '*' if result_agg['p_value'] < 0.05 else 'n.s.'}")
    print(f"    Rank-biserial r:   {result_agg['r_rb']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTATION 2: E[H per query] for all 12 model configurations
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 100}")
print("COMPUTATION 2: E[H per query] — Expected Hallucinated Tools per Query")
print("=" * 100)

all_configs = [
    ("Gen1", "Standard", GEN1_STANDARD),
    ("Gen1", "Thinking", GEN1_THINKING),
    ("Gen2", "Standard", GEN2_STANDARD),
    ("Gen2", "Thinking", GEN2_THINKING),
]

for config_label in ['C5', 'C0']:
    print(f"\n{'─' * 100}")
    print(f"  Configuration: {config_label}")
    print(f"{'─' * 100}")
    print(f"  {'Generation':<8} {'Mode':<10} {'Provider':<12} {'Model':<32} {'E[H/query]':>12} {'Total H':>10} {'Queries':>10}")
    print(f"  {'─'*8:<8} {'─'*10:<10} {'─'*12:<12} {'─'*32:<32} {'─'*12:>12} {'─'*10:>10} {'─'*10:>10}")

    for gen_label, mode_label, config_dict in all_configs:
        mode_e_h_values = []
        for provider in ['openai', 'anthropic', 'google']:
            phase, model_id, cat = config_dict[provider]
            display = MODEL_DISPLAY[(phase, model_id, cat)]
            data = load_results(phase, model_id, cat, config_label)
            e_h, total_h, n_q = compute_expected_h(data)
            mode_e_h_values.append(e_h)
            print(f"  {gen_label:<8} {mode_label:<10} {provider:<12} {display:<32} {e_h:>12.4f} {total_h:>10d} {n_q:>10d}")

        avg_e_h = np.mean(mode_e_h_values)
        print(f"  {gen_label:<8} {mode_label:<10} {'---':<12} {'Cross-provider average':<32} {avg_e_h:>12.4f}")
        print()

# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL: Summary table of hallucination rates (%) for verification
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 100}")
print("VERIFICATION: Mean Hallucination Rates (%) under C5")
print("=" * 100)

print(f"\n  {'Generation':<8} {'Mode':<10} {'Provider':<12} {'Model':<32} {'H-Rate %':>10} {'Queries':>10}")
print(f"  {'─'*8:<8} {'─'*10:<10} {'─'*12:<12} {'─'*32:<32} {'─'*10:>10} {'─'*10:>10}")

for gen_label, mode_label, config_dict in all_configs:
    rates = []
    for provider in ['openai', 'anthropic', 'google']:
        phase, model_id, cat = config_dict[provider]
        display = MODEL_DISPLAY[(phase, model_id, cat)]
        data = load_results(phase, model_id, cat, 'C5')
        mean_rate = np.mean([v['h_rate'] for v in data.values()]) * 100
        rates.append(mean_rate)
        print(f"  {gen_label:<8} {mode_label:<10} {provider:<12} {display:<32} {mean_rate:>10.2f}% {len(data):>9d}")
    avg_rate = np.mean(rates)
    print(f"  {gen_label:<8} {mode_label:<10} {'---':<12} {'Cross-provider average':<32} {avg_rate:>10.2f}%")
    print()
