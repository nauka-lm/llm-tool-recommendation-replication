#!/usr/bin/env python3
"""
Compute Reviewer-Requested Metrics for Journal Paper Revision
=====================================================================
Computes all metrics flagged in the reviewer response:
  1. Bootstrap CIs (10,000 resamples, percentile) for all key HR values
  2. Cliff's delta for all major comparisons
  3. P_any (query-level hallucination probability)
  4. Domain appropriateness (% of grounded tools matching queried domain)
  5. Detection validation (sample 50 responses, check precision/recall)
  6. Tools-per-query distributions per configuration

Output: Both human-readable text and LaTeX-ready snippets for paper insertion.
"""

import json
import glob
import math
import os
import random
import re
from collections import defaultdict

# ===========================================================================
# Configuration
# ===========================================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "results")

BOOTSTRAP_N = 10000
RANDOM_SEED = 42
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reviewer_metrics_results.txt')

# Domain tool lists (ground truth from the CadcomOnline database)
# These are the tools that BELONG to each domain (primary + related)
# Extracted from the evaluation framework's domain configuration
DOMAIN_TOOLS = {
    'D1': {  # PCB Design Tool
        'primary': ['KiCad', 'EasyEDA', 'Eagle', 'Altium CircuitMaker', 'fritzing',
                     'gEDA', 'LibrePCB', 'Horizon EDA'],
        'related': ['PCB Calculator', 'Impedance Calculator', 'Saturn PCB Toolkit',
                     'Via Calculator', 'Thermal Calculator', 'Trace Width Calculator',
                     'Differential Pair Calculator', 'Stackup Planner', 'Signal Integrity Analyzer',
                     'BOM Manager', 'Gerber Viewer', 'Pick and Place Generator',
                     'DFM Checker', 'Panelization Tool', 'Assembly Drawing Generator',
                     'Test Point Optimizer', 'Footprint Creator', 'Symbol Editor',
                     'Design Rule Checker', '3D Viewer', 'Net Inspector']
    },
    'D2': {  # PCB Design Calculator
        'primary': ['PCB Calculator', 'Impedance Calculator', 'Saturn PCB Toolkit',
                     'Via Calculator', 'Thermal Calculator', 'Trace Width Calculator',
                     'Differential Pair Calculator', 'Stackup Planner', 'Signal Integrity Analyzer',
                     'BOM Manager', 'Gerber Viewer', 'Pick and Place Generator',
                     'DFM Checker', 'Panelization Tool', 'Assembly Drawing Generator'],
        'related': ['KiCad', 'EasyEDA', 'Eagle', 'Altium CircuitMaker', 'fritzing',
                     'gEDA', 'LibrePCB', 'Horizon EDA',
                     'Test Point Optimizer', 'Footprint Creator', 'Symbol Editor',
                     'Design Rule Checker', '3D Viewer', 'Net Inspector',
                     'Schematic Capture', 'Layout Editor', 'Autorouter',
                     'Component Library Manager', 'Version Control Integration',
                     'Collaboration Hub']
    },
    'D3': {  # SMPS/Converters
        'primary': ['SMPS Designer', 'Inductor Calculator', 'MOSFET Selector',
                     'Feedback Loop Analyzer', 'Capacitor ESR Calculator',
                     'Thermal Derating Tool', 'EMI Filter Designer', 'Snubber Calculator'],
        'related': ['Power Stage Simulator', 'Bode Plot Analyzer', 'Control Loop Designer',
                     'Component Stress Analyzer', 'Efficiency Calculator',
                     'Magnetics Designer', 'Gate Driver Calculator', 'PCB Layout Advisor']
    },
    'D4': {  # Transformers
        'primary': ['Transformer Designer', 'Core Loss Calculator', 'Winding Calculator',
                     'Bobbin Selector', 'Insulation Tester', 'Leakage Inductance Calculator',
                     'Thermal Model', 'Saturation Analyzer', 'Core Material Selector',
                     'Wire Gauge Calculator'],
        'related': ['Magnetics Simulator', 'B-H Curve Plotter', 'Proximity Effect Analyzer',
                     'Skin Effect Calculator', 'Safety Margin Calculator',
                     'Creepage Distance Tool', 'Hipot Test Planner',
                     'Transformer Equivalent Circuit', 'Frequency Response Analyzer',
                     'Core Geometry Optimizer', 'Turns Ratio Calculator',
                     'Regulation Calculator', 'Load Characterizer',
                     'Power Factor Analyzer', 'Harmonic Analyzer',
                     'Thermal Runaway Predictor', 'Demagnetization Checker',
                     'Inductance Calculator', 'Capacitance Calculator']
    }
}


# ===========================================================================
# Utility Functions
# ===========================================================================

def mean(data):
    return sum(data) / len(data) if data else 0

def std(data, ddof=1):
    if len(data) <= ddof:
        return 0
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / (len(data) - ddof))

def bootstrap_ci(data, n_boot=BOOTSTRAP_N, alpha=0.05, seed=RANDOM_SEED):
    """Bootstrap confidence interval using percentile method."""
    if len(data) < 2:
        m = mean(data)
        return m, m, m

    rng = random.Random(seed)
    n = len(data)
    obs_mean = mean(data)

    boot_means = []
    for _ in range(n_boot):
        sample = [data[rng.randint(0, n-1)] for _ in range(n)]
        boot_means.append(mean(sample))

    boot_means.sort()
    lo_idx = int(n_boot * (alpha / 2))
    hi_idx = int(n_boot * (1 - alpha / 2))

    return obs_mean, boot_means[lo_idx], boot_means[hi_idx]

def t_ci(data, alpha=0.05):
    """Parametric t-distribution CI for comparison."""
    n = len(data)
    if n < 2:
        m = mean(data)
        return m, m, m
    m = mean(data)
    se = std(data) / math.sqrt(n)
    # Use 1.96 for large n, approximate t for smaller
    t_crit = 1.96 if n > 120 else 1.977  # approx for n=144
    return m, m - t_crit * se, m + t_crit * se

def cliffs_delta(x, y):
    """Cliff's delta: nonparametric effect size.
    Returns delta in [-1, 1] where:
      |delta| < 0.147 = negligible
      |delta| < 0.33  = small
      |delta| < 0.474 = medium
      |delta| >= 0.474 = large
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0

    count = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                count += 1
            elif xi < yj:
                count -= 1

    return count / (nx * ny)

def cliffs_delta_label(d):
    d = abs(d)
    if d < 0.147: return "negligible"
    if d < 0.33: return "small"
    if d < 0.474: return "medium"
    return "large"

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)
    pooled_s = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_s == 0:
        return 0
    return (m1 - m2) / pooled_s


# ===========================================================================
# Data Loading
# ===========================================================================

def load_all_queries(base_dir=None):
    """Load ALL individual query results across all 4 phases."""
    if base_dir is None:
        base_dir = BASE_DIR
    all_queries = []

    for phase_name in ['phase1', 'phase2', 'phase3a', 'phase3b']:
        phase_dir = os.path.join(base_dir, phase_name)
        if not os.path.exists(phase_dir):
            print(f"  WARNING: {phase_dir} not found, trying alternate...")
            continue

        files = sorted(glob.glob(os.path.join(phase_dir, 'results_*.json')))
        for f in files:
            with open(f, encoding='utf-8') as fh:
                results = json.load(fh)
            for item in results:
                item['_phase'] = phase_name
                item['_file'] = os.path.basename(f)
                all_queries.append(item)

    return all_queries

def group_queries(queries):
    """Group queries by various dimensions."""
    groups = defaultdict(list)
    for q in queries:
        phase = q['_phase']
        provider = q.get('ProviderId', '?')
        config = q.get('ConfigId', '?')
        domain = q.get('DomainId', '?')
        category = q.get('ModelCategory', '?')

        # Per-query HR
        h = len(q.get('HallucinatedToolNames', []))
        g = len(q.get('GroundedToolNames', []))
        total = h + g
        hr = h / total if total > 0 else 0

        q['_hr'] = hr
        q['_h'] = h
        q['_g'] = g
        q['_total'] = total
        q['_has_halluc'] = 1 if h > 0 else 0

        groups[(phase, provider, config)].append(q)
        groups[(phase, provider, config, domain)].append(q)
        groups[(phase, category, config)].append(q)  # for mode-level grouping

    return groups


# ===========================================================================
# Metric 1: Bootstrap Confidence Intervals
# ===========================================================================

def compute_bootstrap_cis(groups, out):
    out.append("=" * 90)
    out.append("  METRIC 1: BOOTSTRAP 95% CIs vs PARAMETRIC CIs FOR KEY HR VALUES")
    out.append("=" * 90)
    out.append("")
    out.append(f"  Bootstrap: {BOOTSTRAP_N} resamples, percentile method, seed={RANDOM_SEED}")
    out.append("")

    header = f"  {'Phase':<12} {'Provider':<12} {'Config':<6} {'N':>4} {'Mean HR':>8} {'Bootstrap CI':>22} {'Parametric CI':>22} {'Match?':>7}"
    out.append(header)
    out.append(f"  {'-'*95}")

    latex_lines = []

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                      ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
        configs = ['C0', 'C5'] if 'phase2' in phase_name or 'phase3b' in phase_name else ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        for provider in ['openai', 'anthropic', 'google']:
            for config in ['C5']:  # Focus on C5 for paper
                key = (phase_name, provider, config)
                if key not in groups:
                    continue
                qs = groups[key]
                hrs = [q['_hr'] for q in qs]
                n = len(hrs)

                bm, blo, bhi = bootstrap_ci(hrs)
                tm, tlo, thi = t_ci(hrs)

                match = "yes" if abs(blo - tlo) < 0.005 and abs(bhi - thi) < 0.005 else "~close" if abs(blo - tlo) < 0.01 else "DIFFER"

                out.append(f"  {phase_label:<12} {provider:<12} {config:<6} {n:>4} {bm*100:>7.1f}% [{blo*100:>5.1f}%, {bhi*100:>5.1f}%] [{tlo*100:>5.1f}%, {thi*100:>5.1f}%] {match:>7}")
                latex_lines.append(f"  % {phase_label} {provider} {config}: {bm*100:.1f}\\% [{{bootstrap: {blo*100:.1f}, {bhi*100:.1f}}}]")

    out.append("")
    out.append("  --- LaTeX-ready CI values (bootstrap) ---")
    out.extend(latex_lines)

    # Also compute for cross-provider averages
    out.append("")
    out.append("  --- Cross-Provider Averages ---")
    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                      ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
        all_hrs = []
        for provider in ['openai', 'anthropic', 'google']:
            key = (phase_name, provider, 'C5')
            if key in groups:
                all_hrs.extend([q['_hr'] for q in groups[key]])
        if all_hrs:
            bm, blo, bhi = bootstrap_ci(all_hrs)
            out.append(f"  {phase_label} cross-provider C5: {bm*100:.1f}% [{blo*100:.1f}%, {bhi*100:.1f}%] (N={len(all_hrs)})")


# ===========================================================================
# Metric 2: Cliff's Delta for All Major Comparisons
# ===========================================================================

def compute_cliffs_delta(groups, out):
    out.append("")
    out.append("=" * 90)
    out.append("  METRIC 2: CLIFF'S DELTA (NONPARAMETRIC EFFECT SIZE)")
    out.append("=" * 90)
    out.append("")

    # 2a. C0 vs C5 per provider
    out.append("  --- 2a. C0 vs C5 (Architecture Effectiveness) ---")
    out.append(f"  {'Phase':<12} {'Provider':<12} {'C0 HR':>7} {'C5 HR':>7} {'Cliff d':>8} {'Effect':>12} {'Cohen d':>8}")
    out.append(f"  {'-'*72}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                      ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
        for provider in ['openai', 'anthropic', 'google']:
            c0_key = (phase_name, provider, 'C0')
            c5_key = (phase_name, provider, 'C5')
            if c0_key not in groups or c5_key not in groups:
                continue
            c0_hrs = [q['_hr'] for q in groups[c0_key]]
            c5_hrs = [q['_hr'] for q in groups[c5_key]]

            cd = cliffs_delta(c0_hrs, c5_hrs)
            cohensd = cohens_d(c0_hrs, c5_hrs)
            out.append(f"  {phase_label:<12} {provider:<12} {mean(c0_hrs)*100:>6.1f}% {mean(c5_hrs)*100:>6.1f}% {cd:>+7.3f} {cliffs_delta_label(cd):>12} {cohensd:>+7.2f}")

    # 2b. Standard vs Thinking (C5)
    out.append("")
    out.append("  --- 2b. Standard vs Thinking under C5 ---")
    out.append(f"  {'Gen':<6} {'Provider':<12} {'Std C5':>7} {'Thn C5':>7} {'Cliff d':>8} {'Effect':>12} {'Cohen d':>8}")
    out.append(f"  {'-'*66}")

    for gen_label, std_phase, thn_phase in [('Gen1', 'phase1', 'phase2'), ('Gen2', 'phase3a', 'phase3b')]:
        for provider in ['openai', 'anthropic', 'google']:
            std_key = (std_phase, provider, 'C5')
            thn_key = (thn_phase, provider, 'C5')
            if std_key not in groups or thn_key not in groups:
                continue
            std_hrs = [q['_hr'] for q in groups[std_key]]
            thn_hrs = [q['_hr'] for q in groups[thn_key]]

            cd = cliffs_delta(std_hrs, thn_hrs)
            cohensd = cohens_d(std_hrs, thn_hrs)
            out.append(f"  {gen_label:<6} {provider:<12} {mean(std_hrs)*100:>6.1f}% {mean(thn_hrs)*100:>6.1f}% {cd:>+7.3f} {cliffs_delta_label(cd):>12} {cohensd:>+7.2f}")

        # Cross-provider average
        all_std = []
        all_thn = []
        for provider in ['openai', 'anthropic', 'google']:
            std_key = (std_phase, provider, 'C5')
            thn_key = (thn_phase, provider, 'C5')
            if std_key in groups: all_std.extend([q['_hr'] for q in groups[std_key]])
            if thn_key in groups: all_thn.extend([q['_hr'] for q in groups[thn_key]])
        if all_std and all_thn:
            cd = cliffs_delta(all_std, all_thn)
            cohensd = cohens_d(all_std, all_thn)
            out.append(f"  {gen_label:<6} {'AVG':<12} {mean(all_std)*100:>6.1f}% {mean(all_thn)*100:>6.1f}% {cd:>+7.3f} {cliffs_delta_label(cd):>12} {cohensd:>+7.2f}")

    # 2c. C3 anomaly
    out.append("")
    out.append("  --- 2c. C3 Anomaly (C0 vs C3) ---")
    out.append(f"  {'Phase':<12} {'Provider':<12} {'C0 HR':>7} {'C3 HR':>7} {'Cliff d':>8} {'Effect':>12} {'Cohen d':>8}")
    out.append(f"  {'-'*72}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase3a', 'Gen2 Std')]:
        for provider in ['openai', 'anthropic', 'google']:
            c0_key = (phase_name, provider, 'C0')
            c3_key = (phase_name, provider, 'C3')
            if c0_key not in groups or c3_key not in groups:
                continue
            c0_hrs = [q['_hr'] for q in groups[c0_key]]
            c3_hrs = [q['_hr'] for q in groups[c3_key]]

            cd = cliffs_delta(c3_hrs, c0_hrs)  # C3 > C0 direction
            cohensd = cohens_d(c3_hrs, c0_hrs)
            out.append(f"  {phase_label:<12} {provider:<12} {mean(c0_hrs)*100:>6.1f}% {mean(c3_hrs)*100:>6.1f}% {cd:>+7.3f} {cliffs_delta_label(cd):>12} {cohensd:>+7.2f}")

    # 2d. Cross-generational (Gen1 C5 vs Gen2 C5)
    out.append("")
    out.append("  --- 2d. Cross-Generational (Gen1 C5 vs Gen2 C5) ---")
    out.append(f"  {'Mode':<10} {'Provider':<12} {'Gen1':>7} {'Gen2':>7} {'Cliff d':>8} {'Effect':>12} {'Cohen d':>8}")
    out.append(f"  {'-'*70}")

    for mode_label, g1_phase, g2_phase in [('Standard', 'phase1', 'phase3a'), ('Thinking', 'phase2', 'phase3b')]:
        for provider in ['openai', 'anthropic', 'google']:
            g1_key = (g1_phase, provider, 'C5')
            g2_key = (g2_phase, provider, 'C5')
            if g1_key not in groups or g2_key not in groups:
                continue
            g1_hrs = [q['_hr'] for q in groups[g1_key]]
            g2_hrs = [q['_hr'] for q in groups[g2_key]]

            cd = cliffs_delta(g1_hrs, g2_hrs)
            cohensd = cohens_d(g1_hrs, g2_hrs)
            out.append(f"  {mode_label:<10} {provider:<12} {mean(g1_hrs)*100:>6.1f}% {mean(g2_hrs)*100:>6.1f}% {cd:>+7.3f} {cliffs_delta_label(cd):>12} {cohensd:>+7.2f}")


# ===========================================================================
# Metric 3: P_any (Query-Level Hallucination Probability)
# ===========================================================================

def compute_p_any(groups, out):
    out.append("")
    out.append("=" * 90)
    out.append("  METRIC 3: P_any (QUERY-LEVEL HALLUCINATION PROBABILITY)")
    out.append("=" * 90)
    out.append("")
    out.append("  P_any = % of queries with at least one hallucinated tool")
    out.append("")

    # C5 results
    out.append("  --- Under C5 (Full Architecture) ---")
    out.append(f"  {'Phase':<12} {'Provider':<12} {'N':>4} {'P_any':>8} {'Mean HR':>8} {'Avg H/Q':>8} {'Avg T/Q':>8}")
    out.append(f"  {'-'*68}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                      ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
        for provider in ['openai', 'anthropic', 'google']:
            key = (phase_name, provider, 'C5')
            if key not in groups:
                continue
            qs = groups[key]
            n = len(qs)
            n_any = sum(1 for q in qs if q['_h'] > 0)
            p_any = n_any / n * 100 if n > 0 else 0
            mean_hr = mean([q['_hr'] for q in qs]) * 100
            avg_h_per_q = mean([q['_h'] for q in qs])
            avg_t_per_q = mean([q['_total'] for q in qs])

            out.append(f"  {phase_label:<12} {provider:<12} {n:>4} {p_any:>7.1f}% {mean_hr:>7.1f}% {avg_h_per_q:>7.2f} {avg_t_per_q:>7.1f}")

    # C0 results for comparison
    out.append("")
    out.append("  --- Under C0 (No Constraints) for comparison ---")
    out.append(f"  {'Phase':<12} {'Provider':<12} {'N':>4} {'P_any':>8} {'Mean HR':>8}")
    out.append(f"  {'-'*52}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase3a', 'Gen2 Std')]:
        for provider in ['openai', 'anthropic', 'google']:
            key = (phase_name, provider, 'C0')
            if key not in groups:
                continue
            qs = groups[key]
            n = len(qs)
            n_any = sum(1 for q in qs if q['_h'] > 0)
            p_any = n_any / n * 100 if n > 0 else 0
            mean_hr = mean([q['_hr'] for q in qs]) * 100

            out.append(f"  {phase_label:<12} {provider:<12} {n:>4} {p_any:>7.1f}% {mean_hr:>7.1f}%")

    # Cross-provider summary for paper
    out.append("")
    out.append("  --- SUMMARY FOR PAPER ---")
    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                      ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
        p_anys = []
        for provider in ['openai', 'anthropic', 'google']:
            key = (phase_name, provider, 'C5')
            if key not in groups:
                continue
            qs = groups[key]
            n = len(qs)
            n_any = sum(1 for q in qs if q['_h'] > 0)
            p_anys.append((provider, n_any / n * 100 if n > 0 else 0))
        if p_anys:
            min_p = min(p for _, p in p_anys)
            max_p = max(p for _, p in p_anys)
            min_prov = [prov for prov, p in p_anys if p == min_p][0]
            max_prov = [prov for prov, p in p_anys if p == max_p][0]
            out.append(f"  {phase_label} C5: P_any range = {min_p:.1f}% ({min_prov}) to {max_p:.1f}% ({max_prov})")


# ===========================================================================
# Metric 4: Domain Appropriateness
# ===========================================================================

def compute_domain_appropriateness(groups, out):
    out.append("")
    out.append("=" * 90)
    out.append("  METRIC 4: DOMAIN APPROPRIATENESS")
    out.append("=" * 90)
    out.append("")
    out.append("  % of grounded tool recommendations that match the queried domain")
    out.append("  (tool is in domain's primary or related tool list)")
    out.append("")

    def get_domain_tool_set(domain_id):
        """Get all valid tool names for a domain (primary + related), case-insensitive."""
        if domain_id not in DOMAIN_TOOLS:
            return set()
        dt = DOMAIN_TOOLS[domain_id]
        all_names = set()
        for name in dt.get('primary', []) + dt.get('related', []):
            all_names.add(name.lower())
        return all_names

    def check_tool_in_domain(tool_name, domain_id):
        """Check if a grounded tool belongs to the queried domain."""
        domain_tools = get_domain_tool_set(domain_id)
        tool_lower = tool_name.lower()
        # Exact match
        if tool_lower in domain_tools:
            return True
        # Containment match (same fuzzy logic as detector)
        for dt in domain_tools:
            if dt in tool_lower or tool_lower in dt:
                ratio = min(len(dt), len(tool_lower)) / max(len(dt), len(tool_lower))
                if ratio >= 0.55:
                    return True
        return False

    # Compute for C5 and C0
    for config_label, config_id in [('C5 (Full Architecture)', 'C5'), ('C0 (No Constraints)', 'C0')]:
        out.append(f"  --- Under {config_label} ---")
        out.append(f"  {'Phase':<12} {'Provider':<12} {'Grounded':>9} {'InDomain':>9} {'%Match':>8}")
        out.append(f"  {'-'*54}")

        total_grounded_all = 0
        total_in_domain_all = 0

        for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase2', 'Gen1 Thn'),
                                          ('phase3a', 'Gen2 Std'), ('phase3b', 'Gen2 Thn')]:
            for provider in ['openai', 'anthropic', 'google']:
                total_grounded = 0
                total_in_domain = 0

                for domain in ['D1', 'D2', 'D3', 'D4']:
                    key = (phase_name, provider, config_id, domain)
                    if key not in groups:
                        continue
                    for q in groups[key]:
                        grounded = q.get('GroundedToolNames', [])
                        for tool in grounded:
                            total_grounded += 1
                            if check_tool_in_domain(tool, domain):
                                total_in_domain += 1

                if total_grounded > 0:
                    pct = total_in_domain / total_grounded * 100
                    out.append(f"  {phase_label:<12} {provider:<12} {total_grounded:>9} {total_in_domain:>9} {pct:>7.1f}%")
                    total_grounded_all += total_grounded
                    total_in_domain_all += total_in_domain

        if total_grounded_all > 0:
            pct_all = total_in_domain_all / total_grounded_all * 100
            out.append(f"  {'TOTAL':<12} {'ALL':<12} {total_grounded_all:>9} {total_in_domain_all:>9} {pct_all:>7.1f}%")
        out.append("")


# ===========================================================================
# Metric 5: Detection Validation (50 Sample Check)
# ===========================================================================

def validate_detection(all_queries, out):
    out.append("")
    out.append("=" * 90)
    out.append("  METRIC 5: DETECTION VALIDATION (STRATIFIED SAMPLE)")
    out.append("=" * 90)
    out.append("")

    # Sample 25 C0 and 25 C5 responses, stratified by provider
    rng = random.Random(RANDOM_SEED)

    c0_queries = [q for q in all_queries if q.get('ConfigId') == 'C0' and q.get('_phase') in ('phase1', 'phase3a')]
    c5_queries = [q for q in all_queries if q.get('ConfigId') == 'C5' and q.get('_phase') in ('phase1', 'phase3a')]

    # Stratify: ~8-9 per provider for each config
    def stratified_sample(queries, n_per_provider=9):
        samples = []
        for provider in ['openai', 'anthropic', 'google']:
            prov_qs = [q for q in queries if q.get('ProviderId') == provider]
            if len(prov_qs) > n_per_provider:
                samples.extend(rng.sample(prov_qs, n_per_provider))
            else:
                samples.extend(prov_qs)
        return samples[:25]  # cap at 25

    c0_sample = stratified_sample(c0_queries)
    c5_sample = stratified_sample(c5_queries)

    out.append(f"  Sampled {len(c0_sample)} C0 + {len(c5_sample)} C5 = {len(c0_sample)+len(c5_sample)} responses")
    out.append(f"  Stratified across providers: ~{len(c0_sample)//3} per provider per config")
    out.append("")

    # For each sample, analyze the raw response for tool name extraction quality
    # We can do automated validation by checking:
    # 1. Known software names in response that were NOT extracted (false negatives)
    # 2. Extracted names that are clearly not tools (false positives)

    KNOWN_COMMERCIAL_TOOLS = {
        'altium designer', 'orcad', 'cadence', 'mentor graphics', 'zuken',
        'proteus', 'multisim', 'ltspice', 'pspice', 'fusion 360',
        'solidworks', 'autocad', 'ansys', 'comsol', 'matlab', 'simulink',
        'kicad', 'easyeda', 'eagle', 'fritzing', 'diptrace',
        'ni multisim', 'tina-ti', 'simetrix', 'micro-cap',
        'pcb artist', 'sprint-layout', 'target 3001', 'designspark',
        'circuitmaker', 'upverter', 'pad2pad', 'zentipcb',
        'expresspcb', 'batchpcb', 'pcbway', 'jlcpcb', 'oshpark',
    }

    total_mentioned_all = 0
    total_extracted = 0
    false_positives = 0
    missed_tools = 0

    for config_label, samples in [('C0', c0_sample), ('C5', c5_sample)]:
        config_extracted = 0
        config_fp = 0
        config_fn = 0

        for q in samples:
            raw = q.get('RawResponse', '')
            mentioned = set(q.get('MentionedToolNames', []))
            hallucinated = set(q.get('HallucinatedToolNames', []))
            grounded = set(q.get('GroundedToolNames', []))

            config_extracted += len(mentioned)

            # Check for potential false negatives: known tools in response not in MentionedToolNames
            raw_lower = raw.lower()
            for tool in KNOWN_COMMERCIAL_TOOLS:
                if tool in raw_lower:
                    # Check if any variant was captured
                    captured = False
                    for m in mentioned:
                        if tool in m.lower() or m.lower() in tool:
                            captured = True
                            break
                    if not captured:
                        # Possible false negative - but only if it looks like a tool recommendation
                        # Skip if it's just mentioned in passing (e.g., "unlike Altium Designer...")
                        config_fn += 1

            # Check for potential false positives in hallucinated set
            # These would be words that were extracted as tool names but are actually
            # common English words or non-tool references
            COMMON_WORDS = {'design', 'view', 'edit', 'properties', 'tool', 'system',
                          'analysis', 'simulation', 'test', 'check', 'model', 'layout',
                          'board', 'circuit', 'power', 'signal', 'component'}
            for h in hallucinated:
                if h.lower() in COMMON_WORDS and len(h.split()) == 1:
                    config_fp += 1

        total_extracted += config_extracted
        false_positives += config_fp
        missed_tools += config_fn

        n_samples = len(samples)
        out.append(f"  {config_label}: {n_samples} responses, {config_extracted} tool mentions extracted")
        out.append(f"    Potential false positives (common words as tools): {config_fp}")
        out.append(f"    Potential false negatives (known tools not captured): {config_fn}")

    out.append("")
    # Compute precision and recall estimates
    # Precision = (extracted - false positives) / extracted
    # Recall = (extracted - false positives) / (extracted - false positives + missed)
    if total_extracted > 0:
        precision = (total_extracted - false_positives) / total_extracted * 100
        true_positives = total_extracted - false_positives
        recall = true_positives / (true_positives + missed_tools) * 100 if (true_positives + missed_tools) > 0 else 100

        out.append(f"  ESTIMATED DETECTION QUALITY (automated check):")
        out.append(f"    Total tool mentions extracted: {total_extracted}")
        out.append(f"    Estimated false positives: {false_positives}")
        out.append(f"    Estimated missed (false neg): {missed_tools}")
        out.append(f"    Estimated precision: {precision:.1f}%")
        out.append(f"    Estimated recall: {recall:.1f}%")
        out.append("")
        out.append(f"  NOTE: This is an AUTOMATED estimate. For the paper, a manual review")
        out.append(f"  of these {len(c0_sample)+len(c5_sample)} samples is recommended to verify.")

    # Print a few examples for manual inspection
    out.append("")
    out.append("  --- SAMPLE RESPONSES FOR MANUAL INSPECTION ---")
    out.append("  (First 3 C0 and 3 C5 responses with hallucinated tools)")
    out.append("")

    shown = 0
    for q in c0_sample:
        if q.get('HallucinatedToolNames') and shown < 3:
            out.append(f"  [{q['_phase']}] {q['ProviderId']} C0 {q['DomainId']} P{q['PromptIndex']+1} R{q['Repetition']}")
            out.append(f"    Mentioned: {q.get('MentionedToolNames', [])}")
            out.append(f"    Grounded:  {q.get('GroundedToolNames', [])}")
            out.append(f"    Halluc:    {q.get('HallucinatedToolNames', [])}")
            # Show first 300 chars of response
            resp = q.get('RawResponse', '')[:300].replace('\n', ' ')
            out.append(f"    Response:  {resp}...")
            out.append("")
            shown += 1

    shown = 0
    for q in c5_sample:
        if q.get('HallucinatedToolNames') and shown < 3:
            out.append(f"  [{q['_phase']}] {q['ProviderId']} C5 {q['DomainId']} P{q['PromptIndex']+1} R{q['Repetition']}")
            out.append(f"    Mentioned: {q.get('MentionedToolNames', [])}")
            out.append(f"    Grounded:  {q.get('GroundedToolNames', [])}")
            out.append(f"    Halluc:    {q.get('HallucinatedToolNames', [])}")
            resp = q.get('RawResponse', '')[:300].replace('\n', ' ')
            out.append(f"    Response:  {resp}...")
            out.append("")
            shown += 1


# ===========================================================================
# Metric 6: Tools-Per-Query Distributions
# ===========================================================================

def compute_tools_per_query(groups, out):
    out.append("")
    out.append("=" * 90)
    out.append("  METRIC 6: TOOLS-PER-QUERY DISTRIBUTIONS BY CONFIG")
    out.append("=" * 90)
    out.append("")
    out.append(f"  {'Phase':<12} {'Provider':<12} {'Config':<6} {'Mean T/Q':>9} {'Std':>7} {'Min':>5} {'Max':>5} {'Mean H/Q':>9}")
    out.append(f"  {'-'*72}")

    for phase_name, phase_label in [('phase1', 'Gen1 Std'), ('phase3a', 'Gen2 Std')]:
        for provider in ['openai', 'anthropic', 'google']:
            for config in ['C0', 'C2', 'C4', 'C5']:
                key = (phase_name, provider, config)
                if key not in groups:
                    continue
                qs = groups[key]
                t_per_q = [q['_total'] for q in qs]
                h_per_q = [q['_h'] for q in qs]

                out.append(f"  {phase_label:<12} {provider:<12} {config:<6} {mean(t_per_q):>8.1f} {std(t_per_q):>6.1f} {min(t_per_q):>5} {max(t_per_q):>5} {mean(h_per_q):>8.2f}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    results_dir = BASE_DIR
    print(f"Loading data from: {results_dir}")
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found at {results_dir}")
        print("Trying alternate paths...")
        alt_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results"),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                results_dir = p
                print(f"Found at: {results_dir}")
                break
        else:
            print("Could not find Results directory!")
            return

    all_queries = load_all_queries(results_dir)
    print(f"Loaded {len(all_queries)} total queries")

    # Verify counts
    phase_counts = defaultdict(int)
    for q in all_queries:
        phase_counts[q['_phase']] += 1
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count} queries")

    groups = group_queries(all_queries)
    print(f"Grouped into {len(groups)} dimension combinations\n")

    out = []
    out.append("=" * 90)
    out.append("  REVIEWER-REQUESTED METRICS FOR PAPER REVISION")
    out.append("  Generated by compute_reviewer_metrics.py")
    out.append("=" * 90)
    out.append("")

    compute_bootstrap_cis(groups, out)
    compute_cliffs_delta(groups, out)
    compute_p_any(groups, out)
    compute_domain_appropriateness(groups, out)
    validate_detection(all_queries, out)
    compute_tools_per_query(groups, out)

    # Write output
    output_text = '\n'.join(out)
    print(output_text)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
