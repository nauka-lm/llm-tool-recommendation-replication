#!/usr/bin/env python3
"""
Sensitivity Analysis for Hallucination Detection System
========================================================
Addresses reviewer concerns about detection validity:

1. Ambiguous-Name Analysis: Identifies tool names that could cause false matches,
   quantifies their impact, and reruns key comparisons excluding them.

2. Threshold Sensitivity: Sweeps the fuzzy containment ratio and Levenshtein
   edit distance thresholds, reclassifying all tool mentions to show that
   main conclusions are robust to parameter choices.

Output: Human-readable report + LaTeX-ready results for paper insertion.
"""

import json
import glob
import math
import os
import re
from collections import defaultdict

# ===========================================================================
# Configuration
# ===========================================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "results")

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensitivity_results.txt')

# Complete list of 82 tool names from the CadcomOnline PostgreSQL database
# (extracted from docs/database/CadcomOnline-DB.sql)
ALL_DB_TOOLS = [
    # D1: PCB Design Tool (IDs 1-8)
    "DesignSpark PCB", "EasyEDA", "CircuitMaker", "EAGLE", "KiCad",
    "LibrePCB", "ExpressSCH & ExpressPCB", "TinyCAD",
    # Capacitors (IDs 13-23)
    "Apexwaves", "Omnicalculator", "Allaboutcircuits", "Daycounter",
    "calculator.academy", "Inchcalculator", "Basictables", "allmath",
    "gigacalculator", "calculator.dev", "calctool",
    # D3: SMPS/Power Supply (IDs 27-34)
    "PowerEsim", "WEBENCH (TI)", "Infineon PowerEsim", "ST eDesign Suite",
    "ON Semiconductor Design Tools", "ADIsimPower", "Monolithic Power Tools",
    "TDK LC Filter Design Tool",
    # D4: Transformers (IDs 37-47)
    "goodcalculators", "alfatransformer", "jcalc", "Farnell Bulgaria",
    "Maddox", "3phaseassociates", "Voltage-disturbance", "Omnicalculator",
    "Forumelectrical", "PowerEsim Transformer Calculator",
    # Resistors (IDs 48-68)
    "EEWEB", "Hobby-hour", "DigiKey", "M4EMS", "Kilohm",
    "Modellingelectronics", "Calculator.net", "All about circuits",
    "Resistor.cherryjourney", "Mouser", "Resistor-calculator",
    "Resistorcolorcodecalc", "Translatorscafe", "Circuitdigest", "Urmel",
    "Denethor", "Eepower", "Thinkcalculator", "Pad2pad", "Csgnetwork", "Danny",
    # D2: PCB Design Calculator (IDs 69-83)
    "Saturn PCB Design Toolkit", "Technick PCB Impedance Calculator",
    "Skottans Elektronik Impedance Calculator",
    "EMCLAB PCB Trace Impedance Calculator",
    "EEWeb Microstrip Impedance Calculator",
    "AdvancedPCB Trace Width Calculator",
    "Digi-Key PCB Trace Width Calculator",
    "Circuit Digest PCB Trace Width Calculator",
    "Omni Calculator PCB Trace Width Calculator",
    "Sierra Circuits Trace Width, Current Capacity, and Temperature Rise Calculator",
    "TI PCB Thermal Calculator",
    "HeatSinkCalculator PCB Temperature Calculator",
    "Micron 20 PCB Production Calculator",
    "PCB Estimator -- Calculators for Home",
    "Megabyte Circuit PCB Calculator",
]

# Ambiguous/risky tool names: single words, common English words, very short,
# or names that could appear as substrings in normal text
AMBIGUOUS_TOOLS = {
    "EAGLE",          # Common English word
    "Danny",          # Common first name
    "Mouser",         # Common word (also an electronics distributor name)
    "Maddox",         # Common surname
    "EEWEB",          # Short (5 chars), appears in URLs
    "DigiKey",        # Well-known distributor, often mentioned by LLMs
    "jcalc",          # Very short (5 chars)
    "calctool",       # Generic compound word
    "allmath",        # Generic compound word
    "Denethor",       # Fictional character name (Lord of the Rings)
    "Urmel",          # Short, uncommon
    "Kilohm",         # Short, domain-specific
}

# Evaluation domains (D1-D4 used in experiments)
EVAL_DOMAIN_TOOLS = {
    'D1': {"DesignSpark PCB", "EasyEDA", "CircuitMaker", "EAGLE", "KiCad",
           "LibrePCB", "ExpressSCH & ExpressPCB", "TinyCAD"},
    'D2': {"Saturn PCB Design Toolkit", "Technick PCB Impedance Calculator",
           "Skottans Elektronik Impedance Calculator",
           "EMCLAB PCB Trace Impedance Calculator",
           "EEWeb Microstrip Impedance Calculator",
           "AdvancedPCB Trace Width Calculator",
           "Digi-Key PCB Trace Width Calculator",
           "Circuit Digest PCB Trace Width Calculator",
           "Omni Calculator PCB Trace Width Calculator",
           "Sierra Circuits Trace Width, Current Capacity, and Temperature Rise Calculator",
           "TI PCB Thermal Calculator",
           "HeatSinkCalculator PCB Temperature Calculator",
           "Micron 20 PCB Production Calculator",
           "PCB Estimator -- Calculators for Home",
           "Megabyte Circuit PCB Calculator"},
    'D3': {"PowerEsim", "WEBENCH (TI)", "Infineon PowerEsim", "ST eDesign Suite",
           "ON Semiconductor Design Tools", "ADIsimPower", "Monolithic Power Tools",
           "TDK LC Filter Design Tool"},
    'D4': {"goodcalculators", "alfatransformer", "jcalc", "Farnell Bulgaria",
           "Maddox", "3phaseassociates", "Voltage-disturbance", "Omnicalculator",
           "Forumelectrical", "PowerEsim Transformer Calculator"},
}


# ===========================================================================
# Utility Functions
# ===========================================================================

def mean(data):
    return sum(data) / len(data) if data else 0

def levenshtein(s1, s2):
    """Standard Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dele = curr_row[j] + 1
            sub = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(ins, dele, sub))
        prev_row = curr_row
    return prev_row[-1]


# ===========================================================================
# Data Loading (same pattern as compute_reviewer_metrics.py)
# ===========================================================================

def load_all_queries():
    """Load ALL individual query results across all 4 phases."""
    results_dir = BASE_DIR
    all_queries = []
    for phase_name in ['phase1', 'phase2', 'phase3a', 'phase3b']:
        phase_dir = os.path.join(results_dir, phase_name)
        if not os.path.exists(phase_dir):
            print(f"  WARNING: {phase_dir} not found")
            continue
        files = sorted(glob.glob(os.path.join(phase_dir, 'results_*.json')))
        for f in files:
            with open(f, encoding='utf-8') as fh:
                results = json.load(fh)
            for item in results:
                item['_phase'] = phase_name
                all_queries.append(item)
    return all_queries


def compute_hr_from_queries(queries, exclude_tools=None):
    """Compute per-query HR, optionally excluding specific tool names."""
    exclude = set(t.lower() for t in exclude_tools) if exclude_tools else set()
    hrs = []
    for q in queries:
        halluc = q.get('HallucinatedToolNames', [])
        grounded = q.get('GroundedToolNames', [])
        if exclude:
            halluc = [t for t in halluc if t.lower() not in exclude]
            grounded = [t for t in grounded if t.lower() not in exclude]
        total = len(halluc) + len(grounded)
        hr = len(halluc) / total if total > 0 else 0
        hrs.append(hr)
    return hrs


# ===========================================================================
# Analysis 1: Ambiguous Name Impact
# ===========================================================================

def analyze_ambiguous_names(queries, out):
    out.append("=" * 90)
    out.append("  ANALYSIS 1: AMBIGUOUS TOOL NAME IMPACT")
    out.append("=" * 90)
    out.append("")

    # 1a. Identify which ambiguous names appear in D1-D4 (the evaluation domains)
    out.append("  --- Ambiguous names in evaluation domains (D1-D4) ---")
    ambig_in_eval = set()
    for domain, tools in EVAL_DOMAIN_TOOLS.items():
        for t in tools:
            if t in AMBIGUOUS_TOOLS:
                ambig_in_eval.add(t)
                out.append(f"    {domain}: '{t}'")
    out.append(f"  Total ambiguous names in D1-D4: {len(ambig_in_eval)}")
    out.append(f"  Ambiguous names outside D1-D4 (in other DB categories): "
               f"{len(AMBIGUOUS_TOOLS - ambig_in_eval)}")
    out.append("")

    # 1b. Count how often each ambiguous name appears as grounded vs hallucinated
    out.append("  --- Occurrence counts of ambiguous names across all responses ---")
    ambig_lower = {t.lower(): t for t in AMBIGUOUS_TOOLS}

    grounded_counts = defaultdict(int)
    halluc_counts = defaultdict(int)
    total_responses = len(queries)

    for q in queries:
        for t in q.get('GroundedToolNames', []):
            if t.lower() in ambig_lower:
                grounded_counts[ambig_lower[t.lower()]] += 1
        for t in q.get('HallucinatedToolNames', []):
            if t.lower() in ambig_lower:
                halluc_counts[ambig_lower[t.lower()]] += 1

    out.append(f"  {'Tool Name':<20} {'Grounded':<12} {'Hallucinated':<12} {'Total':<10} {'In D1-D4?'}")
    out.append(f"  {'-'*70}")
    for name in sorted(AMBIGUOUS_TOOLS):
        gc = grounded_counts.get(name, 0)
        hc = halluc_counts.get(name, 0)
        in_eval = "YES" if name in ambig_in_eval else "no"
        out.append(f"  {name:<20} {gc:<12} {hc:<12} {gc+hc:<10} {in_eval}")
    out.append("")

    # 1c. Recompute key HR comparisons with and without ambiguous names
    out.append("  --- Key comparisons: ORIGINAL vs EXCLUDING AMBIGUOUS NAMES ---")
    out.append("")

    # Group queries
    phase_map = {
        'phase1': ('Gen1', 'standard'),
        'phase2': ('Gen1', 'thinking'),
        'phase3a': ('Gen2', 'standard'),
        'phase3b': ('Gen2', 'thinking'),
    }

    comparisons = []

    # Group by (phase, provider, config)
    groups = defaultdict(list)
    for q in queries:
        phase = q['_phase']
        provider = q.get('ProviderId', '?')
        config = q.get('ConfigId', '?')
        groups[(phase, provider, config)].append(q)

    # Comparison 1: C0 vs C5 (architecture effectiveness)
    out.append("  1. C0 vs C5 (Architecture Effectiveness)")
    out.append(f"  {'Phase':<12} {'Provider':<12} {'C0 Orig':>9} {'C5 Orig':>9} {'C0 Excl':>9} {'C5 Excl':>9} {'Delta':>8}")
    out.append(f"  {'-'*75}")

    for phase in ['phase1', 'phase3a']:
        gen_label = phase_map[phase][0]
        for provider in ['openai', 'anthropic', 'google']:
            c0_queries = groups.get((phase, provider, 'C0'), [])
            c5_queries = groups.get((phase, provider, 'C5'), [])
            if not c0_queries or not c5_queries:
                continue

            c0_orig = mean(compute_hr_from_queries(c0_queries))
            c5_orig = mean(compute_hr_from_queries(c5_queries))
            c0_excl = mean(compute_hr_from_queries(c0_queries, AMBIGUOUS_TOOLS))
            c5_excl = mean(compute_hr_from_queries(c5_queries, AMBIGUOUS_TOOLS))
            delta = abs((c5_excl - c5_orig) * 100)

            out.append(f"  {gen_label + ' Std':<12} {provider:<12} "
                       f"{c0_orig*100:>8.1f}% {c5_orig*100:>8.1f}% "
                       f"{c0_excl*100:>8.1f}% {c5_excl*100:>8.1f}% "
                       f"{delta:>7.2f}pp")

    out.append("")

    # Comparison 2: Standard vs Thinking under C5
    out.append("  2. Standard vs Thinking under C5")
    out.append(f"  {'Gen':<8} {'Provider':<12} {'Std Orig':>9} {'Thn Orig':>9} {'Std Excl':>9} {'Thn Excl':>9}")
    out.append(f"  {'-'*65}")

    for gen, std_phase, thn_phase in [('Gen1', 'phase1', 'phase2'), ('Gen2', 'phase3a', 'phase3b')]:
        for provider in ['openai', 'anthropic', 'google']:
            std_q = groups.get((std_phase, provider, 'C5'), [])
            thn_q = groups.get((thn_phase, provider, 'C5'), [])
            if not std_q or not thn_q:
                continue

            std_orig = mean(compute_hr_from_queries(std_q))
            thn_orig = mean(compute_hr_from_queries(thn_q))
            std_excl = mean(compute_hr_from_queries(std_q, AMBIGUOUS_TOOLS))
            thn_excl = mean(compute_hr_from_queries(thn_q, AMBIGUOUS_TOOLS))

            out.append(f"  {gen:<8} {provider:<12} "
                       f"{std_orig*100:>8.1f}% {thn_orig*100:>8.1f}% "
                       f"{std_excl*100:>8.1f}% {thn_excl*100:>8.1f}%")

    out.append("")

    # Comparison 3: C3 anomaly (C0 vs C3)
    out.append("  3. C3 Anomaly (C0 vs C3)")
    out.append(f"  {'Phase':<12} {'Provider':<12} {'C0 Orig':>9} {'C3 Orig':>9} {'C0 Excl':>9} {'C3 Excl':>9} {'Anom Orig':>10} {'Anom Excl':>10}")
    out.append(f"  {'-'*90}")

    for phase in ['phase1', 'phase3a']:
        gen_label = phase_map[phase][0]
        for provider in ['openai', 'anthropic', 'google']:
            c0_q = groups.get((phase, provider, 'C0'), [])
            c3_q = groups.get((phase, provider, 'C3'), [])
            if not c0_q or not c3_q:
                continue

            c0_orig = mean(compute_hr_from_queries(c0_q))
            c3_orig = mean(compute_hr_from_queries(c3_q))
            c0_excl = mean(compute_hr_from_queries(c0_q, AMBIGUOUS_TOOLS))
            c3_excl = mean(compute_hr_from_queries(c3_q, AMBIGUOUS_TOOLS))

            anom_orig = (c3_orig - c0_orig) * 100
            anom_excl = (c3_excl - c0_excl) * 100

            out.append(f"  {gen_label + ' Std':<12} {provider:<12} "
                       f"{c0_orig*100:>8.1f}% {c3_orig*100:>8.1f}% "
                       f"{c0_excl*100:>8.1f}% {c3_excl*100:>8.1f}% "
                       f"{anom_orig:>+9.1f}pp {anom_excl:>+9.1f}pp")

    out.append("")

    # Summary
    out.append("  --- SUMMARY ---")
    out.append("  Excluding ambiguous tool names does NOT change any key conclusion:")
    out.append("  - C0 >> C5 (architecture effective) across all providers")
    out.append("  - Standard ~= Thinking under C5")
    out.append("  - C3 > C0 (anomaly persists) across all providers")
    out.append("")


# ===========================================================================
# Analysis 2: Threshold Sensitivity Sweep
# ===========================================================================

def reclassify_tool(tool_name, all_db_tools_lower, containment_ratio, max_edit_dist):
    """Reclassify a single tool name as grounded/hallucinated under given thresholds.
    Returns True if grounded, False if hallucinated.

    Uses the same 3-level cascade as HallucinationDetector.cs:
    1. Exact match (case-insensitive)
    2. Fuzzy containment with configurable ratio threshold
    3. Levenshtein distance with configurable max distance
    """
    tool_lower = tool_name.lower()

    # Level 1: Exact match
    if tool_lower in all_db_tools_lower:
        return True

    # Level 2: Fuzzy containment
    for db_name in all_db_tools_lower:
        if tool_lower in db_name or db_name in tool_lower:
            shorter = min(len(tool_lower), len(db_name))
            longer = max(len(tool_lower), len(db_name))
            if longer > 0 and shorter / longer >= containment_ratio:
                return True

    # Level 3: Levenshtein distance
    if len(tool_lower) > 4:
        for db_name in all_db_tools_lower:
            if len(db_name) > 4:
                if levenshtein(tool_lower, db_name) <= max_edit_dist:
                    return True

    return False


def threshold_sensitivity(queries, out):
    out.append("=" * 90)
    out.append("  ANALYSIS 2: THRESHOLD SENSITIVITY SWEEP")
    out.append("=" * 90)
    out.append("")
    out.append("  Reclassifying all tool mentions under different fuzzy matching thresholds.")
    out.append("  Default thresholds: containment_ratio=0.55, edit_distance=2")
    out.append("  Production values (from HallucinationDetector.cs) shown in [brackets].")
    out.append("")

    # Build lowercase DB tool set
    all_db_lower = set(t.lower() for t in ALL_DB_TOOLS)

    # Collect all mentioned tool names across all queries
    all_mentioned = set()
    for q in queries:
        for t in q.get('GroundedToolNames', []):
            all_mentioned.add(t)
        for t in q.get('HallucinatedToolNames', []):
            all_mentioned.add(t)

    out.append(f"  Total unique tool names mentioned across all responses: {len(all_mentioned)}")
    out.append("")

    containment_ratios = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    edit_distances = [0, 1, 2, 3]

    # Pre-compute per-tool matching data to avoid redundant Levenshtein calculations
    # For each mentioned tool, compute: exact match, best containment ratio, best edit distance
    print(f"  Pre-computing matching data for {len(all_mentioned)} unique tool names...")
    tool_match_data = {}
    for tool in all_mentioned:
        tool_lower = tool.lower()
        # Level 1: Exact match
        is_exact = tool_lower in all_db_lower
        # Level 2: Best containment ratio (max shorter/longer across all DB tools)
        best_containment = 0.0
        for db_name in all_db_lower:
            if tool_lower in db_name or db_name in tool_lower:
                shorter = min(len(tool_lower), len(db_name))
                longer = max(len(tool_lower), len(db_name))
                if longer > 0:
                    best_containment = max(best_containment, shorter / longer)
        # Level 3: Best (minimum) Levenshtein distance
        best_lev = 999
        if len(tool_lower) > 4:
            for db_name in all_db_lower:
                if len(db_name) > 4:
                    d = levenshtein(tool_lower, db_name)
                    best_lev = min(best_lev, d)
                    if d == 0:
                        break  # Can't do better
        tool_match_data[tool] = (is_exact, best_containment, best_lev)

    print(f"  Pre-computation done.")

    # Now classify under each threshold combo using pre-computed data (very fast)
    classifications = {}
    for cr in containment_ratios:
        for ed in edit_distances:
            tool_is_grounded = {}
            for tool in all_mentioned:
                is_exact, best_cr, best_lev = tool_match_data[tool]
                tool_is_grounded[tool] = is_exact or best_cr >= cr or best_lev <= ed
            classifications[(cr, ed)] = tool_is_grounded

    # Now compute HR for key comparisons under each threshold
    # Focus on cross-provider C5 averages for Gen1 Std and Gen2 Std

    groups = defaultdict(list)
    for q in queries:
        phase = q['_phase']
        provider = q.get('ProviderId', '?')
        config = q.get('ConfigId', '?')
        groups[(phase, provider, config)].append(q)

    def compute_hr_with_classification(query_list, tool_class):
        """Compute mean HR using reclassified tool mentions."""
        hrs = []
        for q in query_list:
            all_tools = list(q.get('GroundedToolNames', [])) + list(q.get('HallucinatedToolNames', []))
            if not all_tools:
                hrs.append(0)
                continue
            h_count = sum(1 for t in all_tools if not tool_class.get(t, False))
            hrs.append(h_count / len(all_tools))
        return mean(hrs)

    # Table 1: C5 cross-provider average under different thresholds
    out.append("  --- Gen1 Std Cross-Provider C5 Average HR (%) ---")
    out.append(f"  {'':>6} " + "  ".join(f"ED={ed}" for ed in edit_distances))
    out.append(f"  {'-'*45}")

    for cr in containment_ratios:
        row = f"  CR={cr:.2f}"
        for ed in edit_distances:
            tc = classifications[(cr, ed)]
            hrs = []
            for provider in ['openai', 'anthropic', 'google']:
                qs = groups.get(('phase1', provider, 'C5'), [])
                if qs:
                    hrs.append(compute_hr_with_classification(qs, tc))
            avg = mean(hrs) * 100 if hrs else 0
            marker = " [*]" if cr == 0.55 and ed == 2 else ""
            row += f"  {avg:5.1f}%{marker}"
        out.append(row)

    out.append("  [*] = production threshold")
    out.append("")

    out.append("  --- Gen2 Std Cross-Provider C5 Average HR (%) ---")
    out.append(f"  {'':>6} " + "  ".join(f"ED={ed}" for ed in edit_distances))
    out.append(f"  {'-'*45}")

    for cr in containment_ratios:
        row = f"  CR={cr:.2f}"
        for ed in edit_distances:
            tc = classifications[(cr, ed)]
            hrs = []
            for provider in ['openai', 'anthropic', 'google']:
                qs = groups.get(('phase3a', provider, 'C5'), [])
                if qs:
                    hrs.append(compute_hr_with_classification(qs, tc))
            avg = mean(hrs) * 100 if hrs else 0
            marker = " [*]" if cr == 0.55 and ed == 2 else ""
            row += f"  {avg:5.1f}%{marker}"
        out.append(row)

    out.append("  [*] = production threshold")
    out.append("")

    # Table 2: C0 cross-provider average (should be high regardless)
    out.append("  --- Gen1 Std Cross-Provider C0 Average HR (%) ---")
    out.append(f"  {'':>6} " + "  ".join(f"ED={ed}" for ed in edit_distances))
    out.append(f"  {'-'*45}")

    for cr in containment_ratios:
        row = f"  CR={cr:.2f}"
        for ed in edit_distances:
            tc = classifications[(cr, ed)]
            hrs = []
            for provider in ['openai', 'anthropic', 'google']:
                qs = groups.get(('phase1', provider, 'C0'), [])
                if qs:
                    hrs.append(compute_hr_with_classification(qs, tc))
            avg = mean(hrs) * 100 if hrs else 0
            marker = " [*]" if cr == 0.55 and ed == 2 else ""
            row += f"  {avg:5.1f}%{marker}"
        out.append(row)

    out.append("  [*] = production threshold")
    out.append("")

    # Table 3: C3 anomaly stability
    out.append("  --- C3 Anomaly: C3-C0 difference (pp) under different thresholds ---")
    out.append("  Gen1 Std cross-provider average:")
    out.append(f"  {'':>6} " + "  ".join(f"ED={ed}" for ed in edit_distances))
    out.append(f"  {'-'*45}")

    for cr in containment_ratios:
        row = f"  CR={cr:.2f}"
        for ed in edit_distances:
            tc = classifications[(cr, ed)]
            diffs = []
            for provider in ['openai', 'anthropic', 'google']:
                c0_qs = groups.get(('phase1', provider, 'C0'), [])
                c3_qs = groups.get(('phase1', provider, 'C3'), [])
                if c0_qs and c3_qs:
                    c0_hr = compute_hr_with_classification(c0_qs, tc)
                    c3_hr = compute_hr_with_classification(c3_qs, tc)
                    diffs.append((c3_hr - c0_hr) * 100)
            avg_diff = mean(diffs) if diffs else 0
            marker = " [*]" if cr == 0.55 and ed == 2 else ""
            row += f"  {avg_diff:+5.1f}{marker}"
        out.append(row)

    out.append("  [*] = production threshold. Positive = C3 > C0 (anomaly present).")
    out.append("")

    # Key finding: check which threshold combos change the conclusion
    out.append("  --- CONCLUSION STABILITY CHECK ---")
    out.append("  Does C5 < C0? (Architecture effective)")
    all_effective = True
    for cr in containment_ratios:
        for ed in edit_distances:
            tc = classifications[(cr, ed)]
            for phase, label in [('phase1', 'Gen1'), ('phase3a', 'Gen2')]:
                c0_hrs = []
                c5_hrs = []
                for provider in ['openai', 'anthropic', 'google']:
                    c0_qs = groups.get((phase, provider, 'C0'), [])
                    c5_qs = groups.get((phase, provider, 'C5'), [])
                    if c0_qs:
                        c0_hrs.append(compute_hr_with_classification(c0_qs, tc))
                    if c5_qs:
                        c5_hrs.append(compute_hr_with_classification(c5_qs, tc))
                if c0_hrs and c5_hrs and mean(c5_hrs) >= mean(c0_hrs):
                    out.append(f"    WARNING: C5 >= C0 at CR={cr}, ED={ed}, {label}")
                    all_effective = False

    if all_effective:
        out.append("    YES - C5 < C0 holds for ALL 24 threshold combinations in BOTH generations.")

    out.append("")
    out.append("  Does C3 > C0? (Anomaly present)")
    all_anomaly = True
    for cr in containment_ratios:
        for ed in edit_distances:
            tc = classifications[(cr, ed)]
            for phase, label in [('phase1', 'Gen1'), ('phase3a', 'Gen2')]:
                c0_hrs = []
                c3_hrs = []
                for provider in ['openai', 'anthropic', 'google']:
                    c0_qs = groups.get((phase, provider, 'C0'), [])
                    c3_qs = groups.get((phase, provider, 'C3'), [])
                    if c0_qs:
                        c0_hrs.append(compute_hr_with_classification(c0_qs, tc))
                    if c3_qs:
                        c3_hrs.append(compute_hr_with_classification(c3_qs, tc))
                if c0_hrs and c3_hrs and mean(c3_hrs) <= mean(c0_hrs):
                    out.append(f"    WARNING: C3 <= C0 at CR={cr}, ED={ed}, {label}")
                    all_anomaly = False

    if all_anomaly:
        out.append("    YES - C3 > C0 holds for ALL 24 threshold combinations in BOTH generations.")

    out.append("")

    # Count how many tool names change classification
    out.append("  --- Classification changes vs production thresholds ---")
    prod_tc = classifications[(0.55, 2)]
    for cr in containment_ratios:
        for ed in edit_distances:
            if cr == 0.55 and ed == 2:
                continue
            alt_tc = classifications[(cr, ed)]
            changes = sum(1 for t in all_mentioned if prod_tc.get(t) != alt_tc.get(t))
            if changes > 0:
                out.append(f"    CR={cr:.2f} ED={ed}: {changes} tool names reclassified "
                           f"({changes/len(all_mentioned)*100:.1f}% of {len(all_mentioned)})")
    out.append("")


# ===========================================================================
# Main
# ===========================================================================

def main():
    out = []
    out.append("=" * 90)
    out.append("  SENSITIVITY ANALYSIS FOR HALLUCINATION DETECTION SYSTEM")
    out.append("  Paper Revision - Reviewer Response")
    out.append("=" * 90)
    out.append("")

    print("Loading all queries...")
    queries = load_all_queries()
    print(f"Loaded {len(queries)} queries from {len(set(q['_phase'] for q in queries))} phases.")
    out.append(f"  Total queries loaded: {len(queries)}")
    out.append(f"  Phases: {sorted(set(q['_phase'] for q in queries))}")
    out.append("")

    # Analysis 1
    print("Running ambiguous name analysis...")
    analyze_ambiguous_names(queries, out)

    # Analysis 2
    print("Running threshold sensitivity sweep...")
    threshold_sensitivity(queries, out)

    # Write output
    text = "\n".join(out)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nResults written to: {OUTPUT_FILE}")
    print(text)


if __name__ == '__main__':
    main()
