#!/usr/bin/env python3
"""
H2-Only Hallucination Rate Analysis
=====================================
Computes hallucination rates decomposed by H2 (real external tools) vs. other
categories (near-miss, non-specific mentions, H1-fabricated).

For each config (C0, C5) and provider, reports:
  - Total HR:    all hallucinated / all mentioned
  - H2-only HR:  only H2 hallucinated / all mentioned
  - Inflation:   Total HR - H2-only HR (non-specific + near-miss + H1 noise)

Uses the manually verified H2 classification dictionary from
classify_hallucinations.py (80 names verified February 2026).
"""

import json
import glob
import os
import re
from collections import defaultdict

# ===========================================================================
# Configuration
# ===========================================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "results")

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'h2_rate_results.txt')

PHASES = {
    'phase1':  'Gen1 Standard (C0-C5)',
    'phase2':  'Gen1 Thinking (C0+C5)',
    'phase3a': 'Gen2 Standard (C0-C5)',
    'phase3b': 'Gen2 Thinking (C0+C5)',
}

# Phase-to-generation mapping
PHASE_GEN = {
    'phase1': 'Gen1', 'phase2': 'Gen1',
    'phase3a': 'Gen2', 'phase3b': 'Gen2',
}

# ===========================================================================
# H2 Classification Set (from classify_hallucinations.py CLASSIFICATIONS dict)
# ===========================================================================
# All names classified as H2 (real external tools, commercially available or
# open-source, NOT in our 82-tool inventory).

H2_NAMES = {
    "LTspice", "Altium", "PSpice", "Fusion", "SolidWorks", "JLCPCB", "FEMM",
    "LTSpice", "ANSYS Maxwell", "PLECS", "PSIM", "Cadence Allegro",
    "Autodesk Fusion", "COMSOL", "Cadence Sigrity", "Magnetics Designer",
    "Polar Si9000", "CST Studio", "MATLAB/Simulink", "Keysight ADS",
    "Mentor Graphics", "WEBDESIGNER",
    "Quite Universal Circuit Simulator", "AutoCAD", "HyperLynx", "QUCS",
    "Ansys Maxwell", "PSPICE", "Ansys Icepak", "Polar Si9000e",
    "Strata Developer Studio", "Simbeor", "Power Stage Designer",
    "TI Power Stage Designer", "Falstad", "FreeCAD", "ngspice",
    "WebDesigner+", "Cadence OrCAD", "EasyEDA PCB Editor",
    "EasyEDA Schematic", "PCBWay", "Altair Flux", "SIMPLIS",
    "SIMetrix/SIMPLIS", "MagNet", "OpenModelica", "AppCAD", "SimScale",
    "Ansys HFSS", "OpenEMS", "QuickField", "TINA-TI", "OSH Park",
    "Seeed Studio", "OrCAD PSpice", "Cadence PSpice", "Multisim",
    "Fritzing", "Sonnet", "Qucs-S", "Infineon IPOSIM",
    "ON Semiconductor WebDesigner",
    "ON Semiconductor Power Supply Designer (PSD)",
    "onsemi Design Tools", "Texas Instruments WEBENCH", "Ansys SIwave",
    "EasyEDA 3D Viewer", "Atlc", "KiCad / Altium", "EasyEDA BOM Manager",
    "EasyEDA SPICE", "EasyEDA Simulator", "EasyEDA Gerber Viewer",
    "EasyEDA + JLCPCB", "Excel", "Git", "Analog Devices",
    "WEBENCH Power Designer", "OrCAD PCB", "OrCAD PSpice",
}

# Build a lowercase lookup for case-insensitive matching
H2_NAMES_LOWER = {n.lower() for n in H2_NAMES}


def normalize_tool_name(name):
    """Normalize a tool name (strip whitespace, trailing punctuation)."""
    clean = name.strip()
    clean = re.sub(r'[\.\,\;\:\!\?]+$', '', clean)
    return clean


def is_h2(tool_name):
    """Check if a hallucinated tool name matches an H2 classification.

    Uses exact match first, then case-insensitive match, then substring
    matching against known H2 names for compound references like
    'LTspice for' or 'LTspice or'.
    """
    clean = normalize_tool_name(tool_name)
    # Exact match
    if clean in H2_NAMES:
        return True
    # Case-insensitive match
    if clean.lower() in H2_NAMES_LOWER:
        return True
    # Substring: check if tool name starts with an H2 name (handles
    # "LTspice for", "LTspice or", "SolidWorks Electrical", etc.)
    lower = clean.lower()
    for h2_lower in H2_NAMES_LOWER:
        if lower.startswith(h2_lower + ' ') or lower.startswith(h2_lower + ','):
            return True
        # Also check if the H2 name starts with our tool name (shorter references)
        if h2_lower.startswith(lower + ' ') and len(clean) >= 4:
            return True
    return False


def load_all_results():
    """Load all result JSON files from all phases."""
    all_entries = []
    phase_counts = {}
    for phase_dir in PHASES:
        phase_path = os.path.join(BASE_DIR, phase_dir)
        if not os.path.exists(phase_path):
            print(f"WARNING: Phase directory not found: {phase_path}")
            continue
        json_files = glob.glob(os.path.join(phase_path, 'results_*.json'))
        count = 0
        for fpath in json_files:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        entry['_phase'] = phase_dir
                        all_entries.append(entry)
                        count += 1
                elif isinstance(data, dict):
                    data['_phase'] = phase_dir
                    all_entries.append(data)
                    count += 1
        phase_counts[phase_dir] = count
    return all_entries, phase_counts


def main():
    all_entries, phase_counts = load_all_results()
    total = sum(phase_counts.values())
    print(f"Loaded {total} entries from {len(phase_counts)} phases")
    for p, c in phase_counts.items():
        print(f"  {p}: {c}")

    # =====================================================================
    # Accumulate per (config, provider, generation) and per (config, generation)
    # =====================================================================
    # Key: (config, provider, gen) -> {mentioned, hallucinated, h2_hallucinated, queries}
    stats = defaultdict(lambda: {'mentioned': 0, 'hallucinated': 0, 'h2': 0, 'queries': 0})

    for entry in all_entries:
        config = entry.get('ConfigId', '?')
        provider = entry.get('ProviderId', '?')
        phase = entry.get('_phase', '?')
        gen = PHASE_GEN.get(phase, '?')
        mode = 'thinking' if phase in ('phase2', 'phase3b') else 'standard'

        mentioned = entry.get('MentionedToolNames', []) or []
        hallucinated = entry.get('HallucinatedToolNames', []) or []

        n_mentioned = len(mentioned)
        n_hallucinated = len(hallucinated)
        n_h2 = sum(1 for t in hallucinated if is_h2(t))

        key = (config, provider, gen, mode)
        stats[key]['mentioned'] += n_mentioned
        stats[key]['hallucinated'] += n_hallucinated
        stats[key]['h2'] += n_h2
        stats[key]['queries'] += 1

    # =====================================================================
    # Report
    # =====================================================================
    lines = []
    lines.append("=" * 100)
    lines.append("H2-ONLY HALLUCINATION RATE ANALYSIS")
    lines.append("=" * 100)
    lines.append(f"Total entries: {total}")
    lines.append("")

    # For each generation + mode combination, show C0 vs C5
    for gen in ['Gen1', 'Gen2']:
        for mode in ['standard', 'thinking']:
            lines.append("")
            lines.append("=" * 100)
            lines.append(f"{gen} {mode.upper()}")
            lines.append("=" * 100)
            lines.append("")

            providers_seen = sorted(set(
                k[1] for k in stats if k[2] == gen and k[3] == mode
            ))

            for target_config in ['C0', 'C5']:
                lines.append(f"--- {target_config} ---")
                lines.append(f"{'Provider':<12} {'Queries':>7} {'Mentioned':>10} {'Halluc':>8} "
                             f"{'H2':>6} {'TotalHR':>9} {'H2-HR':>9} {'Inflate':>9}")
                lines.append("-" * 80)

                agg_mentioned = 0
                agg_halluc = 0
                agg_h2 = 0
                agg_queries = 0

                for prov in providers_seen:
                    key = (target_config, prov, gen, mode)
                    s = stats.get(key)
                    if not s or s['queries'] == 0:
                        continue
                    m = s['mentioned']
                    h = s['hallucinated']
                    h2 = s['h2']
                    q = s['queries']

                    total_hr = (h / m * 100) if m > 0 else 0
                    h2_hr = (h2 / m * 100) if m > 0 else 0
                    inflate = total_hr - h2_hr

                    lines.append(f"{prov:<12} {q:>7} {m:>10} {h:>8} "
                                 f"{h2:>6} {total_hr:>8.2f}% {h2_hr:>8.2f}% {inflate:>8.2f}%")

                    agg_mentioned += m
                    agg_halluc += h
                    agg_h2 += h2
                    agg_queries += q

                # Cross-provider average
                if agg_mentioned > 0:
                    agg_total_hr = agg_halluc / agg_mentioned * 100
                    agg_h2_hr = agg_h2 / agg_mentioned * 100
                    agg_inflate = agg_total_hr - agg_h2_hr
                    lines.append("-" * 80)
                    lines.append(f"{'AVERAGE':<12} {agg_queries:>7} {agg_mentioned:>10} {agg_halluc:>8} "
                                 f"{agg_h2:>6} {agg_total_hr:>8.2f}% {agg_h2_hr:>8.2f}% {agg_inflate:>8.2f}%")
                lines.append("")

    # =====================================================================
    # Summary table: C0 vs C5 across all phases
    # =====================================================================
    lines.append("")
    lines.append("=" * 100)
    lines.append("SUMMARY: C0 vs C5 ACROSS ALL DATA")
    lines.append("=" * 100)
    lines.append("")

    for target_config in ['C0', 'C5']:
        agg = {'mentioned': 0, 'hallucinated': 0, 'h2': 0, 'queries': 0}
        for key, s in stats.items():
            if key[0] == target_config:
                for field in agg:
                    agg[field] += s[field]

        m = agg['mentioned']
        h = agg['hallucinated']
        h2 = agg['h2']
        q = agg['queries']
        total_hr = (h / m * 100) if m > 0 else 0
        h2_hr = (h2 / m * 100) if m > 0 else 0
        inflate = total_hr - h2_hr
        non_h2 = h - h2

        lines.append(f"{target_config}:")
        lines.append(f"  Queries:           {q}")
        lines.append(f"  Mentioned tools:   {m}")
        lines.append(f"  All hallucinated:  {h}  (Total HR = {total_hr:.2f}%)")
        lines.append(f"  H2 hallucinated:   {h2}  (H2-only HR = {h2_hr:.2f}%)")
        lines.append(f"  Non-H2 halluc:     {non_h2}  (Inflation = {inflate:.2f}%)")
        lines.append("")

    # =====================================================================
    # Key finding: architecture effectiveness on H2-only
    # =====================================================================
    lines.append("=" * 100)
    lines.append("KEY FINDING: ARCHITECTURE EFFECTIVENESS (C0 -> C5)")
    lines.append("=" * 100)
    lines.append("")

    # Compute C0 and C5 totals
    c0_agg = {'mentioned': 0, 'hallucinated': 0, 'h2': 0}
    c5_agg = {'mentioned': 0, 'hallucinated': 0, 'h2': 0}
    for key, s in stats.items():
        if key[0] == 'C0':
            for f in c0_agg: c0_agg[f] += s[f]
        elif key[0] == 'C5':
            for f in c5_agg: c5_agg[f] += s[f]

    c0_total_hr = c0_agg['hallucinated'] / c0_agg['mentioned'] * 100 if c0_agg['mentioned'] else 0
    c5_total_hr = c5_agg['hallucinated'] / c5_agg['mentioned'] * 100 if c5_agg['mentioned'] else 0
    c0_h2_hr = c0_agg['h2'] / c0_agg['mentioned'] * 100 if c0_agg['mentioned'] else 0
    c5_h2_hr = c5_agg['h2'] / c5_agg['mentioned'] * 100 if c5_agg['mentioned'] else 0

    total_reduction = (1 - c5_total_hr / c0_total_hr) * 100 if c0_total_hr else 0
    h2_reduction = (1 - c5_h2_hr / c0_h2_hr) * 100 if c0_h2_hr else 0

    lines.append(f"Total HR:  C0 = {c0_total_hr:.2f}%  ->  C5 = {c5_total_hr:.2f}%  "
                 f"(reduction: {total_reduction:.1f}%)")
    lines.append(f"H2-only:   C0 = {c0_h2_hr:.2f}%  ->  C5 = {c5_h2_hr:.2f}%  "
                 f"(reduction: {h2_reduction:.1f}%)")
    lines.append("")
    lines.append(f"Inflation (Total HR - H2 HR):")
    lines.append(f"  C0: {c0_total_hr - c0_h2_hr:.2f} pp  "
                 f"({(c0_total_hr - c0_h2_hr)/c0_total_hr*100:.1f}% of total HR)" if c0_total_hr else "  C0: n/a")
    lines.append(f"  C5: {c5_total_hr - c5_h2_hr:.2f} pp  "
                 f"({(c5_total_hr - c5_h2_hr)/c5_total_hr*100:.1f}% of total HR)" if c5_total_hr else "  C5: n/a")
    lines.append("")

    # =====================================================================
    # Per-provider C0->C5 decomposition
    # =====================================================================
    lines.append("=" * 100)
    lines.append("PER-PROVIDER H2-ONLY RATES (C0 vs C5, all modes combined)")
    lines.append("=" * 100)
    lines.append("")

    providers = sorted(set(k[1] for k in stats))
    lines.append(f"{'Provider':<12} {'C0_TotalHR':>11} {'C0_H2HR':>9} {'C0_Infl':>9} "
                 f"{'C5_TotalHR':>11} {'C5_H2HR':>9} {'C5_Infl':>9}")
    lines.append("-" * 80)

    for prov in providers:
        for target_config, agg_dict in [('C0', {}), ('C5', {})]:
            agg_dict_local = {'mentioned': 0, 'hallucinated': 0, 'h2': 0}
            for key, s in stats.items():
                if key[0] == target_config and key[1] == prov:
                    for f in agg_dict_local:
                        agg_dict_local[f] += s[f]
            if target_config == 'C0':
                c0_prov = agg_dict_local
            else:
                c5_prov = agg_dict_local

        c0m = c0_prov['mentioned']
        c5m = c5_prov['mentioned']
        c0_thr = c0_prov['hallucinated'] / c0m * 100 if c0m else 0
        c0_h2r = c0_prov['h2'] / c0m * 100 if c0m else 0
        c5_thr = c5_prov['hallucinated'] / c5m * 100 if c5m else 0
        c5_h2r = c5_prov['h2'] / c5m * 100 if c5m else 0

        lines.append(f"{prov:<12} {c0_thr:>10.2f}% {c0_h2r:>8.2f}% {c0_thr-c0_h2r:>8.2f}% "
                     f"{c5_thr:>10.2f}% {c5_h2r:>8.2f}% {c5_thr-c5_h2r:>8.2f}%")

    lines.append("")

    # =====================================================================
    # H2 name frequency in the data (top 20)
    # =====================================================================
    lines.append("=" * 100)
    lines.append("TOP 20 H2 NAMES FOUND IN HALLUCINATED TOOLS")
    lines.append("=" * 100)
    lines.append("")

    h2_freq = defaultdict(int)
    for entry in all_entries:
        hallucinated = entry.get('HallucinatedToolNames', []) or []
        for t in hallucinated:
            clean = normalize_tool_name(t)
            if is_h2(clean):
                # Map to canonical H2 name for counting
                lower = clean.lower()
                matched = clean
                for h2_name in H2_NAMES:
                    if lower == h2_name.lower() or lower.startswith(h2_name.lower() + ' '):
                        matched = h2_name
                        break
                h2_freq[matched] += 1

    top_h2 = sorted(h2_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (name, count) in enumerate(top_h2, 1):
        lines.append(f"  {i:>2}. {name:<45} {count:>5} mentions")
    lines.append("")

    # =====================================================================
    # P_any and P_any,H2: Query-level hallucination prevalence under C5
    # =====================================================================
    # P_any    = fraction of C5 queries with >= 1 hallucinated mention (any type)
    # P_any,H2 = fraction of C5 queries with >= 1 H2 hallucinated mention
    lines.append("=" * 100)
    lines.append("QUERY-LEVEL HALLUCINATION PREVALENCE UNDER C5")
    lines.append("P_any    = fraction of C5 queries with >= 1 hallucinated mention (any type)")
    lines.append("P_any,H2 = fraction of C5 queries with >= 1 H2 hallucinated mention")
    lines.append("=" * 100)
    lines.append("")

    # Accumulate per (provider, gen, mode) for C5 only
    # Key: (provider, gen, mode) -> {total_queries, queries_any_halluc, queries_any_h2}
    pany_stats = defaultdict(lambda: {'total': 0, 'any_halluc': 0, 'any_h2': 0})

    for entry in all_entries:
        config = entry.get('ConfigId', '?')
        if config != 'C5':
            continue

        provider = entry.get('ProviderId', '?')
        phase = entry.get('_phase', '?')
        gen = PHASE_GEN.get(phase, '?')
        mode = 'thinking' if phase in ('phase2', 'phase3b') else 'standard'

        hallucinated = entry.get('HallucinatedToolNames', []) or []

        has_any = len(hallucinated) > 0
        has_h2 = any(is_h2(t) for t in hallucinated)

        key = (provider, gen, mode)
        pany_stats[key]['total'] += 1
        if has_any:
            pany_stats[key]['any_halluc'] += 1
        if has_h2:
            pany_stats[key]['any_h2'] += 1

    # Report per generation+mode
    for gen in ['Gen1', 'Gen2']:
        for mode in ['standard', 'thinking']:
            lines.append(f"--- {gen} {mode.upper()} ---")
            lines.append(f"{'Provider':<12} {'Queries':>8} {'Any_Halluc':>11} {'Any_H2':>8} "
                         f"{'P_any':>9} {'P_any,H2':>10}")
            lines.append("-" * 70)

            providers_seen = sorted(set(
                k[0] for k in pany_stats if k[1] == gen and k[2] == mode
            ))

            agg_total = 0
            agg_any = 0
            agg_h2 = 0

            for prov in providers_seen:
                key = (prov, gen, mode)
                s = pany_stats.get(key)
                if not s or s['total'] == 0:
                    continue
                t_q = s['total']
                a_q = s['any_halluc']
                h_q = s['any_h2']
                p_any = a_q / t_q * 100
                p_h2 = h_q / t_q * 100

                lines.append(f"{prov:<12} {t_q:>8} {a_q:>11} {h_q:>8} "
                             f"{p_any:>8.2f}% {p_h2:>9.2f}%")

                agg_total += t_q
                agg_any += a_q
                agg_h2 += h_q

            if agg_total > 0:
                lines.append("-" * 70)
                lines.append(f"{'AVERAGE':<12} {agg_total:>8} {agg_any:>11} {agg_h2:>8} "
                             f"{agg_any/agg_total*100:>8.2f}% {agg_h2/agg_total*100:>9.2f}%")
            lines.append("")

    # Overall C5 summary
    lines.append("--- OVERALL C5 (all providers, generations, modes) ---")
    overall_total = sum(s['total'] for s in pany_stats.values())
    overall_any = sum(s['any_halluc'] for s in pany_stats.values())
    overall_h2 = sum(s['any_h2'] for s in pany_stats.values())
    if overall_total > 0:
        lines.append(f"  Total C5 queries:           {overall_total}")
        lines.append(f"  Queries with any halluc:    {overall_any}  (P_any    = {overall_any/overall_total*100:.2f}%)")
        lines.append(f"  Queries with any H2:        {overall_h2}  (P_any,H2 = {overall_h2/overall_total*100:.2f}%)")
        lines.append(f"  P_any,H2 / P_any ratio:     {overall_h2/overall_any*100:.1f}%" if overall_any > 0 else "  P_any,H2 / P_any ratio:     n/a")
    lines.append("")

    # Per-provider summary (all modes combined)
    lines.append("--- PER-PROVIDER C5 SUMMARY (all gen+modes combined) ---")
    lines.append(f"{'Provider':<12} {'Queries':>8} {'P_any':>9} {'P_any,H2':>10} {'Ratio':>8}")
    lines.append("-" * 55)

    prov_agg = defaultdict(lambda: {'total': 0, 'any_halluc': 0, 'any_h2': 0})
    for key, s in pany_stats.items():
        prov = key[0]
        for f in ('total', 'any_halluc', 'any_h2'):
            prov_agg[prov][f] += s[f]

    for prov in sorted(prov_agg.keys()):
        s = prov_agg[prov]
        t_q = s['total']
        if t_q == 0:
            continue
        p_any = s['any_halluc'] / t_q * 100
        p_h2 = s['any_h2'] / t_q * 100
        ratio = s['any_h2'] / s['any_halluc'] * 100 if s['any_halluc'] > 0 else 0
        lines.append(f"{prov:<12} {t_q:>8} {p_any:>8.2f}% {p_h2:>9.2f}% {ratio:>7.1f}%")

    lines.append("")

    output = '\n'.join(lines)
    print(output)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
