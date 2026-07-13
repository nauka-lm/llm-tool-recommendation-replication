#!/usr/bin/env python3
"""
Temperature-confound sensitivity analysis (Reviewer 3, Point 3).

Compares Google standard-mode Hallucination Rate (HR) at the original temp 0.7
(stored/published data) vs the new temp 1.0 run, for both generations, configs C0 & C5.

HR aggregation:
  - macroD : mean over the 4 domains of (per-domain micro HR)  -> matches the paper's tables
  - micro  : sum(hallucinated)/sum(mentioned) over all records
  - perq   : mean over records of h/(h+g)  -> the unit used for the paper's bootstrap CIs

Bootstrap: percentile method, 10,000 resamples, seed 42 (identical to compute_reviewer_metrics.py).
"""
import json, glob, os, random, sys
from collections import defaultdict
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPL = r"C:\Users\Nauka-LM\source\repos\llm-tool-recommendation-replication\data\results"
NEW  = r"C:\Users\Nauka-LM\source\repos\CadcomOnline\CadcomOnline.Evaluation\Results\temp-sensitivity"

# (label, generation, baseline phase dir, new temp-1.0 dir)
SETS = [
    ("Gen1  Gemini 2.5 Flash-Lite", os.path.join(REPL, "phase1"),  os.path.join(NEW, "gen1-temp10")),
    ("Gen2  Gemini 3.1 Flash-Lite", os.path.join(REPL, "phase3a"), os.path.join(NEW, "gen2-temp10")),
]
BOOT_N, SEED = 10000, 42

def load(dirpath, config):
    """Load google standard records for a config from a directory (all 4 domains)."""
    recs = []
    if not os.path.isdir(dirpath):
        return None
    for f in sorted(glob.glob(os.path.join(dirpath, f"results_google_standard_*_{config}.json"))):
        with open(f, encoding="utf-8") as fh:
            for r in json.load(fh):
                if r.get("Error") is None and r.get("RawResponse"):
                    recs.append(r)
    return recs

def perq_hrs(recs):
    out = []
    for r in recs:
        h = len(r["HallucinatedToolNames"]); g = len(r["GroundedToolNames"])
        if h + g > 0:
            out.append(h / (h + g))
    return out

def macroD(recs):
    by = defaultdict(list)
    for r in recs: by[r["DomainId"]].append(r)
    hrs = []
    for d, rs in sorted(by.items()):
        tm = sum(len(r["MentionedToolNames"]) for r in rs)
        th = sum(len(r["HallucinatedToolNames"]) for r in rs)
        hrs.append(th / tm * 100 if tm else 0.0)
    return sum(hrs) / len(hrs) if hrs else 0.0

def micro(recs):
    tm = sum(len(r["MentionedToolNames"]) for r in recs)
    th = sum(len(r["HallucinatedToolNames"]) for r in recs)
    return th / tm * 100 if tm else 0.0

def mean(x): return sum(x) / len(x) if x else 0.0

def boot_ci(data, n=BOOT_N, alpha=0.05, seed=SEED):
    if len(data) < 2:
        m = mean(data); return m, m, m
    rng = random.Random(seed); N = len(data); bm = []
    for _ in range(n):
        s = [data[rng.randint(0, N-1)] for _ in range(N)]
        bm.append(mean(s))
    bm.sort()
    return mean(data), bm[int(n*alpha/2)], bm[int(n*(1-alpha/2))]

def boot_diff_ci(new, old, n=BOOT_N, alpha=0.05, seed=SEED):
    """Bootstrap CI for (mean(new) - mean(old)), independent resampling of each group."""
    if len(new) < 2 or len(old) < 2:
        return mean(new) - mean(old), None, None
    rng = random.Random(seed); Nn, No = len(new), len(old); diffs = []
    for _ in range(n):
        sn = mean([new[rng.randint(0, Nn-1)] for _ in range(Nn)])
        so = mean([old[rng.randint(0, No-1)] for _ in range(No)])
        diffs.append(sn - so)
    diffs.sort()
    return mean(new) - mean(old), diffs[int(n*alpha/2)], diffs[int(n*(1-alpha/2))]

print("="*100)
print("  TEMPERATURE-CONFOUND SENSITIVITY: Google standard HR, temp 0.7 (published) vs temp 1.0 (new)")
print("="*100)
rows_for_summary = []
for label, base_dir, new_dir in SETS:
    print(f"\n### {label}")
    print(f"  {'Cfg':4} {'temp':5} {'macroD%':>8} {'micro%':>7} {'perQ%':>6} {'perQ 95% CI':>16} {'N':>5}")
    for config in ["C0", "C5"]:
        base = load(base_dir, config)
        new  = load(new_dir, config)
        # baseline (temp 0.7)
        bq = perq_hrs(base)
        bm, blo, bhi = boot_ci([x*100 for x in bq])
        print(f"  {config:4} {'0.7':5} {macroD(base):8.1f} {micro(base):7.1f} {mean(bq)*100:6.1f} [{blo:5.1f},{bhi:5.1f}]   {len(base):5}")
        if new is None:
            print(f"  {config:4} {'1.0':5} {'--- temp-1.0 data not present yet ---':>40}")
            continue
        nq = perq_hrs(new)
        nm, nlo, nhi = boot_ci([x*100 for x in nq])
        print(f"  {config:4} {'1.0':5} {macroD(new):8.1f} {micro(new):7.1f} {mean(nq)*100:6.1f} [{nlo:5.1f},{nhi:5.1f}]   {len(new):5}")
        # deltas
        d_macro = macroD(new) - macroD(base)
        d_perq, dlo, dhi = boot_diff_ci([x*100 for x in nq], [x*100 for x in bq])
        ci = f"[{dlo:+.1f},{dhi:+.1f}]" if dlo is not None else "n/a"
        print(f"       -> DELTA {config}: macroD {d_macro:+.1f} pp | per-query {d_perq:+.1f} pp 95% CI {ci}")
        rows_for_summary.append((label, config, macroD(base), macroD(new), d_macro, d_perq, dlo, dhi))

if rows_for_summary:
    print("\n" + "="*100)
    print("  SUMMARY (macro-by-domain point estimates; delta CI from per-query bootstrap)")
    print("="*100)
    print(f"  {'Set':30} {'Cfg':4} {'t=0.7':>7} {'t=1.0':>7} {'d-macroD':>9} {'d-perQ':>8} {'d-95%CI':>16}")
    for (label, config, b, nw, dm, dq, lo, hi) in rows_for_summary:
        ci = f"[{lo:+.1f},{hi:+.1f}]" if lo is not None else "n/a"
        print(f"  {label:30} {config:4} {b:7.1f} {nw:7.1f} {dm:+9.1f} {dq:+8.1f} {ci:>16}")
