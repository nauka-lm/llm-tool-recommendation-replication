#!/usr/bin/env python3
"""
Reasoning x C3 interaction analysis (Reviewer 3, Point 6).

Measures the C3 anomaly (C3 HR - C0 HR) in THINKING mode (newly run) and compares it
to the STANDARD-mode C3 anomaly, per Gen2 provider.

Data:
  standard C0/C3 : replication phase3a
  thinking C0    : replication phase3b (model-filtered to exclude stray rows)
  thinking C3    : NEW run -> CadcomOnline.Evaluation/Results/reasoning-c3
HR = mean of per-domain HR (paper method). CIs = per-query bootstrap, 10k, seed 42.
"""
import json, glob, os, random, sys
from collections import defaultdict
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

REPL = r"C:\Users\Nauka-LM\source\repos\llm-tool-recommendation-replication\data\results"
P3A  = os.path.join(REPL, "phase3a")   # Gen2 standard
P3B  = os.path.join(REPL, "phase3b")   # Gen2 thinking (C0/C5)
NEW  = r"C:\Users\Nauka-LM\source\repos\CadcomOnline\CadcomOnline.Evaluation\Results\reasoning-c3"
GEN2_MODEL = {"openai": "gpt-5.2", "anthropic": "claude-sonnet-4-6", "google": "gemini-3.1-flash-lite-preview"}
BOOT_N, SEED = 10000, 42

def load(dirpath, prov, mode, cfg, model_keep=None):
    recs = []
    if not os.path.isdir(dirpath): return None
    for f in sorted(glob.glob(os.path.join(dirpath, f"results_{prov}_{mode}_*_{cfg}.json"))):
        for r in json.load(open(f, encoding="utf-8")):
            if r.get("Error") is None and r.get("RawResponse") and (model_keep is None or r["ModelId"] == model_keep):
                recs.append(r)
    return recs

def hr_perq(r):
    h = len(r["HallucinatedToolNames"]); g = len(r["GroundedToolNames"])
    return (h/(h+g)) if (h+g) > 0 else None

def macroD(recs):
    by = defaultdict(list)
    for r in recs: by[r["DomainId"]].append(r)
    hrs = []
    for d, rs in sorted(by.items()):
        tm = sum(len(r["MentionedToolNames"]) for r in rs); th = sum(len(r["HallucinatedToolNames"]) for r in rs)
        hrs.append(th/tm*100 if tm else 0.0)
    return sum(hrs)/len(hrs) if hrs else 0.0

def perq(recs):
    return [hr_perq(r)*100 for r in recs if hr_perq(r) is not None]

def mean(x): return sum(x)/len(x) if x else 0.0

def boot_ci(data, n=BOOT_N, seed=SEED):
    if len(data) < 2: m=mean(data); return m,m,m
    rng=random.Random(seed); N=len(data); bm=[mean([data[rng.randint(0,N-1)] for _ in range(N)]) for _ in range(n)]; bm.sort()
    return mean(data), bm[int(n*0.025)], bm[int(n*0.975)]

def boot_diff(a, b, n=BOOT_N, seed=SEED):
    if len(a)<2 or len(b)<2: return mean(a)-mean(b), None, None
    rng=random.Random(seed); Na,Nb=len(a),len(b)
    ds=[mean([a[rng.randint(0,Na-1)] for _ in range(Na)])-mean([b[rng.randint(0,Nb-1)] for _ in range(Nb)]) for _ in range(n)]; ds.sort()
    return mean(a)-mean(b), ds[int(n*0.025)], ds[int(n*0.975)]

def cliffs_delta(x, y):
    if not x or not y: return 0.0
    gt = sum(1 for xi in x for yj in y if xi > yj)
    lt = sum(1 for xi in x for yj in y if xi < yj)
    return (gt - lt) / (len(x) * len(y))

def cliffs_label(d):
    d=abs(d)
    return "negligible" if d<0.147 else "small" if d<0.33 else "medium" if d<0.474 else "large"

def wilcoxon_paired(pairs):
    """pairs: list of (a,b). Returns (W, n_nonzero, approx_p two-sided via normal approx)."""
    import math
    diffs = [a-b for a,b in pairs if a != b]
    n = len(diffs)
    if n == 0: return 0.0, 0, 1.0
    ranks = sorted(range(n), key=lambda i: abs(diffs[i]))
    # assign average ranks for ties in |diff|
    absd = sorted(abs(d) for d in diffs)
    rank_of = {}
    i = 0
    while i < n:
        j = i
        while j < n and absd[j] == absd[i]: j += 1
        avg = (i + j + 1) / 2.0  # 1-based avg rank
        for k in range(i, j): rank_of[k] = avg
        i = j
    # map each diff to a rank by abs value position
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    signed = 0.0; Wpos = 0.0; Wneg = 0.0
    for pos, idx in enumerate(order):
        r = rank_of[pos]
        if diffs[idx] > 0: Wpos += r
        else: Wneg += r
    W = min(Wpos, Wneg)
    mu = n*(n+1)/4.0
    sigma = math.sqrt(n*(n+1)*(2*n+1)/24.0)
    if sigma == 0: return W, n, 1.0
    z = (W - mu)/sigma
    p = 2*(1 - 0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    return W, n, p

print("="*104)
print("  REASONING x C3 INTERACTION (Reviewer 3, Point 6) — Gen2, thinking-mode C3 vs standard-mode C3")
print("="*104)
print(f"  {'Provider':10} {'std C0':>7} {'std C3':>7} {'std anom':>9} | {'thn C0':>7} {'thn C3':>7} {'thn anom':>9} | {'thn-std anom':>12}")
summary=[]
for prov in ["openai","anthropic","google"]:
    m = GEN2_MODEL[prov]
    sC0 = load(P3A,prov,"standard","C0",m); sC3 = load(P3A,prov,"standard","C3",m)
    tC0 = load(P3B,prov,"thinking","C0",m); tC3 = load(NEW,prov,"thinking","C3",m)
    if tC3 is None or len(tC3)==0:
        print(f"  {prov:10} (thinking C3 data not present yet)"); continue
    s_anom = macroD(sC3)-macroD(sC0); t_anom = macroD(tC3)-macroD(tC0)
    print(f"  {prov:10} {macroD(sC0):7.1f} {macroD(sC3):7.1f} {s_anom:+9.1f} | {macroD(tC0):7.1f} {macroD(tC3):7.1f} {t_anom:+9.1f} | {t_anom-s_anom:+12.1f}")
    summary.append((prov, macroD(sC0),macroD(sC3),s_anom, macroD(tC0),macroD(tC3),t_anom, sC3,tC3,sC0,tC0))

if summary:
    print("\n  --- thinking C3 HR with 95% bootstrap CI, and C3 standard-vs-thinking test ---")
    print(f"  {'Provider':10} {'thn C3 (perq mean, 95% CI)':32} {'std vs thn C3: Cliffs d':>24} {'Wilcoxon(prompt-paired) p':>26}")
    alls=[]; allt=[]
    for (prov,sc0,sc3,sa,tc0,tc3,ta,sC3,tC3,sC0,tC0) in summary:
        pq_t = perq(tC3); m,lo,hi = boot_ci(pq_t)
        pq_s = perq(sC3)
        cd = cliffs_delta(pq_t, pq_s)
        # prompt-level pairing: aggregate reps -> per (domain,prompt) mean HR, pair std vs thn
        def prompt_means(recs):
            by=defaultdict(list)
            for r in recs:
                v=hr_perq(r)
                if v is not None: by[(r["DomainId"],r["PromptIndex"])].append(v*100)
            return {k:mean(v) for k,v in by.items()}
        ps=prompt_means(sC3); pt=prompt_means(tC3)
        keys=sorted(set(ps)&set(pt))
        pairs=[(pt[k],ps[k]) for k in keys]
        W,nz,p = wilcoxon_paired(pairs)
        print(f"  {prov:10} {m:5.1f}% [{lo:4.1f}, {hi:4.1f}]{'':14} {cd:+7.3f} ({cliffs_label(cd)}){'':6} p={p:.3f} (n_pairs={len(pairs)}, nz={nz})")
        alls+=pq_s; allt+=pq_t
    cd_all=cliffs_delta(allt, alls)
    print(f"\n  POOLED (all 3 providers) std-C3 vs thn-C3 per-query: Cliffs d = {cd_all:+.3f} ({cliffs_label(cd_all)})")
    print(f"  Mean std C3 = {mean(alls):.1f}%   Mean thn C3 = {mean(allt):.1f}%   diff = {mean(allt)-mean(alls):+.1f} pp")
    print("\n  --- ANOMALY SUMMARY (C3 - C0, pp) ---")
    sa_avg=mean([x[3] for x in summary]); ta_avg=mean([x[6] for x in summary])
    print(f"  standard anomaly avg = {sa_avg:+.1f} pp ; thinking anomaly avg = {ta_avg:+.1f} pp ; amplification = {ta_avg-sa_avg:+.1f} pp")
