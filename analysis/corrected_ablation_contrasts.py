#!/usr/bin/env python3
"""
Secondary ablation significance tests recomputed on the CORRECTED (recall-equalized) per-query HR.
Reuses the EXACT machinery of inferential_stats.py (pairing by Domain/Prompt/Rep, scipy Wilcoxon
zero_method='wilcox', matched-pairs rank-biserial, Bonferroni k=3). No new API calls.

mode='production' -> per-query HR = |Hallucinated|/|Mentioned|  (reproduces the published RES-2/RES-4 = SANITY)
mode='corrected'  -> recall-equalized variant-i per-query HR = (tools+new)/(grounded+tools+new)
                     (same scoring as matched_detector_c3.py with the committed gazetteer)

Output: corrected_ablation_contrasts.txt (does not overwrite existing results).
"""
import json, os, re, sys, math
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scipy.stats import wilcoxon, friedmanchisquare
from inferential_stats import load_block, hr_per_query, matched_pairs_rank_biserial  # exact published machinery

# ---- corrected (recall-equalized) per-query HR: same logic as matched_detector_c3.py ----
GZ=json.load(open(os.path.join(HERE,"matched_detector_c3_gazetteer.json"),encoding="utf-8"))
GAZ=GZ["gazetteer_canonical_to_regex"]; LAB=set(GZ["label_words"])
PATS={c:[re.compile(r"\b"+p+r"\b",re.I) for p in ps] for c,ps in GAZ.items()}
DK={c:c.split()[0] for c in GAZ}
TT=set("ltspice pspice femm comsol ansys psim plecs altium cadence solidworks autocad fusion matlab".split())
def is_label(n):
    t=set(re.findall(r"[a-z0-9]+",n.lower()))
    return (not (t&TT)) and bool(t) and (t<=LAB or n.strip().lower() in ("phase","step","stage"))
def corrected_hr_per_query(rows, gazetteer=True):
    out=[]
    for r in rows:
        ph=r.get("HallucinatedToolNames") or []; pht=" ".join(ph).lower()
        labels=sum(1 for h in ph if is_label(h)); tools=len(ph)-labels
        new=0
        if gazetteer:
            new=sum(1 for c,ps in PATS.items() if DK[c] not in pht and any(p.search(r.get("RawResponse","")) for p in ps))
        g=len(r.get("GroundedToolNames") or [])
        num=tools+new; den=g+num
        out.append(num/den if den>0 else 0.0)
    return out

GEN={"Gen1":("phase1",{"openai":"gpt-4.1","anthropic":"claude-sonnet-4-5-20250929","google":"gemini-2.5-flash-lite"}),
     "Gen2":("phase3a",{"openai":"gpt-5.2","anthropic":"claude-sonnet-4-6","google":"gemini-3.1-flash-lite-preview"})}
CONFIGS=["C0","C1","C2","C3","C4","C5"]
CONTRASTS=[("C2","C4"),("C2","C5"),("C4","C5")]  # paper's k=3 planned set; C4-vs-C5 = M3 contribution
K=3; ALPHA=0.05

def aligned(phase, model, hrfn):
    """Return {config: {key: hr}} aligned to the common key set across C0-C5."""
    cols={}; keysets=[]
    for cfg in CONFIGS:
        rows=load_block(phase, model, "standard", cfg)
        idx={(r["DomainId"],r["PromptIndex"],r["Repetition"]): hr for r,hr in zip(rows, hrfn(rows))}
        cols[cfg]=idx; keysets.append(set(idx))
    common=sorted(set.intersection(*keysets))
    return {cfg:[cols[cfg][k] for k in common] for cfg in CONFIGS}, common

def wilc(a,b):
    diffs=[bb-aa for aa,bb in zip(a,b)]
    nz=sum(1 for d in diffs if d!=0.0)
    if nz<1: return float("nan"), float("nan"), 0.0, nz
    try: stat,p=wilcoxon(a,b,zero_method="wilcox",alternative="two-sided")
    except ValueError: return float("nan"),float("nan"),matched_pairs_rank_biserial(diffs),nz
    return float(stat), float(p), matched_pairs_rank_biserial(diffs), nz

def run(mode):
    use_gaz = (mode=="corrected")
    hrfn = (lambda rows: corrected_hr_per_query(rows, gazetteer=use_gaz)) if mode!="production" else hr_per_query
    L=[]; P=lambda s="":L.append(s)
    P("="*104); P(f"  MODE = {mode.upper()}   (per-query HR; pairing Domain/Prompt/Rep; Wilcoxon zero_method=wilcox; Bonferroni k={K})"); P("="*104)
    for gen,(phase,models) in GEN.items():
        for prov,model in models.items():
            cols,common=aligned(phase,model,hrfn)
            n=len(common)
            means={c: (sum(cols[c])/n*100 if n else 0) for c in CONFIGS}
            chi2,fp=friedmanchisquare(*[cols[c] for c in CONFIGS])
            P(f"\n  {gen} {prov:10} (n={n} paired queries)  cell mean HR%: "+" ".join(f"{c}={means[c]:.1f}" for c in CONFIGS))
            P(f"    Friedman C0-C5: chi2={chi2:.2f} p={fp:.3g}  -> ordering {'SIGNIFICANT' if fp<ALPHA else 'n.s.'}")
            # primary C0 vs C5
            w,p,rb,nz=wilc(cols["C0"],cols["C5"]);
            P(f"    [primary] C0 vs C5: W={w:.0f} p={p:.3g} r_rb={rb:+.3f}  {'SIG' if p<ALPHA else 'n.s.'}  (C5{'<' if means['C5']<means['C0'] else '>='}C0)")
            P(f"    planned k=3 contrasts (Bonferroni-adjusted):")
            P(f"      {'contrast':10} {'meanA->meanB':>14} {'W':>7} {'p_raw':>9} {'p_bonf':>9} {'r_rb':>7} {'dir':>14} {'sig(adj)':>9}")
            for a_cfg,b_cfg in CONTRASTS:
                w,p,rb,nz=wilc(cols[a_cfg],cols[b_cfg])
                padj=min(1.0,p*K) if not math.isnan(p) else float("nan")
                d="B<A (helps)" if means[b_cfg]<means[a_cfg] else "B>A (hurts)"
                lbl=f"{a_cfg}vs{b_cfg}"+("=M3" if (a_cfg,b_cfg)==("C4","C5") else "")
                sig="yes" if (not math.isnan(padj) and padj<ALPHA) else "no"
                P(f"      {lbl:10} {means[a_cfg]:5.1f}->{means[b_cfg]:5.1f}  {w:7.0f} {p:9.3g} {padj:9.3g} {rb:+7.3f} {d:>14} {sig:>9}")
    return "\n".join(L)

if __name__=="__main__":
    sanity=run("production")
    print(sanity)
    print("\n\n")
    corrected=run("corrected")
    print(corrected)
    open(os.path.join(HERE,"corrected_ablation_contrasts.txt"),"w",encoding="utf-8").write(
        "SANITY CHECK (production detector — must reproduce published RES-2 / RES-4)\n"+sanity+
        "\n\n\n"+"#"*104+"\nCORRECTED (recall-equalized detector — PRIMARY) \n"+"#"*104+"\n"+corrected+"\n")
    print("\nsaved corrected_ablation_contrasts.txt")
