#!/usr/bin/env python3
"""
Recall-equalized detector applied to THINKING-mode cells (completes the corrected metric
for the whole paper). Same method + gazetteer as matched_detector_c3.py (loaded from
matched_detector_c3_gazetteer.json). No new API calls. Empty-gazetteer == production (control).

Cells re-scored:
  Gen1 thinking (phase2):  C0,C5  (o4-mini, Sonnet 4.5 thn, Gemini 2.5 Flash)
  Gen2 thinking (phase3b): C0,C5  (GPT-5.2, Sonnet 4.6, Gemini 3.1 Flash-Lite)
  Gen2 thinking C3 (reasoning-c3): C3
Also recomputes the STANDARD corrected C5 (phase1/phase3a) so all 12 configs' C5 CIs are in one table.
"""
import json, glob, os, re, random, sys
from collections import defaultdict
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__))
REPL=os.path.join(HERE,"..","data","results")
sys.path.insert(0, HERE)
from compute_reviewer_metrics import bootstrap_ci, mean

# ---- load the SAME gazetteer/rules used for the standard configs (proves it is unchanged) ----
GZ=json.load(open(os.path.join(HERE,"matched_detector_c3_gazetteer.json"),encoding="utf-8"))
GAZETTEER=GZ["gazetteer_canonical_to_regex"]; LABEL_WORDS=set(GZ["label_words"])
TOOL_PATS={c:[re.compile(r"\b"+p+r"\b",re.I) for p in ps] for c,ps in GAZETTEER.items()}
DEDUP_KEY={c:c.split()[0] for c in GAZETTEER}
TOOL_TOKENS=set("ltspice pspice femm comsol ansys psim plecs altium cadence solidworks autocad fusion matlab".split())
def is_label(name):
    toks=set(re.findall(r"[a-z0-9]+",name.lower()))
    if toks & TOOL_TOKENS: return False
    return bool(toks) and (toks <= LABEL_WORDS or name.strip().lower() in ("phase","step","stage"))

def load(phase,prov,mode,cfg,m):
    recs=[]
    for f in sorted(glob.glob(os.path.join(REPL,phase,f"results_{prov}_{mode}_*_{cfg}.json"))):
        for r in json.load(open(f,encoding="utf-8")):
            if r.get("Error") is None and r.get("RawResponse") and r["ModelId"]==m: recs.append(r)
    return recs
def annotate(recs, tool_pats):
    for r in recs:
        ph=r["HallucinatedToolNames"]; pht=" ".join(ph).lower()
        labels=sum(1 for h in ph if is_label(h)); tools=len(ph)-labels
        new=sum(1 for c,pats in tool_pats.items() if DEDUP_KEY.get(c,c.split()[0]) not in pht and any(p.search(r["RawResponse"]) for p in pats)) if tool_pats else 0
        r["_g"]=len(r["GroundedToolNames"]); r["_tools"]=tools; r["_labels"]=labels; r["_new"]=new
    return recs
def num_den(r,which):
    g=r["_g"]
    if which=="prod": return r["_tools"]+r["_labels"], len(r["MentionedToolNames"])
    if which=="i": n=r["_tools"]+r["_new"]; return n,g+n
    n=r["_tools"]+r["_labels"]+r["_new"]; return n,g+n
def macroD(recs,which):
    by=defaultdict(lambda:[0,0])
    for r in recs:
        n,d=num_den(r,which); by[r["DomainId"]][0]+=n; by[r["DomainId"]][1]+=d
    return mean([v[0]/v[1]*100 if v[1] else 0.0 for d,v in sorted(by.items())])
def perq(recs,which):
    out=[]
    for r in recs:
        n,d=num_den(r,which)
        if d>0: out.append(n/d*100)
    return out

# ---- cell definitions ----
STD=[("phase1","Gen1","standard",{"openai":"gpt-4.1","anthropic":"claude-sonnet-4-5-20250929","google":"gemini-2.5-flash-lite"}),
     ("phase3a","Gen2","standard",{"openai":"gpt-5.2","anthropic":"claude-sonnet-4-6","google":"gemini-3.1-flash-lite-preview"})]
THN=[("phase2","Gen1","thinking",{"openai":"o4-mini","anthropic":"claude-sonnet-4-5-20250929","google":"gemini-2.5-flash"}),
     ("phase3b","Gen2","thinking",{"openai":"gpt-5.2","anthropic":"claude-sonnet-4-6","google":"gemini-3.1-flash-lite-preview"})]
THN_C3=("reasoning-c3","Gen2","thinking",{"openai":"gpt-5.2","anthropic":"claude-sonnet-4-6","google":"gemini-3.1-flash-lite-preview"})
PROVS=["openai","anthropic","google"]
DISP={("Gen1","thinking","openai"):"o4-mini",("Gen1","thinking","anthropic"):"Sonnet 4.5 (thn)",("Gen1","thinking","google"):"Gemini 2.5 Flash",
      ("Gen2","thinking","openai"):"GPT-5.2 (thn)",("Gen2","thinking","anthropic"):"Sonnet 4.6 (thn)",("Gen2","thinking","google"):"Gemini 3.1 FL (thn)",
      ("Gen1","standard","openai"):"GPT-4.1",("Gen1","standard","anthropic"):"Sonnet 4.5",("Gen1","standard","google"):"Gemini 2.5 FL",
      ("Gen2","standard","openai"):"GPT-5.2",("Gen2","standard","anthropic"):"Sonnet 4.6",("Gen2","standard","google"):"Gemini 3.1 FL"}

OUT=[]; P=lambda s="":(OUT.append(s),print(s))

# =================== SANITY CHECK (empty gazetteer == production) ===================
SANITY={("Gen1","C0"):{"openai":72.3,"anthropic":73.6,"google":64.5},("Gen1","C5"):{"openai":6.6,"anthropic":11.3,"google":5.5},
        ("Gen2","C0"):{"openai":61.8,"anthropic":72.3,"google":69.1},("Gen2","C5"):{"openai":3.3,"anthropic":14.9,"google":5.2}}
P("="*108); P("  SANITY CHECK — empty-gazetteer corrected (variant ii) must reproduce PRODUCTION thinking numbers"); P("="*108)
ok=True
for phase,lab,mode,models in THN:
    for cfg in ["C0","C5"]:
        for prov,m in models.items():
            recs=annotate(load(phase,prov,mode,cfg,m), {})   # EMPTY gazetteer
            prod=macroD(recs,"prod"); empty_corr=macroD(recs,"ii")  # ii with empty gaz == production
            target=SANITY[(lab,cfg)][prov]
            match = abs(prod-target)<0.1 and abs(empty_corr-prod)<1e-9
            ok = ok and match
            P(f"  {lab} thn {cfg} {prov:10}: production={prod:5.1f}  empty-gaz-corrected={empty_corr:5.1f}  published={target:5.1f}  {'OK' if match else 'MISMATCH'}")
P("\n  "+("ALL SANITY CHECKS PASSED" if ok else "*** SANITY MISMATCH — STOPPING ***"))
if not ok:
    open(os.path.join(HERE,"matched_detector_thinking_RESULTS.txt"),"w",encoding="utf-8").write("\n".join(OUT))
    sys.exit("Sanity failed")

# =================== CORRECTED ANALYSIS (real gazetteer) ===================
# annotate all needed cells with the REAL gazetteer
CELLS={}
for phase,lab,mode,models in STD:
    for cfg in ["C0","C5"]:
        for prov,m in models.items(): CELLS[(lab,mode,prov,cfg)]=annotate(load(phase,prov,mode,cfg,m),TOOL_PATS)
for phase,lab,mode,models in THN:
    for cfg in ["C0","C5"]:
        for prov,m in models.items(): CELLS[(lab,mode,prov,cfg)]=annotate(load(phase,prov,mode,cfg,m),TOOL_PATS)
for prov,m in THN_C3[3].items(): CELLS[("Gen2","thinking",prov,"C3")]=annotate(load(THN_C3[0],prov,"thinking","C3",m),TOOL_PATS)

# ---- (1) thinking C0/C5 production vs corrected, + corrected Gen2 thinking C3 ----
P("\n"+"="*108); P("  (1) THINKING-MODE: production vs CORRECTED (recall-equalized) HR  [variant i]"); P("="*108)
P(f"  {'Gen':4} {'Model':18} | {'prodC0':>6} {'corrC0':>6} | {'prodC5':>6} {'corrC5':>6} | {'prodC3':>6} {'corrC3':>6}")
# thinking rows in the SAME schema as the standard ablation CSV: gen,mode,provider,detector,C0,C1,C2,C3,C4,C5,reduction
thinking_rows=[]
for lab in ["Gen1","Gen2"]:
    for prov in PROVS:
        c0=CELLS[(lab,"thinking",prov,"C0")]; c5=CELLS[(lab,"thinking",prov,"C5")]
        pc0,cc0=macroD(c0,"prod"),macroD(c0,"i"); pc5,cc5=macroD(c5,"prod"),macroD(c5,"i")
        if lab=="Gen2":
            c3=CELLS[("Gen2","thinking",prov,"C3")]; pc3,cc3=macroD(c3,"prod"),macroD(c3,"i")
            c3p,c3c=f"{pc3:.1f}",f"{cc3:.1f}"; c3s=f"| {pc3:6.1f} {cc3:6.1f}"
        else: c3p=c3c=""; c3s="| {:>6} {:>6}".format("--","--")
        P(f"  {lab:4} {DISP[(lab,'thinking',prov)]:18} | {pc0:6.1f} {cc0:6.1f} | {pc5:6.1f} {cc5:6.1f} {c3s}")
        thinking_rows.append(f"{lab},thinking,{prov},production,{pc0:.1f},,,{c3p},,{pc5:.1f},{pc0-pc5:.1f}")
        thinking_rows.append(f"{lab},thinking,{prov},corrected,{cc0:.1f},,,{c3c},,{cc5:.1f},{cc0-cc5:.1f}")

# ---- (2) corrected C5 with 95% bootstrap CI for ALL TWELVE configs ----
P("\n"+"="*108); P("  (2) CORRECTED C5 HR + 95% bootstrap CI (10k, seed 42) — ALL 12 MODEL CONFIGS"); P("="*108)
P(f"  {'Gen':4} {'Mode':9} {'Model':18} {'prodC5':>7} {'corrC5':>7} {'corr 95% CI':>16}")
allcorr=[]
for lab in ["Gen1","Gen2"]:
    for mode in ["standard","thinking"]:
        for prov in PROVS:
            recs=CELLS[(lab,mode,prov,"C5")]
            pc5=macroD(recs,"prod"); cc5=macroD(recs,"i")
            _,lo,hi=bootstrap_ci(perq(recs,"i"))
            allcorr.append((lab,mode,DISP[(lab,mode,prov)],cc5))
            P(f"  {lab:4} {mode:9} {DISP[(lab,mode,prov)]:18} {pc5:7.1f} {cc5:7.1f}  [{lo:5.1f},{hi:5.1f}]")

# ---- (3) cross-provider corrected C5 averages + min/max range ----
P("\n"+"="*108); P("  (3) CORRECTED C5 cross-provider averages + min/max range (sets abstract C5 range)"); P("="*108)
for lab in ["Gen1","Gen2"]:
    for mode in ["standard","thinking"]:
        vals=[v for (l,mo,nm,v) in allcorr if l==lab and mo==mode]
        P(f"  {lab} {mode:9}: corrected C5 cross-provider avg = {mean(vals):4.1f}%   (per-model: {', '.join(f'{v:.1f}' for v in vals)})")
allvals=[v for (l,mo,nm,v) in allcorr]
mn=min(allcorr,key=lambda x:x[3]); mx=max(allcorr,key=lambda x:x[3])
P(f"\n  CORRECTED C5 range across all 12 models: {mn[3]:.1f}% ({mn[2]}, {mn[0]} {mn[1]})  to  {mx[3]:.1f}% ({mx[2]}, {mx[0]} {mx[1]})")
# production range for reference
P(f"  (production C5 range for reference: GPT-5.2 std 3.3% to Sonnet 4.6 std/thn ~13-15%)")

# ---- std-vs-thinking difference check (reasoning-null must survive) ----
P("\n"+"="*108); P("  STD vs THINKING corrected-C5 difference (the 'reasoning gives no benefit' conclusion must survive)"); P("="*108)
P(f"  {'Gen':4} {'Provider':10} {'std corrC5':>11} {'thn corrC5':>11} {'thn-std':>8}   production(thn-std)")
for lab in ["Gen1","Gen2"]:
    for prov in PROVS:
        sc=macroD(CELLS[(lab,'standard',prov,'C5')],"i"); tc=macroD(CELLS[(lab,'thinking',prov,'C5')],"i")
        sp=macroD(CELLS[(lab,'standard',prov,'C5')],"prod"); tp=macroD(CELLS[(lab,'thinking',prov,'C5')],"prod")
        flag="  <-- CHECK" if abs(tc-sc)>5 else ""
        P(f"  {lab:4} {prov:10} {sc:11.1f} {tc:11.1f} {tc-sc:+8.1f}   {tp-sp:+.1f}{flag}")

# write outputs (new files; copy of standard ablation + thinking rows)
open(os.path.join(HERE,"matched_detector_thinking_RESULTS.txt"),"w",encoding="utf-8").write("\n".join(OUT))
# Merge: copy of the standard ablation CSV (with a 'mode' column inserted) + appended thinking rows, one schema.
std_csv=open(os.path.join(HERE,"matched_detector_c3_ablation.csv"),encoding="utf-8").read().rstrip("\n").splitlines()
newhdr=std_csv[0].replace("generation,","generation,mode,")
std_rows=[l.replace(l.split(",")[0]+",", l.split(",")[0]+",standard,",1) for l in std_csv[1:]]
merged=[newhdr]+std_rows+["# --- thinking-mode rows (C1/C2/C4 not run in thinking mode; blank cells) ---"]+thinking_rows
open(os.path.join(HERE,"matched_detector_thinking_ablation.csv"),"w",encoding="utf-8").write("\n".join(merged)+"\n")
print("\nSaved: matched_detector_thinking_RESULTS.txt, matched_detector_thinking_ablation.csv")
