#!/usr/bin/env python3
"""TASK A: corrected per-tool-mention C5 hallucination rate (micro/pooled) for all 12 configs.
Same recall-equalized detector + gazetteer as matched_detector_c3.py. Variant i (real tools).
Per-mention = sum(hallucinated mentions) / sum(total mentions) pooled over all C5 responses.
Also prints PRODUCTION per-mention as a sanity check vs the current fig11 values."""
import json, glob, os, re, sys
HERE=os.path.dirname(os.path.abspath(__file__)); REPL=os.path.join(HERE,"..","data","results")
GZ=json.load(open(os.path.join(HERE,"matched_detector_c3_gazetteer.json"),encoding="utf-8"))
GAZETTEER=GZ["gazetteer_canonical_to_regex"]; LABEL_WORDS=set(GZ["label_words"])
TOOL_PATS={c:[re.compile(r"\b"+p+r"\b",re.I) for p in ps] for c,ps in GAZETTEER.items()}
DEDUP_KEY={c:c.split()[0] for c in GAZETTEER}
TOOL_TOKENS=set("ltspice pspice femm comsol ansys psim plecs altium cadence solidworks autocad fusion matlab".split())
def is_label(name):
    toks=set(re.findall(r"[a-z0-9]+",name.lower()))
    if toks & TOOL_TOKENS: return False
    return bool(toks) and (toks <= LABEL_WORDS or name.strip().lower() in ("phase","step","stage"))
def load(phase,prov,mode,m):
    recs=[]
    for f in sorted(glob.glob(os.path.join(REPL,phase,f"results_{prov}_{mode}_*_C5.json"))):
        for r in json.load(open(f,encoding="utf-8")):
            if r.get("Error") is None and r.get("RawResponse") and r["ModelId"]==m: recs.append(r)
    return recs
def permention(recs,which):
    num=den=0
    for r in recs:
        ph=r["HallucinatedToolNames"]; pht=" ".join(ph).lower()
        labels=sum(1 for h in ph if is_label(h)); tools=len(ph)-labels
        new=sum(1 for c,pats in TOOL_PATS.items() if DEDUP_KEY[c] not in pht and any(p.search(r["RawResponse"]) for p in pats))
        g=len(r["GroundedToolNames"])
        if which=="prod": num+=len(ph); den+=len(r["MentionedToolNames"])
        else: num+=tools+new; den+=g+tools+new
    return num/den*100 if den else 0.0
CFG=[("Gen1","standard","phase1","openai","gpt-4.1","GPT-4.1"),
     ("Gen1","standard","phase1","anthropic","claude-sonnet-4-5-20250929","Claude Sonnet 4.5"),
     ("Gen1","standard","phase1","google","gemini-2.5-flash-lite","Gemini Flash-Lite 2.5"),
     ("Gen1","thinking","phase2","openai","o4-mini","o4-mini"),
     ("Gen1","thinking","phase2","anthropic","claude-sonnet-4-5-20250929","Claude S. 4.5 (think)"),
     ("Gen1","thinking","phase2","google","gemini-2.5-flash","Gemini Flash 2.5"),
     ("Gen2","standard","phase3a","openai","gpt-5.2","GPT-5.2"),
     ("Gen2","standard","phase3a","anthropic","claude-sonnet-4-6","Claude Sonnet 4.6"),
     ("Gen2","standard","phase3a","google","gemini-3.1-flash-lite-preview","Gemini 3.1 Flash-Lite"),
     ("Gen2","thinking","phase3b","openai","gpt-5.2","GPT-5.2 (think)"),
     ("Gen2","thinking","phase3b","anthropic","claude-sonnet-4-6","Claude S. 4.6 (think)"),
     ("Gen2","thinking","phase3b","google","gemini-3.1-flash-lite-preview","Gemini 3.1 Fl.-Lite (think)")]
# current fig11 production per-mention values (sanity targets)
FIG11={"GPT-4.1":5.58,"Claude Sonnet 4.5":10.83,"Gemini Flash-Lite 2.5":3.62,"o4-mini":6.92,
 "Claude S. 4.5 (think)":10.95,"Gemini Flash 2.5":5.26,"GPT-5.2":3.48,"Claude Sonnet 4.6":13.60,
 "Gemini 3.1 Flash-Lite":5.50,"GPT-5.2 (think)":3.59,"Claude S. 4.6 (think)":15.20,"Gemini 3.1 Fl.-Lite (think)":4.70}
OUT=[]; P=lambda s="":(OUT.append(s),print(s))
P("TASK A — per-tool-mention C5 hallucination rate (micro/pooled)")
P(f"{'Gen':5}{'Mode':9}{'Model':28}{'prod%':>7}{'fig11':>7}{'match':>7}{'CORRECTED%':>12}")
ok=True
for gen,mode,phase,prov,m,disp in CFG:
    recs=load(phase,prov,mode,m); pr=permention(recs,"prod"); co=permention(recs,"i")
    tgt=FIG11[disp]; mt=abs(pr-tgt)<0.06; ok=ok and mt
    P(f"{gen:5}{mode:9}{disp:28}{pr:7.2f}{tgt:7.2f}{('OK' if mt else 'DIFF'):>7}{co:12.2f}")
P("\nproduction sanity: "+("ALL MATCH current fig11" if ok else "*** MISMATCH ***"))
open(os.path.join(HERE,"fig11_permention_corrected.txt"),"w",encoding="utf-8").write("\n".join(OUT))
print("saved fig11_permention_corrected.txt")
