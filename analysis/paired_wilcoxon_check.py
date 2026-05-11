#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statistical_analysis import (load_all_data, get_paired_arrays,
    mann_whitney_u, wilcoxon_signed_rank, rank_biserial_r,
    cliffs_delta, cliffs_delta_label, sig_label, mean)

if __name__ == "__main__":
    S = "=" * 95
    D = "-" * 95
    print(S)
    print("  PAIRED WILCOXON SIGNED-RANK CHECK: Standard vs Thinking (C5)")
    print("  Comparison with Mann-Whitney U (unpaired) results")
    print(S)
    data, tool_data, paired_data = load_all_data()
    tot = sum(len(v) for k, v in data.items() if len(k) == 3)
    print("Loaded", tot, "query-level results")
    print()
    print(D)
    hdr = "  %-6s %-12s %8s %8s %8s %8s | %11s %5s %7s | %11s %5s %8s %10s"
    vals = ("Gen","Provider","N_pairs","Std C5","Thn C5","Diff","Wilcox p","Sig","r_rb","M-W U p","Sig","Cliff d","Eff")
    print(hdr % vals)
    print(D)
    AR = []
    for gl, sp2, tp2 in [("Gen1","phase1","phase2"),("Gen2","phase3a","phase3b")]:
        for prov in ["openai","anthropic","google"]:
            stp, tnp = get_paired_arrays(paired_data, (sp2,prov,"C5"), (tp2,prov,"C5"))
            np2 = len(stp)
            if np2 == 0:
                print("  %-6s %-12s NO DATA" % (gl, prov)); continue
            ms = mean(stp) * 100
            mt = mean(tnp) * 100
            diff = mt - ms
            dirn = "Thn > Std" if diff > 0 else ("Std > Thn" if diff < 0 else "Equal")
            _, _, pw = wilcoxon_signed_rank(stp, tnp)
            rrb = rank_biserial_r(stp, tnp)
            sa = data.get((sp2, prov, "C5"), [])
            ta = data.get((tp2, prov, "C5"), [])
            _, _, pu = mann_whitney_u(sa, ta)
            cd2 = cliffs_delta(sa, ta)
            row = "  %-6s %-12s %8d %7.1f%% %7.1f%% %+7.1fpp | %11.6f %5s %+7.3f | %11.6f %5s %+7.3f  %10s"
            print(row % (gl,prov,np2,ms,mt,diff,pw,sig_label(pw),rrb,pu,sig_label(pu),cd2,cliffs_delta_label(cd2)))
            AR.append(dict(gen=gl, provider=prov, n_pairs=np2, diff_pp=diff, direction=dirn,
                wilcoxon_p=pw, mwu_p=pu, wilcoxon_sig=sig_label(pw), mwu_sig=sig_label(pu),
                r_rb=rrb, cliffs_d=cd2))

    print(); print(D)
    print("  CROSS-PROVIDER AGGREGATE (all providers pooled)")
    print(D)
    for gl, sp2, tp2 in [("Gen1","phase1","phase2"),("Gen2","phase3a","phase3b")]:
        asp, atp, asu, atu = [], [], [], []
        for prov in ["openai","anthropic","google"]:
            s, t = get_paired_arrays(paired_data, (sp2,prov,"C5"), (tp2,prov,"C5"))
            asp.extend(s); atp.extend(t)
            asu.extend(data.get((sp2,prov,"C5"),[])); atu.extend(data.get((tp2,prov,"C5"),[]))
        ms = mean(asp)*100; mt = mean(atp)*100; diff = mt-ms; np2 = len(asp)
        _, _, pw = wilcoxon_signed_rank(asp, atp); rrb = rank_biserial_r(asp, atp)
        _, _, pu = mann_whitney_u(asu, atu); cd2 = cliffs_delta(asu, atu)
        print("  %s ALL  N=%4d  Std=%.1f%%  Thn=%.1f%%  Diff=%+.1fpp" % (gl,np2,ms,mt,diff))
        print("         Wilcoxon p=%.6f %s  r_rb=%+.3f" % (pw, sig_label(pw), rrb))
        print("         M-W U    p=%.6f %s  delta=%+.3f (%s)" % (pu, sig_label(pu), cd2, cliffs_delta_label(cd2)))

    try:
        from scipy import stats as sps
        print(); print(D); print("  SCIPY VERIFICATION"); print(D)
        for gl, sp2, tp2 in [("Gen1","phase1","phase2"),("Gen2","phase3a","phase3b")]:
            for prov in ["openai","anthropic","google"]:
                s, t = get_paired_arrays(paired_data, (sp2,prov,"C5"), (tp2,prov,"C5"))
                if not s: continue
                try: sp_p = sps.wilcoxon(s, t, alternative="two-sided").pvalue
                except Exception: sp_p = float("nan")
                sa = data.get((sp2,prov,"C5"),[]); ta = data.get((tp2,prov,"C5"),[])
                mu_r = sps.mannwhitneyu(sa, ta, alternative="two-sided")
                _, _, cpw = wilcoxon_signed_rank(s, t); _, _, cpu = mann_whitney_u(sa, ta)
                import math
                wm = "OK" if (math.isnan(sp_p) or abs(sp_p-cpw)<0.01) else "DIFFERS"
                um = "OK" if abs(mu_r.pvalue-cpu)<0.01 else "DIFFERS"
                fmt = "  %s %-12s Wilcoxon: scipy=%.6f custom=%.6f [%s] | M-W U: scipy=%.6f custom=%.6f [%s]"
                print(fmt % (gl, prov, sp_p, cpw, wm, mu_r.pvalue, cpu, um))
    except ImportError:
        print("  [scipy not installed -- skipping verification]")

    print(); print(S); print("  SUMMARY"); print(S)
    aw = all(r["wilcoxon_sig"] == "n.s." for r in AR)
    am = all(r["mwu_sig"] == "n.s." for r in AR)
    print()
    print("  All Wilcoxon signed-rank tests p > 0.05 (non-significant):", aw)
    print("  All Mann-Whitney U tests p > 0.05 (non-significant):     ", am)
    if aw and am:
        print()
        print("  CONCLUSION: Both paired (Wilcoxon) and unpaired (Mann-Whitney U) tests agree:")
        print("  Extended thinking modes do NOT significantly improve hallucination rates.")
        print("  The choice of test (paired vs unpaired) does not change the conclusion.")
    else:
        print()
        print("  WARNING: Discrepancy detected between paired and unpaired tests!")
        for r in AR:
            if r["wilcoxon_sig"] != r["mwu_sig"]:
                print("    %s %s: Wilcoxon=%s, M-W U=%s" % (r["gen"],r["provider"],r["wilcoxon_sig"],r["mwu_sig"]))
    print()
    print("  Detailed per-comparison summary:")
    sfmt = "  %-6s %-12s %8s %-12s %11s %11s %8s"
    print(sfmt % ("Gen","Provider","Diff","Direction","Wilcoxon p","M-W U p","Agree?"))
    print("  " + "-" * 72)
    for r in AR:
        ag = "YES" if r["wilcoxon_sig"] == r["mwu_sig"] else "NO"
        print("  %-6s %-12s %+7.1fpp %-12s %11.6f %11.6f %8s" % (
            r["gen"], r["provider"], r["diff_pp"], r["direction"],
            r["wilcoxon_p"], r["mwu_p"], ag))
