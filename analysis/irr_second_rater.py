#!/usr/bin/env python3
"""Inter-rater reliability (IRR) verification of the 197-name H1/H2 classification.

Reproduces the second-rater verification reported in Section 4.2 (Practical
Implications and Deployment) and the Threats to Validity section of the paper:

  - Cohen's kappa and raw agreement between the first author (rater 1) and an
    independent second rater (a university researcher, not an author of this
    study) who classified all 197 names blind to rater 1's labels;
  - the unanimous finding that no name is a fabricated (H1) tool name;
  - the consensus resolution of the 44 disagreements;
  - the mention-weighted decomposition under rater-1 vs consensus labels
    (Section 4.2 percentages), using the per-name mention counts from
    supplementary/supplement_197_classifications.csv.

Inputs (all in this repository):
  ../supplementary/irr-second-rater/irr_rater2_categories.csv
  ../supplementary/irr-second-rater/irr_final_labels_consensus.csv
  ../supplementary/supplement_197_classifications.csv

Uses only the Python standard library. Run from the analysis/ directory:
  python irr_second_rater.py
"""
import csv
import math
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = os.path.join(HERE, "..", "supplementary")
CATS = ["H2", "Near-miss", "Non-specific", "H1"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    final_rows = read_csv(os.path.join(SUPP, "irr-second-rater", "irr_final_labels_consensus.csv"))
    supp_rows = read_csv(os.path.join(SUPP, "supplement_197_classifications.csv"))

    r1 = {int(r["ID"]): r["Rater1_Category"] for r in final_rows}
    r2 = {int(r["ID"]): r["Rater2_Category"] for r in final_rows}
    fin = {int(r["ID"]): r["Final_Category"] for r in final_rows}
    ids = sorted(r1)
    n = len(ids)

    # --- agreement and Cohen's kappa (rater 1 vs rater 2, pre-consensus) ---
    agree = [i for i in ids if r1[i] == r2[i]]
    po = len(agree) / n
    c1 = Counter(r1[i] for i in ids)
    c2 = Counter(r2[i] for i in ids)
    pe = sum((c1[c] / n) * (c2[c] / n) for c in CATS)
    kappa = (po - pe) / (1 - pe)
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    lo, hi = kappa - 1.96 * se, min(kappa + 1.96 * se, 1.0)

    print("=== Inter-rater agreement (rater 1 vs rater 2, %d names) ===" % n)
    print("raw agreement : %d/%d = %.1f%%" % (len(agree), n, 100 * po))
    print("Cohen's kappa : %.3f  (approx. 95%% CI [%.3f, %.3f])" % (kappa, lo, hi))
    print()
    print("category distributions:")
    print("  %-14s %8s %8s" % ("category", "rater1", "rater2"))
    for c in CATS:
        print("  %-14s %8d %8d" % (c, c1[c], c2[c]))
    print()
    print("confusion matrix (rows = rater 1, cols = rater 2):")
    M = defaultdict(int)
    for i in ids:
        M[(r1[i], r2[i])] += 1
    print("  %-14s" % "" + "".join("%14s" % c for c in CATS))
    for a in CATS:
        print("  %-14s" % a + "".join("%14d" % M[(a, b)] for b in CATS))
    print()
    h1_unanimous = (c1["H1"] == 0 and c2["H1"] == 0)
    print("fabricated (H1) names: rater1 = %d, rater2 = %d -> unanimous zero-H1: %s"
          % (c1["H1"], c2["H1"], h1_unanimous))
    print()

    # --- consensus resolution of the disagreements ---
    dis = [i for i in ids if r1[i] != r2[i]]
    to_r1 = sum(1 for i in dis if fin[i] == r1[i])
    to_r2 = sum(1 for i in dis if fin[i] == r2[i])
    third = sum(1 for i in dis if fin[i] not in (r1[i], r2[i]))
    cf = Counter(fin[i] for i in ids)
    print("=== Consensus resolution ===")
    print("disagreements: %d  (resolved: %d to rater 1's label, %d to rater 2's, %d to a jointly agreed third category)"
          % (len(dis), to_r1, to_r2, third))
    print("final (consensus) name-level distribution: "
          + ", ".join("%s %d" % (c, cf[c]) for c in CATS))
    print()

    # --- mention-weighted decomposition: rater-1 vs consensus labels ---
    counts = {int(r["Rank"]): (int(r["TotalMentions"]), int(r["C5_Mentions"]), int(r["C0_Mentions"]))
              for r in supp_rows}
    tot = defaultdict(int); new = defaultdict(int)
    c5_old = c5_new = c0_old = c0_new = 0
    T = 0
    for i in ids:
        m, c5, c0 = counts[i]
        T += m
        tot[r1[i]] += m
        new[fin[i]] += m
        if r1[i] == "H2":
            c5_old += c5; c0_old += c0
        if fin[i] == "H2":
            c5_new += c5; c0_new += c0
    print("=== Mention-weighted decomposition (%d classified mentions) ===" % T)
    print("  %-14s %12s %8s %12s %8s" % ("category", "rater1", "%", "consensus", "%"))
    max_shift = 0.0
    for c in CATS:
        p_old, p_new = 100 * tot[c] / T, 100 * new[c] / T
        max_shift = max(max_shift, abs(p_old - p_new))
        print("  %-14s %12d %7.1f%% %12d %7.1f%%" % (c, tot[c], p_old, new[c], p_new))
    print("maximum category shift: %.1f pp (paper's sensitivity bound: +/-4.5 pp)" % max_shift)
    print()
    print("C5 mentions of H2-labelled names: rater1 = %d, consensus = %d -> C5-level H2 figures %s"
          % (c5_old, c5_new, "identical" if c5_old == c5_new else "DIFFER"))
    print("C0 mentions of H2-labelled names: rater1 = %d, consensus = %d" % (c0_old, c0_new))
    # the paper's H2-only reduction-from-C0 figure: 91.3% under rater-1 labels;
    # rescaling the implied H2-only C0 rate by the consensus C0 mention ratio gives the updated value.
    reduction_old = 0.913
    c0_rate_ratio = c0_new / c0_old
    reduction_new = 1 - (1 - reduction_old) / c0_rate_ratio
    print("H2-only reduction from C0: %.1f%% (rater-1 labels) -> %.1f%% (consensus labels)"
          % (100 * reduction_old, 100 * reduction_new))


if __name__ == "__main__":
    main()
