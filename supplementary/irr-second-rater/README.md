# Inter-Rater Reliability (IRR) Verification of the 197-Name Classification

This directory contains the data and protocol of the independent second-rater
verification of the H1/H2/near-miss/non-specific classification reported in the
paper (Section 4.2, Practical Implications and Deployment; Threats to Validity).

## Procedure

The 197 most frequent out-of-inventory tool names (`../supplement_197_classifications.csv`)
were originally classified by the first author (rater 1). At a reviewer's request, an
independent second rater -- a university researcher with an engineering background who is
not an author of the study -- classified all 197 names **blind to rater 1's labels**, using
the same category definitions and his own web searches. The written protocol given to the
second rater is `RATER-INSTRUCTIONS.md`; the sheet he received is
`irr_blind_rating_sheet.csv` (names only: no labels, no evidence links). The second rater
confirmed in writing that he rated independently and chose to remain unnamed; he is
therefore identified only by role.

Disagreements were then resolved in a joint consensus discussion, and the paper's
Section 4.2 decomposition uses the consensus labels.

## Key results (reproduce with `../../analysis/irr_second_rater.py`)

- Raw agreement: 153/197 = 77.7%
- Cohen's kappa: 0.645 (approx. 95% CI [0.55, 0.74]) -- substantial agreement
- Fabricated (H1) names: 0 for both raters -- unanimous (197/197) on the
  fabricated-vs-real distinction, the claim the classification underpins
- 44 disagreements, all on H2 / near-miss / non-specific boundaries; consensus
  resolved 29 to rater 1's label, 12 to rater 2's, 3 to a jointly agreed third category
- Consensus mention-weighted decomposition: 57.5% H2, 24.8% near-miss,
  17.7% non-specific, 0.0% H1 (maximum shift vs rater-1 labels: 2.1 pp);
  C5-level H2 figures are identical under both label sets

## Files

| File | Description |
|---|---|
| `irr_blind_rating_sheet.csv` | The blind sheet given to the second rater (ID, ToolName; empty Category/Notes) |
| `irr_rater2_categories.csv` | The second rater's independent categories (ID, ToolName, Category) |
| `irr_final_labels_consensus.csv` | Both raters' labels and the agreed final label per name |
| `RATER-INSTRUCTIONS.md` | The written protocol given to the second rater |

Rater 1's original labels, evidence links, and per-name mention counts are in
`../supplement_197_classifications.csv` (column `Category` = rater 1).
