# Matched / recall-equalized detector — C3-anomaly robustness check (Reviewer 3, Point 2)

This folder documents a sensitivity analysis testing whether the **C3 anomaly** (JSON-enforcement-only
config C3 shows a higher hallucination rate than the unconstrained baseline C0) is an artifact of
**detection recall** rather than a real model behavior. It involves **no new API calls** — it re-scores
the already-collected standard-mode responses.

## The concern
Production scoring uses three strategies: **S1** (extract tool names from JSON `toolName`/`tool` fields),
**S2** (match the 82-tool inventory by name), **S3** (regex for software-product names). S1 only fires on
valid JSON, so it gives **C3** (JSON) high recall on out-of-inventory tools, while **C0** (free-form prose)
relies on S2/S3, which miss many real out-of-inventory tools (e.g. FEMM, PSIM, PLECS, COMSOL, ANSYS Maxwell,
SIMetrix, Magnetics Designer). Diagnostic: across standard responses, S1-invisible real tools are detected in
C3 but **~85–90 % missed in C0** — so part of the production C3−C0 gap is a detection-format asymmetry.

## What this script does
`matched_detector_c3.py` equalizes recall in the **correct direction** (raise C0 to C3's recall, rather than
crippling C3): it keeps production's grounding and hallucination detection and **adds, to every response, any
high-precision out-of-inventory tool from a curated gazetteer that appears in the raw text but production
missed** (dedup by key token, so already-flagged tools are not double-counted). Two variants:
- **variant i** — real out-of-inventory **tools only** (production slot-filling *labels* removed)
- **variant ii** — including non-specific **slot-filling labels** ("Practical Tips", "Step", "Score", …)

`HR` per cell = mean over the 4 domains of (Σ hallucinated / Σ mentioned), the paper's convention.
Grounding = production `GroundedToolNames` (unchanged). With an empty gazetteer the script reproduces the
published production numbers exactly.

## Matching rules
- Case-insensitive **word-boundary** (`\b…\b`) match of canonical tool roots in `RawResponse`.
- A gazetteer tool is counted **only when production did not already flag it** (dedup by first token), so the
  augmentation strictly *adds* missed detections.
- **Excluded** ambiguous / inventory-colliding tokens (never counted): EAGLE, jcalc, Maddox, Maxwell (alone),
  Saber, Tina, Sonnet, WEBENCH, Micron, and inventory tools KiCad/EasyEDA/Saturn PCB/PowerEsim; plus common
  slot-fill words Phase/Step/Stage/Type/Layer/Score.
- The full gazetteer, exclusions, label words, and rules are serialized to
  [`matched_detector_c3_gazetteer.json`](matched_detector_c3_gazetteer.json).

## FP-guard validation
A 40-sample stratified audit of the C0 gazetteer matches found **~100 % precision** — all were genuine
out-of-inventory tool *recommendations* (e.g. "Recommended Tool: LTspice", "consider TI WEBENCH",
"use FreeCAD"), confirming C0 is not being over-corrected by spurious matches.

## How to run
```bash
cd analysis
python matched_detector_c3.py      # requires scipy; reuses compute_reviewer_metrics.py
```
Outputs (written here):
- `matched_detector_c3_RESULTS.txt`  — full human-readable report
- `matched_detector_c3_ablation.csv` — production-vs-corrected ablation (machine-readable)
- `matched_detector_c3_gazetteer.json` — gazetteer + matching rules

## Key findings
- Under equal recall the production C3−C0 gap **shrinks ~70–80 %** but stays positive:
  cross-provider corrected gap ≈ **+2.6 pp (Gen1)** and **+4.8 pp (Gen2)** (variant i).
- The residual is **statistically significant only for Gen2 OpenAI (+7.5 pp, p≈2e-4) and Anthropic
  (+5.5 pp, p≈2e-5)**; Gen1 (all) and Gen2 Google are small and non-significant.
- **variant i ≈ variant ii** for every cell → the residual is **real-tool over-generation**, *not*
  slot-filling junk.
- The **C0→C5 reduction is unaffected, even strengthened** (corrected C0 baseline is higher), so the paper's
  central architecture-effectiveness claim is robust.

## Caveat on magnitude
The exact residual depends on gazetteer completeness and on whether C0 *passing* references are counted as
recommendations; the plausible range is **~+3 to +9 pp** (cf. an independent rough upper-bound of +6.3 pp
Gen1 / +8.6 pp Gen2). The **qualitative** conclusion is stable across methods: most of the production gap is a
detection-format artifact, a modest real residual survives for current-generation OpenAI/Anthropic, and the
C0→C5 reduction is untouched. Production and corrected numbers are both retained; which is primary is an
author/editorial decision.
