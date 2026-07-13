# Supplementary Materials

This directory contains supporting materials for the paper.

## Files

- `supplement_197_classifications.csv` — Hand-coded H1/H2/near-miss/non-specific classifications for 197 distinct hallucinated tool names observed across the experiments (first author's labels = rater 1, with evidence URLs and per-name mention counts). Used by `analysis/classify_hallucinations.py` and `analysis/irr_second_rater.py`.
- `irr-second-rater/` — Independent second-rater verification of the 197-name classification: the blind rating sheet, the second rater's categories, the consensus final labels, and the rating protocol. Reproduces Cohen's kappa = 0.645 and the consensus decomposition reported in Section 4.2 of the paper (see `analysis/irr_second_rater.py`).

## Paper Supplementary

The paper's full supplementary-materials LaTeX document is submitted to the journal alongside the main manuscript. This repository hosts the underlying data only; the typeset supplementary PDF is part of the journal submission package.
