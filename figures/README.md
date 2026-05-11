# Figures

Self-contained interactive HTML figures + PNG renders for the paper.

## Files

| File | Description | Paper |
| --- | --- | --- |
| `fig7-cross-generational-c5.html` | Gen1 vs Gen2 hallucination rates per provider, under C5 | Figure 7 |
| `fig8-gen2-ablation-heatmap.html` | Gen2 ablation HR heatmap (configs C0-C5 by provider) | Figure 8 |
| `fig9-c3-anomaly-amplification.html` | C3 (JSON-only) anomaly: Gen1 vs Gen2 amplification | Figure 9 |
| `fig10-confidence-intervals.html` | C5 hallucination rates with 95% Wilson CIs | Figure 10 |
| `fig11-per-tool-mention-rate.html` | Per-tool-mention HR by model and category | Figure 11 |

PNG renders (`png/fig{7..11}.png`) for paper inclusion are produced by:
```bash
chrome --headless --disable-gpu --window-size=1500,600 --hide-scrollbars \
       --screenshot=fig7.png file:///$(realpath fig7-cross-generational-c5.html)
```

Figs 1-6 (architecture, methodology, Gen1 results) are unchanged from prior work and shipped as static PNGs in the paper folder.
