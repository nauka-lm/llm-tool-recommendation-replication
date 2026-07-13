# Replication Package: Hallucination Mitigation in LLM-Based Tool Recommendation

**Paper**: *Hallucination Mitigation in LLM-Based Tool Recommendation: A Cross-Provider Architectural Ablation Study Across Two Model Generations*

**Authors**: Lavdim Menxhiqi and Galia Marinova

**Status**: Under review (2026)

---

## Overview

This repository contains the complete replication package for reproducing all results, tables, and figures reported in the paper. The study evaluates a database-grounded anti-hallucination architecture across **3 LLM providers**, **2 model generations**, and **6 ablation configurations**, totaling **6,912 API calls** with a 100% success rate.

### Key Findings

| Finding | Detail |
|---------|--------|
| Architecture effectiveness | Generation-independent (~7.5 % avg hallucination rate at C5, std mode) |
| Best model | GPT-5.2 at 3.3 % hallucination rate |
| Thinking vs standard | Standard models match or beat thinking variants |
| C3 anomaly | JSON enforcement alone *increases* hallucination (+9-15 pp) |
| Minimum effective config | M1 + M2 (C4) captures the bulk of full-architecture benefit |

---

## Repository Structure

```
llm-tool-recommendation-replication/
|
|-- README.md                          # This file
|-- LICENSE                            # CC BY 4.0
|-- CITATION.cff                       # Citation metadata
|-- DATA_DICTIONARY.md                 # JSON result field descriptions
|-- .gitignore                         # Git ignore rules
|
|-- data/
|   |-- results/
|   |   |-- phase1/                    # Gen1 standard models (72 files, 2592 queries)
|   |   |-- phase2/                    # Gen1 thinking models (24 files, 864 queries)
|   |   |-- phase3a/                   # Gen2 standard models (72 files, 2592 queries)
|   |   |-- phase3b/                   # Gen2 thinking models (24 files, 864 queries)
|   |   |-- temp-sensitivity/          # Common-temperature control re-run (Google at temp 1.0)
|   |   |-- reasoning-c3/              # Reasoning-at-C3 re-run (Gen2, all three providers)
|   |-- tool-inventory/
|       |-- tool_inventory.json        # Verified engineering tools (ground truth)
|
|-- analysis/                          # Python analysis scripts (no dependencies)
|   |-- analyze_results.py             # Single-phase analysis
|   |-- analyze_cross_generational.py  # Cross-generational analysis
|   |-- statistical_analysis.py        # Wilcoxon, Friedman, effect sizes, CIs
|   |-- classify_hallucinations.py     # H1/H2/near-miss classification
|   |-- compute_h2_rates.py            # H2-only operational rates
|   |-- compute_reviewer_metrics.py    # Query-level metrics (P_any, E[H/Q])
|   |-- compute_zero_diff.py           # Zero-difference pair analysis
|   |-- sensitivity_analysis.py        # Detection threshold sweep
|   |-- paired_wilcoxon_check.py       # Standard vs thinking Wilcoxon tests
|   |-- wilcoxon_and_expected_h.py     # Combined tests and expected hallucinations
|   |-- matched_detector_c3.py         # Recall-equalized detector: C3 detection-format bias
|   |-- matched_detector_thinking.py   # Recall-equalized detector: reasoning variant
|   |-- analyze_temp_sensitivity.py    # Common-temperature control (temperature confound)
|   |-- analyze_reasoning_c3.py        # Reasoning-at-C3 experiment
|   |-- corrected_ablation_contrasts.py # Ablation contrasts under recall-equalized detector
|   |-- irr_second_rater.py            # Inter-rater reliability of the 197-name classification
|
|-- evaluation-harness/                # .NET 8 evaluation framework (C#)
|   |-- README.md                      # Setup instructions for re-running experiments
|   |-- CadcomOnline.Evaluation.csproj # Project file
|   |-- Program.cs                     # CLI entry point
|   |-- appsettings.example.json       # Configuration template (no secrets)
|   |-- Clients/                       # LLM provider clients
|   |-- Services/                      # Core evaluation services
|   |-- Models/                        # Data models
|
|-- figures/                           # Interactive HTML visualizations
|
|-- supplementary/                     # Supplementary materials
|   |-- supplement_197_classifications.csv  # 197-name classification (rater 1, evidence URLs, mention counts)
|   |-- irr-second-rater/              # Independent second-rater verification (blind sheet, ratings, consensus)
|
|-- database/                          # Database schema and seed data
|   |-- schema.sql                     # PostgreSQL table definitions
|   |-- seed_tools.sql                 # Tool inventory insert script
|
|-- scripts/                           # Utility scripts
    |-- extract_tool_inventory.py      # Extract tool inventory from results
    |-- validate_results.py            # Verify result file integrity
```

---

## Quick Start: Verify Paper Claims

All analysis scripts use **Python 3.10+ standard library only** (no external dependencies required, except for the inferential-tests script which uses scipy).

### 1. Clone the repository

```bash
git clone https://github.com/nauka-lm/llm-tool-recommendation-replication.git
cd llm-tool-recommendation-replication
```

### 2. Run cross-generational analysis (reproduces the cross-provider tables)

```bash
cd analysis
python analyze_cross_generational.py
```

### 3. Run statistical tests (reproduces all statistical claims)

```bash
python statistical_analysis.py
```

### 4. Run hallucination classification

```bash
python classify_hallucinations.py
```

### 5. View interactive figures

Open any HTML file from `figures/` in a web browser.

---

## Experiment Design

### Models Evaluated (12 configurations)

| Generation | Provider | Standard Model | Thinking Model |
|------------|----------|----------------|----------------|
| Gen1 | OpenAI | GPT-4.1 | o4-mini |
| Gen1 | Anthropic | Claude Sonnet 4.5 | Claude Sonnet 4.5 (thinking) |
| Gen1 | Google | Gemini 2.5 Flash-Lite | Gemini 2.5 Flash * |
| Gen2 | OpenAI | GPT-5.2 | GPT-5.2 (reasoning_effort=high) |
| Gen2 | Anthropic | Claude Sonnet 4.6 | Claude Sonnet 4.6 (adaptive) |
| Gen2 | Google | Gemini 3.1 Flash-Lite | Gemini 3.1 Flash-Lite (thinkingLevel=high) |

\* Gen1 Google thinking uses Flash (not Flash-Lite) because Flash-Lite 2.5 does not support extended thinking. Gen2 Flash-Lite 3.1 supports both modes natively.

### Ablation Configurations (C0–C5)

| Config | Context Builder (M1) | Closed Vocabulary (M2) | JSON Enforcement (M3) | Description |
|--------|----------------------|------------------------|-----------------------|-------------|
| C0 | Off | Off | Off | Ungrounded baseline |
| C1 | On | Off | Off | Database context only |
| C2 | Off | On | Off | Tool name whitelist only |
| C3 | Off | Off | On | JSON format only |
| C4 | On | On | Off | Context + vocabulary |
| C5 | On | On | On | Full architecture |

### Evaluation Domains (4 engineering domains)

| Domain | Description | Tools in DB |
|--------|-------------|-------------|
| D1 | PCB Design Tool Selection | 8 |
| D2 | PCB Design Calculators | 15 |
| D3 | SMPS Design (Power Supplies, Converters, Regulators) | 8 |
| D4 | Transformer Design | 10 |

### Evaluation Metrics

| Metric | Formula | Ideal |
|--------|---------|-------|
| Hallucination Rate (HR) | Hallucinated tools / Total mentioned tools | 0 % |
| Grounding Rate (GR) | 100 % − HR | 100 % |
| Workflow Coverage (WC) | Distinct workflow stages / 6 | 100 % |
| Response Consistency (RC) | Jaccard similarity across repetitions | 100 % |

---

## Key Results Summary

### Gen1 C5 Hallucination Rates

| Provider | Standard | Thinking |
|----------|----------|----------|
| OpenAI (GPT-4.1 / o4-mini) | 5.3 % | 6.6 % |
| Anthropic (Claude Sonnet 4.5) | 11.3 % | 11.3 % |
| Google (Flash-Lite / Flash) | 3.8 % | 5.5 % |
| **Cross-provider average** | **6.8 %** | **7.8 %** |

### Gen2 C5 Hallucination Rates

| Provider | Standard | Thinking |
|----------|----------|----------|
| OpenAI (GPT-5.2) | 3.3 % | 3.3 % |
| Anthropic (Claude Sonnet 4.6) | 13.3 % | 14.9 % |
| Google (Gemini 3.1 Flash-Lite) | 5.9 % | 5.2 % |
| **Cross-provider average** | **7.5 %** | **7.8 %** |

The cross-provider Gen1↔Gen2 averages under the full architecture (C5) differ by less than one percentage point on standard mode (6.8 % → 7.5 %), supporting the **generational-stability** finding across both model generations.

---

## Full Reproduction: Re-Running Experiments

If you want to re-execute the 6,912 API calls (not required for verifying claims):

### Prerequisites

- .NET 8.0 SDK
- PostgreSQL 14+
- API keys for OpenAI, Anthropic, and Google

### Setup

1. Set up the database:
   ```bash
   psql -U postgres -d your_database -f database/schema.sql
   psql -U postgres -d your_database -f database/seed_tools.sql
   ```

2. Configure API keys:
   ```bash
   cd evaluation-harness
   cp appsettings.example.json appsettings.json
   # Edit appsettings.json with your API keys and database connection
   ```

3. Build and run:
   ```bash
   dotnet build
   dotnet run
   ```

See [evaluation-harness/README.md](evaluation-harness/README.md) for detailed instructions.

### Estimated Cost and Time

| Phase | API Calls | Estimated Cost | Estimated Time |
|-------|-----------|----------------|----------------|
| Phase 1 (Gen1 standard) | 2,592 | ~$15 | ~3 hours |
| Phase 2 (Gen1 thinking) | 864 | ~$8 | ~1.5 hours |
| Phase 3a (Gen2 standard) | 2,592 | ~$25 | ~3 hours |
| Phase 3b (Gen2 thinking) | 864 | ~$15 | ~1.5 hours |
| **Total** | **6,912** | **~$63** | **~9 hours (sequential)** |

The CLI mode supports parallel execution per `(provider, category, domain)` to compress wall-clock time.

---

## Analysis Scripts Reference

| Script | Paper Section | Description |
|--------|--------------|-------------|
| `analyze_results.py` | Tables 6–8 | Single-phase ablation analysis |
| `analyze_cross_generational.py` | Tables 9–13, Figs 7–9 | Cross-generational comparison |
| `statistical_analysis.py` | Sections IV–V | Wilcoxon signed-rank, Friedman, effect sizes |
| `classify_hallucinations.py` | Section V.4 | H1/H2/near-miss/non-specific classification |
| `compute_h2_rates.py` | Section V.4 | H2-only operational hallucination rates |
| `compute_reviewer_metrics.py` | Tables 7, 12 | P_any, P_any_H2, E[H/Q] metrics |
| `compute_zero_diff.py` | Section III.7 | Zero-difference pair analysis |
| `sensitivity_analysis.py` | Section V.8 | Detection threshold sensitivity sweep |
| `paired_wilcoxon_check.py` | Section IV.4 | Standard vs thinking Wilcoxon tests |
| `wilcoxon_and_expected_h.py` | Tables 7, 12 | Combined statistical tests |
| `matched_detector_c3.py` | Section 3 (detection), Threats to Validity | Recall-equalized detector: quantifies the C3 detection-format bias (C3-minus-C0 gap under equal recall) |
| `matched_detector_thinking.py` | Results (Gen2 reasoning) | Recall-equalized detector applied to the reasoning-mode responses |
| `analyze_temp_sensitivity.py` | Threats to Validity (temperature confound) | Common-temperature control: Google re-run at temperature 1.0 |
| `analyze_reasoning_c3.py` | Results (Gen2 reasoning) | Reasoning-at-C3 experiment across the three Gen2 providers |
| `corrected_ablation_contrasts.py` | Section 3 (ablation) | Ablation contrasts recomputed under the recall-equalized detector |
| `irr_second_rater.py` | Section 4.2, Threats to Validity | Inter-rater reliability (Cohen's kappa) and consensus decomposition of the 197-name classification |

All scripts read from `../data/results/` and use only Python standard library modules (except `statistical_analysis.py` which uses scipy).

---

## License

This replication package is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Contact

- **Lavdim Menxhiqi** — lavdim.menxhiqi@ubt-uni.net (Technical University of Sofia, Bulgaria)
- **Galia Marinova** — gim@tu-sofia.bg (Technical University of Sofia, Bulgaria)
