"""
Analysis Script for the journal Cross-Model Evaluation Results
Processes JSON result files and generates tables/statistics for the paper.

Usage:
    python analyze_results.py <results_directory>
    python analyze_results.py ../data/results/phase1
"""

import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

def load_results(results_dir):
    """Load all JSON result files from a directory."""
    results = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "results_*.json"))):
        name = os.path.basename(f).replace("results_", "").replace(".json", "")
        with open(f) as fh:
            data = json.load(fh)
        results[name] = data
    return results


def compute_metrics(queries):
    """Compute metrics from a list of query results."""
    valid = [q for q in queries if not q.get("Error")]
    if not valid:
        return None

    total_tools = sum(len(q.get("MentionedToolNames", [])) for q in valid)
    total_halluc = sum(len(q.get("HallucinatedToolNames", [])) for q in valid)
    total_grounded = sum(len(q.get("GroundedToolNames", [])) for q in valid)

    hr = (total_halluc / total_tools * 100) if total_tools > 0 else 0
    gr = (total_grounded / total_tools * 100) if total_tools > 0 else 0

    # Workflow coverage
    all_stages = {"Selection", "Simulation", "Calculation", "Verification", "Manufacturing", "Prototyping"}
    avg_wc = sum(
        len(set(q.get("WorkflowStages", [])) & all_stages) / len(all_stages)
        for q in valid
    ) / len(valid) * 100

    # Average tokens and latency
    avg_input = sum(q.get("PromptTokens", 0) for q in valid) / len(valid)
    avg_output = sum(q.get("CompletionTokens", 0) for q in valid) / len(valid)
    avg_think = sum(q.get("ThinkingTokens", 0) for q in valid) / len(valid)
    avg_latency = sum(q.get("LatencyMs", 0) for q in valid) / len(valid)
    avg_tools = total_tools / len(valid)

    # JSON validity rate
    json_valid = sum(1 for q in valid if q.get("IsValidJson", False))
    json_rate = json_valid / len(valid) * 100

    return {
        "valid_queries": len(valid),
        "errors": len(queries) - len(valid),
        "total_tools": total_tools,
        "total_halluc": total_halluc,
        "total_grounded": total_grounded,
        "hr": hr,
        "gr": gr,
        "wc": avg_wc,
        "avg_tools_per_query": avg_tools,
        "avg_input_tokens": avg_input,
        "avg_output_tokens": avg_output,
        "avg_thinking_tokens": avg_think,
        "avg_latency_ms": avg_latency,
        "json_valid_rate": json_rate,
    }


def parse_key(key):
    """Parse result file key like 'openai_standard_D1_C0' into components."""
    parts = key.split("_")
    provider = parts[0]
    category = parts[1]
    domain = parts[2]
    config = parts[3]
    return provider, category, domain, config


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def main(results_dir):
    results = load_results(results_dir)

    if not results:
        print(f"No result files found in {results_dir}")
        return

    print_section("LOADED RESULT FILES")
    for key in sorted(results.keys()):
        n = len(results[key])
        err = sum(1 for q in results[key] if q.get("Error"))
        print(f"  {key:45s} | {n} queries ({err} errors)")

    # Compute metrics per file
    metrics = {}
    for key, queries in results.items():
        m = compute_metrics(queries)
        if m:
            metrics[key] = m

    # =====================================================================
    # TABLE 1: C0 Baseline Hallucination Rates
    # =====================================================================
    print_section("TABLE: C0 BASELINE HALLUCINATION RATES")
    print(f"{'Model':<30s} {'D1':>8s} {'D2':>8s} {'D3':>8s} {'D4':>8s} {'Avg':>8s}")
    print("-" * 72)

    models = set()
    for key in metrics:
        provider, category, domain, config = parse_key(key)
        models.add(f"{provider}_{category}")

    for model in sorted(models):
        provider, category = model.split("_", 1)
        row = f"{provider} ({category})"
        hrs = []
        for d in ["D1", "D2", "D3", "D4"]:
            key = f"{model}_{d}_C0"
            if key in metrics:
                hr = metrics[key]["hr"]
                hrs.append(hr)
                row += f"  {hr:6.1f}%"
            else:
                row += f"     ---"
        if hrs:
            avg = sum(hrs) / len(hrs)
            row += f"  {avg:6.1f}%"
        else:
            row += f"     ---"
        print(f"  {row}")

    # =====================================================================
    # TABLE 2: C5 Full Architecture Results
    # =====================================================================
    print_section("TABLE: C5 FULL ARCHITECTURE RESULTS")
    print(f"{'Model':<30s} {'D1':>8s} {'D2':>8s} {'D3':>8s} {'D4':>8s} {'Avg':>8s} {'Reduc.':>8s}")
    print("-" * 80)

    for model in sorted(models):
        provider, category = model.split("_", 1)
        row = f"{provider} ({category})"
        hrs_c5 = []
        hrs_c0 = []
        for d in ["D1", "D2", "D3", "D4"]:
            key_c5 = f"{model}_{d}_C5"
            key_c0 = f"{model}_{d}_C0"
            if key_c5 in metrics:
                hr = metrics[key_c5]["hr"]
                hrs_c5.append(hr)
                row += f"  {hr:6.1f}%"
            else:
                row += f"     ---"
            if key_c0 in metrics:
                hrs_c0.append(metrics[key_c0]["hr"])

        if hrs_c5:
            avg_c5 = sum(hrs_c5) / len(hrs_c5)
            row += f"  {avg_c5:6.1f}%"
        else:
            avg_c5 = None
            row += f"     ---"

        if hrs_c0 and avg_c5 is not None:
            avg_c0 = sum(hrs_c0) / len(hrs_c0)
            reduction = ((avg_c0 - avg_c5) / avg_c0) * 100 if avg_c0 > 0 else 0
            row += f"  {reduction:5.1f}%"
        else:
            row += f"     ---"
        print(f"  {row}")

    # =====================================================================
    # TABLE 3: Response Characteristics
    # =====================================================================
    print_section("TABLE: RESPONSE CHARACTERISTICS (C5)")
    print(f"{'Model':<30s} {'Tools/Q':>8s} {'OutTok':>8s} {'Latency':>10s} {'ThinkTok':>10s} {'JSON%':>8s}")
    print("-" * 84)

    for model in sorted(models):
        c5_metrics = [m for k, m in metrics.items() if k.startswith(model) and "_C5" in k]
        if not c5_metrics:
            continue

        avg_tools = sum(m["avg_tools_per_query"] for m in c5_metrics) / len(c5_metrics)
        avg_out = sum(m["avg_output_tokens"] for m in c5_metrics) / len(c5_metrics)
        avg_lat = sum(m["avg_latency_ms"] for m in c5_metrics) / len(c5_metrics)
        avg_think = sum(m["avg_thinking_tokens"] for m in c5_metrics) / len(c5_metrics)
        avg_json = sum(m["json_valid_rate"] for m in c5_metrics) / len(c5_metrics)

        provider, category = model.split("_", 1)
        print(f"  {provider} ({category})"
              f"  {avg_tools:6.1f}"
              f"  {avg_out:6.0f}"
              f"  {avg_lat/1000:8.1f}s"
              f"  {avg_think:8.0f}"
              f"  {avg_json:6.1f}%")

    # =====================================================================
    # TABLE 4: Ablation Study (if C1-C4 data available)
    # =====================================================================
    configs_available = set()
    for key in metrics:
        _, _, _, config = parse_key(key)
        configs_available.add(config)

    if len(configs_available) > 2:  # More than just C0 and C5
        print_section("TABLE: ABLATION STUDY")
        for model in sorted(models):
            provider, category = model.split("_", 1)
            print(f"\n  {provider} ({category}):")
            print(f"  {'Config':<25s} {'Avg HR%':>8s} {'Delta':>8s} {'Reduction':>10s}")
            print(f"  {'-' * 55}")

            baseline = None
            for cfg in ["C0", "C1", "C2", "C3", "C4", "C5"]:
                cfg_metrics = [m for k, m in metrics.items()
                              if k.startswith(model) and k.endswith(f"_{cfg}")]
                if not cfg_metrics:
                    continue

                avg_hr = sum(m["hr"] for m in cfg_metrics) / len(cfg_metrics)
                if cfg == "C0":
                    baseline = avg_hr

                delta = f"{avg_hr - baseline:+.1f}pp" if baseline is not None else "---"
                reduction = f"{((baseline - avg_hr) / baseline * 100):.1f}%" if baseline and baseline > 0 else "---"

                print(f"  {cfg:<25s} {avg_hr:7.1f}%  {delta:>8s} {reduction:>10s}")

    # =====================================================================
    # TABLE 5: Standard vs Reasoning Comparison
    # =====================================================================
    providers_with_both = set()
    for model in models:
        provider, category = model.split("_", 1)
        has_standard = any(k.startswith(f"{provider}_standard") for k in metrics)
        has_thinking = any(k.startswith(f"{provider}_thinking") for k in metrics)
        if has_standard and has_thinking:
            providers_with_both.add(provider)

    if providers_with_both:
        print_section("TABLE: STANDARD vs REASONING COMPARISON")
        print(f"{'Provider':<15s} {'Mode':<12s} {'C0 HR%':>8s} {'C5 HR%':>8s} {'Reduction':>10s}")
        print("-" * 60)

        for provider in sorted(providers_with_both):
            for category in ["standard", "thinking"]:
                c0_metrics = [m for k, m in metrics.items()
                             if k.startswith(f"{provider}_{category}") and "_C0" in k]
                c5_metrics = [m for k, m in metrics.items()
                             if k.startswith(f"{provider}_{category}") and "_C5" in k]

                c0_hr = sum(m["hr"] for m in c0_metrics) / len(c0_metrics) if c0_metrics else None
                c5_hr = sum(m["hr"] for m in c5_metrics) / len(c5_metrics) if c5_metrics else None

                c0_str = f"{c0_hr:6.1f}%" if c0_hr is not None else "   ---"
                c5_str = f"{c5_hr:6.1f}%" if c5_hr is not None else "   ---"
                if c0_hr and c5_hr:
                    red = ((c0_hr - c5_hr) / c0_hr) * 100
                    red_str = f"{red:6.1f}%"
                else:
                    red_str = "   ---"

                print(f"  {provider:<15s} {category:<12s} {c0_str:>8s} {c5_str:>8s} {red_str:>10s}")

    # =====================================================================
    # SUMMARY STATISTICS
    # =====================================================================
    print_section("SUMMARY STATISTICS")
    total_queries = sum(m["valid_queries"] for m in metrics.values())
    total_errors = sum(m["errors"] for m in metrics.values())
    total_tools_mentioned = sum(m["total_tools"] for m in metrics.values())
    total_hallucinated = sum(m["total_halluc"] for m in metrics.values())

    print(f"  Total valid queries:     {total_queries}")
    print(f"  Total errors:            {total_errors}")
    print(f"  Total tools mentioned:   {total_tools_mentioned}")
    print(f"  Total hallucinated:      {total_hallucinated}")
    print(f"  Overall HR:              {total_hallucinated/total_tools_mentioned*100:.1f}%")
    print(f"  Result files:            {len(results)}")
    print(f"  Unique model configs:    {len(models)}")
    print(f"  Configs tested:          {sorted(configs_available)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to phase1 directory
        results_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "results", "phase1"
        )
    else:
        results_dir = sys.argv[1]

    print(f"Analyzing results from: {results_dir}")
    main(results_dir)
