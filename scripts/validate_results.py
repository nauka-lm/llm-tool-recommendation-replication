"""
Validate Result File Integrity

Checks that all expected result files exist and contain the correct number
of query results with no errors.

Usage:
    python validate_results.py

Expected output: 192 files, 6912 queries, 0 errors
"""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")

EXPECTED = {
    "phase1": {
        "description": "Gen1 Standard (C0-C5)",
        "providers": ["openai", "anthropic", "google"],
        "category": "standard",
        "configs": ["C0", "C1", "C2", "C3", "C4", "C5"],
        "domains": ["D1", "D2", "D3", "D4"],
        "queries_per_file": 36,  # 12 prompts x 3 repetitions
    },
    "phase2": {
        "description": "Gen1 Thinking (C0, C5)",
        "providers": ["openai", "anthropic", "google"],
        "category": "thinking",
        "configs": ["C0", "C5"],
        "domains": ["D1", "D2", "D3", "D4"],
        "queries_per_file": 36,
    },
    "phase3a": {
        "description": "Gen2 Standard (C0-C5)",
        "providers": ["openai", "anthropic", "google"],
        "category": "standard",
        "configs": ["C0", "C1", "C2", "C3", "C4", "C5"],
        "domains": ["D1", "D2", "D3", "D4"],
        "queries_per_file": 36,
    },
    "phase3b": {
        "description": "Gen2 Thinking (C0, C5)",
        "providers": ["openai", "anthropic", "google"],
        "category": "thinking",
        "configs": ["C0", "C5"],
        "domains": ["D1", "D2", "D3", "D4"],
        "queries_per_file": 36,
    },
}

REQUIRED_FIELDS = [
    "QueryId", "DomainId", "ConfigId", "ProviderId", "ModelId",
    "ModelCategory", "PromptIndex", "Repetition", "RawResponse",
    "MentionedToolNames", "GroundedToolNames", "HallucinatedToolNames",
    "WorkflowStepsCount", "WorkflowStages", "PromptTokens",
    "CompletionTokens", "ThinkingTokens", "LatencyMs", "Error",
]


def validate():
    total_files = 0
    total_queries = 0
    total_errors = 0
    missing_files = []
    issues = []

    for phase, spec in EXPECTED.items():
        phase_dir = os.path.join(RESULTS_DIR, phase)
        phase_files = 0
        phase_queries = 0
        phase_errors = 0

        print(f"\n{'='*60}")
        print(f"Phase: {phase} - {spec['description']}")
        print(f"{'='*60}")

        if not os.path.exists(phase_dir):
            print(f"  ERROR: Directory not found: {phase_dir}")
            expected_count = (len(spec["providers"]) * len(spec["configs"]) *
                              len(spec["domains"]))
            missing_files.extend([f"{phase}/*"] * expected_count)
            continue

        for provider in spec["providers"]:
            for config in spec["configs"]:
                for domain in spec["domains"]:
                    filename = f"results_{provider}_{spec['category']}_{domain}_{config}.json"
                    filepath = os.path.join(phase_dir, filename)

                    if not os.path.exists(filepath):
                        missing_files.append(f"{phase}/{filename}")
                        print(f"  MISSING: {filename}")
                        continue

                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            results = json.load(f)
                    except json.JSONDecodeError as e:
                        issues.append(f"{phase}/{filename}: Invalid JSON - {e}")
                        print(f"  INVALID JSON: {filename}")
                        continue

                    phase_files += 1
                    query_count = len(results)
                    phase_queries += query_count

                    # Check query count
                    if query_count != spec["queries_per_file"]:
                        issues.append(
                            f"{phase}/{filename}: Expected {spec['queries_per_file']} "
                            f"queries, found {query_count}"
                        )

                    # Check for errors and required fields
                    for i, result in enumerate(results):
                        if result.get("Error"):
                            phase_errors += 1
                            issues.append(
                                f"{phase}/{filename}[{i}]: Error - {result['Error'][:100]}"
                            )

                        for field in REQUIRED_FIELDS:
                            if field not in result:
                                issues.append(
                                    f"{phase}/{filename}[{i}]: Missing field '{field}'"
                                )

        total_files += phase_files
        total_queries += phase_queries
        total_errors += phase_errors
        print(f"  Files: {phase_files}")
        print(f"  Queries: {phase_queries}")
        print(f"  Errors: {phase_errors}")

    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files found: {total_files}")
    print(f"Total queries: {total_queries}")
    print(f"Total errors: {total_errors}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Issues found: {len(issues)}")

    if missing_files:
        print(f"\nMissing files ({len(missing_files)}):")
        for f in missing_files[:20]:
            print(f"  - {f}")
        if len(missing_files) > 20:
            print(f"  ... and {len(missing_files) - 20} more")

    if issues:
        print(f"\nIssues ({len(issues)}):")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")

    # Final verdict
    if total_files == 192 and total_queries == 6912 and total_errors == 0 and not issues:
        print(f"\n  PASS: All 192 files present, 6912 queries valid, 0 errors")
        return 0
    else:
        print(f"\n  ISSUES DETECTED: Review above")
        return 1


if __name__ == "__main__":
    sys.exit(validate())
