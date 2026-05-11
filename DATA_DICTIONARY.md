# Data Dictionary

This document describes the structure and fields of the experiment result JSON files in `data/results/`.

## File Organization

Each phase directory contains JSON result files following this naming convention:

```
results_{provider}_{category}_{domain}_{config}.json
```

Where:
- `provider`: `openai`, `anthropic`, or `google`
- `category`: `standard` or `thinking`
- `domain`: `D1`, `D2`, `D3`, or `D4`
- `config`: `C0`, `C1`, `C2`, `C3`, `C4`, or `C5`

### Phase Directories

| Directory | Phase | Models | Configs | Files | Queries |
|-----------|-------|--------|---------|-------|---------|
| `phase1/` | Gen1 Standard | GPT-4.1, Sonnet 4.5, Flash-Lite | C0-C5 | 72 | 2,592 |
| `phase2/` | Gen1 Thinking | o4-mini, Sonnet 4.5 (thinking), Flash | C0, C5 | 24 | 864 |
| `phase3a/` | Gen2 Standard | GPT-5.2, Sonnet 4.6, Gemini 3.1 Flash-Lite | C0-C5 | 72 | 2,592 |
| `phase3b/` | Gen2 Thinking | GPT-5.2 (high), Sonnet 4.6 (adaptive), Flash-Lite 3.1 (thinking) | C0, C5 | 24 | 864 |

## JSON Schema

Each file contains a JSON array of query result objects. Each object has the following fields:

### Identification Fields

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `QueryId` | string | Unique identifier for this query | `"openai_standard_D1_C0_P01_R1"` |
| `DomainId` | string | Evaluation domain identifier | `"D1"`, `"D2"`, `"D3"`, `"D4"` |
| `ConfigId` | string | Ablation configuration identifier | `"C0"` through `"C5"` |
| `ProviderId` | string | LLM provider name | `"openai"`, `"anthropic"`, `"google"` |
| `ModelId` | string | Specific model version | `"gpt-4.1"`, `"claude-sonnet-4-5-20250929"`, `"gemini-2.5-flash-lite"` |
| `ModelCategory` | string | Standard or thinking mode | `"standard"`, `"thinking"` |
| `PromptIndex` | integer | Zero-based test prompt index (0-11) | `0` through `11` |
| `Repetition` | integer | Repetition number (1-based) | `1`, `2`, `3` |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `RawResponse` | string | Complete unmodified LLM response text |
| `IsValidJson` | boolean | Whether the response parsed as valid JSON |

### Hallucination Analysis Fields

| Field | Type | Description |
|-------|------|-------------|
| `MentionedToolNames` | string[] | All tool names extracted from the response |
| `GroundedToolNames` | string[] | Tools that exist in the database (correct recommendations) |
| `HallucinatedToolNames` | string[] | Tools NOT found in the database (hallucinations) |

### Workflow Analysis Fields

| Field | Type | Description |
|-------|------|-------------|
| `WorkflowStepsCount` | integer | Number of distinct workflow stages identified (0-6) |
| `WorkflowStages` | string[] | Names of detected stages: `"Selection"`, `"Simulation"`, `"Calculation"`, `"Verification"`, `"Manufacturing"`, `"Prototyping"` |

### Performance Fields

| Field | Type | Description |
|-------|------|-------------|
| `PromptTokens` | integer | Number of tokens in the prompt sent to the model |
| `CompletionTokens` | integer | Number of tokens in the model's visible response |
| `ThinkingTokens` | integer | Reasoning/thinking tokens (0 for standard models) |
| `LatencyMs` | number | Round-trip API call latency in milliseconds |

### Error Field

| Field | Type | Description |
|-------|------|-------------|
| `Error` | string or null | Error message if the API call failed; `null` on success |

## Computed Metrics

The following metrics are derived from the raw fields above:

### Hallucination Rate (HR)

```
HR = |HallucinatedToolNames| / |MentionedToolNames| * 100
```

Where `|...|` denotes set cardinality. A value of 0% means all mentioned tools exist in the database.

### Grounding Rate (GR)

```
GR = 100 - HR
```

### Workflow Coverage (WC)

```
WC = |WorkflowStages| / 6 * 100
```

The 6 possible workflow stages are: Selection, Simulation, Calculation, Verification, Manufacturing, Prototyping.

### Response Consistency (RC)

```
RC = mean(Jaccard(GroundedTools_i, GroundedTools_j)) for all repetition pairs (i,j)
```

Jaccard similarity of grounded tool sets across repetitions of the same prompt.

## Tool Inventory (Ground Truth)

The ground truth for hallucination detection is the set of 82 engineering tools stored in the database. See `data/tool-inventory/tool_inventory.json` for the complete list.

### Grounding Classification Logic

A mentioned tool name is classified as **grounded** if any of:
1. **Exact match** (case-insensitive) with a database tool name
2. **Containment match**: Database tool name appears within the mentioned name (or vice versa) with >= 55% length ratio
3. **Levenshtein distance** <= 2 (for tools with names > 4 characters)

Otherwise, the tool is classified as **hallucinated**.

## Domains

| Domain ID | Name | Tool Count | Role |
|-----------|------|------------|------|
| D1 | PCB Design Tool Selection | 8 | Ablation + Cross-validation |
| D2 | PCB Design Calculators | 15 | Ablation + Cross-validation |
| D3 | SMPS Design | 8 | Ablation + Cross-validation |
| D4 | Transformer Design | 10 | Ablation + Cross-validation |

**Note**: In the ICISE 2026 conference paper, D1/D2 were cross-validation only. In this journal extension, all 4 domains undergo full ablation (C0-C5) for all providers.

## Sample Data

A single query result object:

```json
{
  "QueryId": "openai_standard_D3_C5_P01_R1",
  "DomainId": "D3",
  "ConfigId": "C5",
  "ProviderId": "openai",
  "ModelId": "gpt-4.1",
  "ModelCategory": "standard",
  "PromptIndex": 0,
  "Repetition": 1,
  "RawResponse": "{ \"recommendations\": [...] }",
  "IsValidJson": true,
  "MentionedToolNames": ["LTspice", "PSIM", "PLECS"],
  "GroundedToolNames": ["LTspice", "PSIM", "PLECS"],
  "HallucinatedToolNames": [],
  "WorkflowStepsCount": 4,
  "WorkflowStages": ["Selection", "Simulation", "Calculation", "Verification"],
  "PromptTokens": 2847,
  "CompletionTokens": 983,
  "ThinkingTokens": 0,
  "LatencyMs": 4521.3,
  "Error": null
}
```
