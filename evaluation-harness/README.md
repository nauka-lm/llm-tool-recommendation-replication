# Evaluation Harness

This directory contains the .NET 8 evaluation framework used to execute all 6,912 API calls across 3 LLM providers, 2 model generations, and 6 ablation configurations.

**Note**: You do NOT need to re-run experiments to verify the paper's claims. All raw results are provided in `data/results/`. This code is provided for full transparency and reproducibility.

## Architecture

```
evaluation-harness/
|-- Program.cs                    # Interactive menu + CLI argument mode
|-- Clients/
|   |-- ILlmClient.cs             # Unified LLM interface
|   |-- OpenAiLlmClient.cs        # OpenAI SDK (GPT-4.1, o4-mini)
|   |-- OpenAiHttpLlmClient.cs    # OpenAI HTTP (GPT-5.2, reasoning_effort)
|   |-- AnthropicLlmClient.cs     # Anthropic HTTP (Sonnet 4.5, Sonnet 4.6)
|   |-- GoogleLlmClient.cs        # Google HTTP (Flash-Lite 2.5, Flash 2.5, Flash-Lite 3.1)
|-- Services/
|   |-- EvaluationRunner.cs       # Core experiment orchestration
|   |-- PromptBuilder.cs          # C0-C5 ablation prompt construction
|   |-- HallucinationDetector.cs  # Tool name extraction + grounding check
|   |-- MetricsCalculator.cs      # HR, GR, WC, RC computation
|   |-- TestPromptGenerator.cs    # 12 domain-specific test queries
|   |-- DatabaseService.cs        # PostgreSQL tool inventory access
|-- Models/
    |-- AblationConfig.cs         # C0-C5 configuration definitions
    |-- EvaluationResult.cs       # QueryResult + MetricsResult schemas
    |-- Domain.cs                 # Domain, ToolInfo, RelatedToolInfo
```

## Prerequisites

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- [PostgreSQL 14+](https://www.postgresql.org/download/)
- API keys for:
  - [OpenAI](https://platform.openai.com/) (GPT-4.1, o4-mini, GPT-5.2)
  - [Anthropic](https://www.anthropic.com/) (Claude Sonnet 4.5, Claude Sonnet 4.6)
  - [Google AI](https://ai.google.dev/) (Gemini 2.5 Flash-Lite, Flash 2.5, Gemini 3.1 Flash-Lite)

## Setup

### 1. Database Setup

```bash
# Create the database and populate tool inventory
psql -U postgres -c "CREATE DATABASE CadcomOnline;"
psql -U postgres -d CadcomOnline -f ../database/schema.sql
psql -U postgres -d CadcomOnline -f ../database/seed_tools.sql
```

### 2. Configuration

```bash
cp appsettings.example.json appsettings.json
```

Edit `appsettings.json` with your credentials:

```json
{
  "ConnectionStrings": {
    "DBConnection": "Host=localhost;Port=5432;Database=CadcomOnline;Username=postgres;Password=YOUR_PASSWORD"
  },
  "OpenAI": { "ApiKey": "sk-proj-..." },
  "Anthropic": { "ApiKey": "sk-ant-api03-..." },
  "Google": { "ApiKey": "AIza..." }
}
```

### 3. Build and Run

```bash
dotnet build
dotnet run
```

The interactive menu offers options for:
- Pilot runs (subset of models, ~$25)
- Individual phases (Gen1/Gen2, standard/thinking)
- Full experiment (all 6,912 calls)
- Custom experiments (select specific models, configs, domains)

### CLI Mode (for parallel execution)

```bash
dotnet run -- --provider openai --category standard --configs C0,C1,C2,C3,C4,C5 --domains D1,D2,D3,D4 --output phase1 --reps 3 --generation gen1
```

## Key Implementation Details

### Prompt Construction (PromptBuilder.cs)

The 6 ablation configurations progressively enable 3 anti-hallucination mechanisms:

- **M1 (Context Builder)**: Provides ranked tool metadata from the database, including features, capabilities, and related tools from a 6-stage context builder
- **M2 (Closed Vocabulary)**: Constrains the model to only use tool names from the database whitelist
- **M3 (JSON Enforcement)**: Requires structured JSON output with specific schema

### Hallucination Detection (HallucinationDetector.cs)

Three-strategy detection:
1. **JSON extraction**: Parses `toolName` and `tool` fields from structured responses
2. **Known tools lookup**: Searches response text for all 82 database tool names
3. **Software product pattern**: Regex-based extraction of commercial tool-like names

Grounding classification uses:
- Exact match (case-insensitive)
- Containment match (>= 55% length ratio)
- Levenshtein distance <= 2 (for tools > 4 characters)

### Error Resilience

- Results are saved incrementally (every 10 API calls)
- Experiments can resume from partial results on failure
- Per-provider rate limiting (OpenAI: 1s, Anthropic: 2s, Google: 5s between calls)

## Important Notes

- **This project references the main CadcomOnline project** for its Entity Framework Core DataContext and domain models. The standalone version provided here requires the database schema and seed data from `database/`.
- **API costs**: A full run of all 6,912 calls costs approximately $53 across all providers.
- **Runtime**: Approximately 9 hours for the complete experiment (with rate limiting).
- **Model availability**: Some models (especially thinking/reasoning variants) may have changed API parameters since the experiments were conducted in February 2026.
