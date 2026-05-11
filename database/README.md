# Database Setup

This directory contains the PostgreSQL schema for the engineering tool inventory used as ground truth in the evaluation.

## Files

| File | Description |
|------|-------------|
| `schema.sql` | Table definitions, indexes, and comments |
| `seed_tools.sql` | Tool inventory data (82 tools with metadata) |

## Quick Setup

```bash
# Create database
psql -U postgres -c "CREATE DATABASE CadcomOnline;"

# Apply schema
psql -U postgres -d CadcomOnline -f schema.sql

# Load tool inventory
psql -U postgres -d CadcomOnline -f seed_tools.sql
```

## Domain Mapping

The evaluation framework maps 4 domains to subcategory names:

| Domain | Subcategory Name | Tools |
|--------|------------------|-------|
| D1 | PCB Design Tool | 8 |
| D2 | PCB Design Calculator | 15 |
| D3 | Switched-mode power supplies (SMPS), converters, regulators | 8 |
| D4 | Transformers | 10 |

**Note**: The remaining tools (41 total) belong to other subcategories and are included in the global tool name set used for hallucination detection but are not directly queried as primary domain tools.

## Data Model

```
Panel (top-level)
  └── Category
        └── Subcategory (= evaluation domain)
              └── ToolPassport (= engineering tool)
                    └── ToolFeature (criteria-value pairs)
```

Each tool has:
- **Name**: Canonical tool name (ground truth for hallucination detection)
- **Function**: What the tool does
- **AbstractionLevel**: Level of design abstraction
- **IOConnection**: Compatible tools for data exchange
- **VerificationTool**: Tools for verifying this tool's output
- **ApplicationArea**: Domain coverage and equivalent tools
- **Features**: Criteria-value pairs (e.g., "Board Size: Up to 1m x 1m")

## Important Notes

- All tables use **soft delete** (`IsDeleted` + `IsActive` flags)
- The evaluation harness only queries tools where `IsDeleted = false AND IsActive = true`
- Tool names are matched **case-insensitively** with fuzzy matching (Levenshtein distance <= 2)
- The `seed_tools.sql` file should be generated from the live database using `pg_dump`
