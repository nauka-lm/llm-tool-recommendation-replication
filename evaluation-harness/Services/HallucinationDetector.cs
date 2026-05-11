using System.Text.Json;
using System.Text.RegularExpressions;
using CadcomOnline.Evaluation.Models;

namespace CadcomOnline.Evaluation.Services;

/// <summary>
/// Detects hallucinated tool names in LLM responses by comparing against the database.
///
/// Approach:
/// 1. For JSON responses: extract values from "toolName" and "tool" fields only
/// 2. For all responses: check if any of the 82 known DB tool names appear in text
/// 3. For all responses: detect software-product-like names not in DB (hallucinations)
///
/// A tool name is "grounded" if it matches a DB entry (exact or fuzzy).
/// A tool name is "hallucinated" if it looks like a software product but isn't in DB.
/// </summary>
public class HallucinationDetector
{
    private readonly HashSet<string> _allToolNames;
    private readonly List<string> _allToolNamesList;

    public HallucinationDetector(HashSet<string> allToolNames)
    {
        _allToolNames = allToolNames;
        _allToolNamesList = allToolNames.ToList();
    }

    /// <summary>
    /// Analyzes a raw LLM response to extract tool names and classify them.
    /// </summary>
    public AnalysisResult Analyze(string rawResponse, Domain domain)
    {
        var result = new AnalysisResult();
        result.IsValidJson = TryParseJson(rawResponse);

        // Collect all candidate tool names from different extraction strategies
        var candidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Strategy 1: Extract from JSON "tool" / "toolName" fields (high confidence)
        if (result.IsValidJson)
            ExtractToolNamesFromJson(rawResponse, candidates);

        // Strategy 2: Find all known DB tool names mentioned in the response
        foreach (var dbName in _allToolNamesList)
        {
            if (rawResponse.Contains(dbName, StringComparison.OrdinalIgnoreCase))
                candidates.Add(dbName);
        }

        // Strategy 3: Detect software-product-like names (potential hallucinations)
        ExtractSoftwareProductNames(rawResponse, candidates);

        // Classify each candidate
        result.MentionedToolNames = candidates.ToList();
        foreach (var name in candidates)
        {
            if (IsGrounded(name))
                result.GroundedToolNames.Add(name);
            else
                result.HallucinatedToolNames.Add(name);
        }

        // Detect workflow stages
        result.WorkflowStages = DetectWorkflowStages(rawResponse);

        return result;
    }

    /// <summary>
    /// Checks if a tool name is grounded (exists in the database).
    /// Uses exact match, then containment with length ratio, then Levenshtein.
    /// </summary>
    private bool IsGrounded(string toolName)
    {
        // Exact match (case-insensitive)
        if (_allToolNames.Contains(toolName))
            return true;

        foreach (var dbName in _allToolNamesList)
        {
            // Containment check (e.g., "KiCad EDA" contains "KiCad")
            if (toolName.Contains(dbName, StringComparison.OrdinalIgnoreCase) ||
                dbName.Contains(toolName, StringComparison.OrdinalIgnoreCase))
            {
                var shorter = Math.Min(toolName.Length, dbName.Length);
                var longer = Math.Max(toolName.Length, dbName.Length);
                if ((double)shorter / longer >= 0.55)
                    return true;
            }

            // Levenshtein distance for typos (e.g., "KiCaad" → "KiCad")
            if (toolName.Length > 4 && dbName.Length > 4 &&
                LevenshteinDistance(toolName.ToLower(), dbName.ToLower()) <= 2)
                return true;
        }

        return false;
    }

    /// <summary>
    /// Extracts tool names from JSON "toolName" and "tool" fields only.
    /// Does NOT extract from "phase", "step", "action", "verdict", etc.
    /// </summary>
    private void ExtractToolNamesFromJson(string json, HashSet<string> toolNames)
    {
        try
        {
            var cleanJson = CleanJsonString(json);
            using var doc = JsonDocument.Parse(cleanJson);
            ExtractToolFieldsFromElement(doc.RootElement, toolNames);
        }
        catch { /* JSON parsing failed */ }
    }

    private void ExtractToolFieldsFromElement(JsonElement element, HashSet<string> toolNames)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                foreach (var prop in element.EnumerateObject())
                {
                    var propLower = prop.Name.ToLower();
                    // Only extract from fields that specifically hold tool names
                    if (propLower is "toolname" or "tool" ||
                        (propLower == "name" && IsInsideToolContext(prop)))
                    {
                        if (prop.Value.ValueKind == JsonValueKind.String)
                        {
                            var value = prop.Value.GetString()?.Trim();
                            if (!string.IsNullOrWhiteSpace(value) && value.Length >= 3 &&
                                !IsJsonMetaValue(value))
                            {
                                toolNames.Add(value);
                            }
                        }
                    }
                    // Recurse into nested objects/arrays
                    ExtractToolFieldsFromElement(prop.Value, toolNames);
                }
                break;

            case JsonValueKind.Array:
                foreach (var item in element.EnumerateArray())
                    ExtractToolFieldsFromElement(item, toolNames);
                break;
        }
    }

    /// <summary>
    /// Detects software product-like names that aren't in the DB.
    /// Targets names like "Altium Designer", "OrCAD PSpice", "PCBMaster Pro", etc.
    /// These are multi-word proper nouns with software-related suffixes/patterns.
    /// </summary>
    private void ExtractSoftwareProductNames(string response, HashSet<string> candidates)
    {
        // Pattern 1: Names ending with software suffixes
        // e.g., "Altium Designer", "PCBMaster Pro", "EasyEDA Pro Designer"
        var suffixPattern = new Regex(
            @"\b([A-Z][a-zA-Z0-9]+(?:[\s\-][A-Z][a-zA-Z0-9]+)*\s+(?:Designer|Pro|Suite|Studio|Toolkit|Calculator|Simulator|Viewer|Editor|Manager|Analyzer|Planner|Solver|Modeler|Builder))\b",
            RegexOptions.Compiled);

        foreach (Match match in suffixPattern.Matches(response))
        {
            var candidate = match.Groups[1].Value.Trim();
            if (!IsGenericPhrase(candidate))
                candidates.Add(candidate);
        }

        // Pattern 2: Names with version numbers or trademark-like patterns
        // e.g., "OrCAD 17.4", "Cadence Allegro"
        var versionPattern = new Regex(
            @"\b([A-Z][a-zA-Z0-9]+(?:[\s\-][A-Z][a-zA-Z0-9]+)*)\s+(?:v?\d+[\.\d]*|[IVXL]+)\b",
            RegexOptions.Compiled);

        foreach (Match match in versionPattern.Matches(response))
        {
            var candidate = match.Groups[1].Value.Trim();
            if (candidate.Length >= 4 && !IsGenericPhrase(candidate))
                candidates.Add(candidate);
        }

        // Pattern 3: Known commercial tool name patterns not in our DB
        // Specifically target well-known commercial EDA tools that would be hallucinations
        var commercialToolPatterns = new[]
        {
            @"\b(Altium\s+Designer)\b",
            @"\b(OrCAD(?:\s+\w+)?)\b",
            @"\b(Cadence\s+\w+)\b",
            @"\b(Mentor\s+Graphics?\s*\w*)\b",
            @"\b(Zuken\s+\w+)\b",
            @"\b(Proteus\s+\w+)\b",
            @"\b(Multisim(?:\s+\w+)?)\b",
            @"\b(LTspice(?:\s+\w+)?)\b",
            @"\b(PSpice(?:\s+\w+)?)\b",
            @"\b(Fusion\s+360)\b",
            @"\b(SolidWorks(?:\s+\w+)?)\b",
            @"\b(AutoCAD(?:\s+\w+)?)\b",
        };

        foreach (var pattern in commercialToolPatterns)
        {
            foreach (Match match in Regex.Matches(response, pattern, RegexOptions.IgnoreCase))
            {
                var candidate = match.Groups[1].Value.Trim();
                if (candidate.Length >= 4)
                    candidates.Add(candidate);
            }
        }
    }

    /// <summary>
    /// Detects which of the 6 workflow stages are covered in the response.
    /// </summary>
    private List<string> DetectWorkflowStages(string response)
    {
        var stages = new List<string>();
        var lower = response.ToLower();

        if (lower.Contains("recommend") || lower.Contains("select") || lower.Contains("choose"))
            stages.Add("Selection");
        if (lower.Contains("simulat") || lower.Contains("spice") ||
            Regex.IsMatch(lower, @"\bmodel(?:ing|led)\b"))
            stages.Add("Simulation");
        if (lower.Contains("calculat") || lower.Contains("analys") ||
            lower.Contains("impedance") || lower.Contains("thermal") || lower.Contains("trace width"))
            stages.Add("Calculation");
        if (lower.Contains("verif") || lower.Contains("drc") ||
            lower.Contains("design rule") || lower.Contains("checking"))
            stages.Add("Verification");
        if (lower.Contains("manufactur") || lower.Contains("gerber") ||
            lower.Contains("fabricat") || lower.Contains("production") ||
            Regex.IsMatch(lower, @"\bbom\b"))
            stages.Add("Manufacturing");
        if (lower.Contains("prototyp") || lower.Contains("fpga") ||
            lower.Contains("arduino") || lower.Contains("usrp"))
            stages.Add("Prototyping");

        return stages;
    }

    // ─── Helper methods ─────────────────────────────────────────────────────

    private static string CleanJsonString(string json)
    {
        var clean = json.Trim();
        if (clean.StartsWith("```"))
        {
            var nl = clean.IndexOf('\n');
            if (nl >= 0) clean = clean[(nl + 1)..];
            if (clean.EndsWith("```")) clean = clean[..^3];
            clean = clean.Trim();
        }
        return clean;
    }

    private static bool TryParseJson(string text)
    {
        try { JsonDocument.Parse(CleanJsonString(text)); return true; }
        catch { return false; }
    }

    /// <summary>
    /// Checks if a JSON property "name" is inside a tool-related context
    /// (heuristic: assume it is if the value doesn't look like a generic description).
    /// </summary>
    private static bool IsInsideToolContext(JsonProperty prop)
    {
        // "name" fields are ambiguous - skip them to avoid false positives
        // Tool names come through "tool" and "toolName" fields
        return false;
    }

    /// <summary>
    /// Filters out values that are JSON schema meta-descriptions, not actual tool names.
    /// e.g., "MUST be from: [...]", "Tool from PRIMARY TOOLS list"
    /// </summary>
    private static bool IsJsonMetaValue(string value)
    {
        var lower = value.ToLower();
        return lower.Contains("must be from") ||
               lower.Contains("from primary") ||
               lower.Contains("from database") ||
               lower.Contains("from list") ||
               lower.Contains("tool name") ||
               lower.StartsWith("the ") ||
               lower.StartsWith("a ") ||
               lower.StartsWith("an ") ||
               value.Length > 100; // Descriptions, not names
    }

    /// <summary>
    /// Filters out generic phrases that match the regex but aren't tool names.
    /// </summary>
    private static bool IsGenericPhrase(string phrase)
    {
        var lower = phrase.ToLower();
        var genericPhrases = new[]
        {
            "recommended design", "design workflow", "pcb layout", "pcb design",
            "schematic capture", "signal integrity", "impedance calculation",
            "trace width", "power supply", "circuit design", "component selection",
            "design rule", "manufacturing preparation", "bill of materials",
            "thermal management", "frequency response", "voltage regulation",
            "current limiting", "data exchange", "file format", "user interface",
            "open source", "real time", "high speed", "low power", "mixed signal"
        };
        return genericPhrases.Any(gp => lower.Contains(gp)) || lower.Split(' ').Length > 4;
    }

    private static int LevenshteinDistance(string s, string t)
    {
        var n = s.Length;
        var m = t.Length;
        var d = new int[n + 1, m + 1];
        for (var i = 0; i <= n; d[i, 0] = i++) { }
        for (var j = 0; j <= m; d[0, j] = j++) { }
        for (var i = 1; i <= n; i++)
            for (var j = 1; j <= m; j++)
            {
                var cost = t[j - 1] == s[i - 1] ? 0 : 1;
                d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + cost);
            }
        return d[n, m];
    }
}

public class AnalysisResult
{
    public bool IsValidJson { get; set; }
    public List<string> MentionedToolNames { get; set; } = [];
    public List<string> GroundedToolNames { get; set; } = [];
    public List<string> HallucinatedToolNames { get; set; } = [];
    public List<string> WorkflowStages { get; set; } = [];
}
