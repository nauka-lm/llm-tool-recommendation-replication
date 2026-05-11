namespace CadcomOnline.Evaluation.Models;

/// <summary>
/// Represents a single evaluation query to be sent to an LLM.
/// </summary>
public class EvaluationQuery
{
    public string QueryId { get; set; } = string.Empty;
    public string DomainId { get; set; } = string.Empty;
    public string ConfigId { get; set; } = string.Empty;
    public int PromptIndex { get; set; }
    public int Repetition { get; set; }
    public string PromptText { get; set; } = string.Empty;
    public string UserQuery { get; set; } = string.Empty;
}

/// <summary>
/// Stores the raw result from a single API call.
/// Extended with provider/model tracking for multi-model evaluation.
/// </summary>
public class QueryResult
{
    public string QueryId { get; set; } = string.Empty;
    public string DomainId { get; set; } = string.Empty;
    public string ConfigId { get; set; } = string.Empty;
    public string ProviderId { get; set; } = string.Empty;
    public string ModelId { get; set; } = string.Empty;
    public string ModelCategory { get; set; } = string.Empty;
    public int PromptIndex { get; set; }
    public int Repetition { get; set; }
    public string RawResponse { get; set; } = string.Empty;
    public bool IsValidJson { get; set; }
    public List<string> MentionedToolNames { get; set; } = [];
    public List<string> HallucinatedToolNames { get; set; } = [];
    public List<string> GroundedToolNames { get; set; } = [];
    public int WorkflowStepsCount { get; set; }
    public List<string> WorkflowStages { get; set; } = [];
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }
    public int ThinkingTokens { get; set; }
    public double LatencyMs { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Aggregated metrics for a domain+config+model combination.
/// Extended with provider/model tracking for cross-model comparison.
/// </summary>
public class MetricsResult
{
    public string DomainId { get; set; } = string.Empty;
    public string ConfigId { get; set; } = string.Empty;
    public string ProviderId { get; set; } = string.Empty;
    public string ModelId { get; set; } = string.Empty;
    public string ModelCategory { get; set; } = string.Empty;

    // Hallucination Rate: % of mentioned tool names NOT in database
    public double HallucinationRate { get; set; }

    // Grounding Rate: % of mentioned tool names found in database (100% - HR)
    public double GroundingRate { get; set; }

    // Workflow Coverage: distinct workflow stages covered / 6
    public double WorkflowCoverage { get; set; }

    // Response Consistency: % of tools consistently recommended across repetitions
    public double ResponseConsistency { get; set; }

    // Average latency in milliseconds
    public double AvgLatencyMs { get; set; }

    // Average thinking tokens per query (0 for standard models)
    public double AvgThinkingTokens { get; set; }

    public int TotalToolsMentioned { get; set; }
    public int TotalHallucinated { get; set; }
    public int TotalGrounded { get; set; }
    public int QueriesExecuted { get; set; }
    public int QueriesWithErrors { get; set; }
}
