using CadcomOnline.Evaluation.Models;

namespace CadcomOnline.Evaluation.Services;

/// <summary>
/// Computes the 4 evaluation metrics defined in the paper:
/// HR (Hallucination Rate), GR (Grounding Rate), WC (Workflow Coverage), RC (Response Consistency).
/// Extended for multi-model evaluation with latency and thinking token tracking.
/// </summary>
public class MetricsCalculator
{
    /// <summary>
    /// Computes aggregated metrics for a set of query results (same domain + config + model).
    /// </summary>
    public MetricsResult ComputeMetrics(string domainId, string configId, List<QueryResult> results,
        string providerId = "", string modelId = "", string modelCategory = "")
    {
        var metrics = new MetricsResult
        {
            DomainId = domainId,
            ConfigId = configId,
            ProviderId = providerId,
            ModelId = modelId,
            ModelCategory = modelCategory,
            QueriesExecuted = results.Count,
            QueriesWithErrors = results.Count(r => r.Error != null)
        };

        var validResults = results.Where(r => r.Error == null).ToList();
        if (!validResults.Any()) return metrics;

        // Metric 1: Hallucination Rate (HR)
        var totalMentioned = validResults.Sum(r => r.MentionedToolNames.Count);
        var totalHallucinated = validResults.Sum(r => r.HallucinatedToolNames.Count);
        var totalGrounded = validResults.Sum(r => r.GroundedToolNames.Count);

        metrics.TotalToolsMentioned = totalMentioned;
        metrics.TotalHallucinated = totalHallucinated;
        metrics.TotalGrounded = totalGrounded;
        metrics.HallucinationRate = totalMentioned > 0
            ? (double)totalHallucinated / totalMentioned * 100.0
            : 0;

        // Metric 2: Grounding Rate (GR) = 100% - HR
        metrics.GroundingRate = 100.0 - metrics.HallucinationRate;

        // Metric 3: Workflow Coverage (WC)
        var allStages = validResults.SelectMany(r => r.WorkflowStages).Distinct().Count();
        metrics.WorkflowCoverage = allStages / 6.0 * 100.0;

        // Metric 4: Response Consistency (RC)
        metrics.ResponseConsistency = ComputeConsistency(validResults);

        // Extended metrics: latency and thinking tokens
        metrics.AvgLatencyMs = validResults.Average(r => r.LatencyMs);
        metrics.AvgThinkingTokens = validResults.Average(r => r.ThinkingTokens);

        return metrics;
    }

    /// <summary>
    /// Computes response consistency using pairwise Jaccard similarity across repetitions.
    /// </summary>
    private double ComputeConsistency(List<QueryResult> results)
    {
        var byPrompt = results.GroupBy(r => r.PromptIndex);
        var consistencyScores = new List<double>();

        foreach (var group in byPrompt)
        {
            var repetitions = group.ToList();
            if (repetitions.Count < 2) continue;

            var toolSets = repetitions
                .Select(r => new HashSet<string>(r.GroundedToolNames, StringComparer.OrdinalIgnoreCase))
                .ToList();

            var pairwiseScores = new List<double>();
            for (int i = 0; i < toolSets.Count; i++)
            {
                for (int j = i + 1; j < toolSets.Count; j++)
                {
                    var intersection = toolSets[i].Intersect(toolSets[j], StringComparer.OrdinalIgnoreCase).Count();
                    var union = toolSets[i].Union(toolSets[j], StringComparer.OrdinalIgnoreCase).Count();
                    if (union > 0)
                        pairwiseScores.Add((double)intersection / union * 100.0);
                }
            }

            if (pairwiseScores.Any())
                consistencyScores.Add(pairwiseScores.Average());
        }

        return consistencyScores.Any() ? consistencyScores.Average() : 0;
    }

    public static string FormatMetricsRow(MetricsResult m)
    {
        var modelLabel = !string.IsNullOrEmpty(m.ModelId)
            ? $"{m.ModelId}({m.ModelCategory[0]})"
            : "";

        return $"| {m.DomainId,-6} | {m.ConfigId,-6} | {modelLabel,-22} | {m.HallucinationRate,6:F1}% | {m.GroundingRate,6:F1}% | {m.WorkflowCoverage,6:F1}% | {m.ResponseConsistency,6:F1}% | {m.TotalToolsMentioned,5} | {m.TotalHallucinated,5} | {m.AvgLatencyMs,8:F0}ms |";
    }

    public static string FormatMetricsHeader()
    {
        return @"| Domain | Config | Model                  |    HR    |    GR    |    WC    |    RC    | Total | Halluc | Latency    |
|--------|--------|------------------------|----------|----------|----------|----------|-------|--------|------------|";
    }
}
