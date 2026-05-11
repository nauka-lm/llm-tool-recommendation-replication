using System.Text.Json;
using CadcomOnline.Evaluation.Clients;
using CadcomOnline.Evaluation.Models;

namespace CadcomOnline.Evaluation.Services;

/// <summary>
/// Orchestrates the multi-model evaluation experiment.
/// Supports running against any ILlmClient (OpenAI, Anthropic, Google) in both standard and thinking modes.
/// </summary>
public class EvaluationRunner
{
    private readonly DatabaseService _dbService;
    private readonly PromptBuilder _promptBuilder;
    private readonly TestPromptGenerator _testPromptGenerator;
    private readonly HallucinationDetector _hallucinationDetector;
    private readonly MetricsCalculator _metricsCalculator;
    private readonly int _repetitions;
    private readonly string _outputDir;

    /// <summary>Per-provider delay in ms to respect rate limits.</summary>
    private static readonly Dictionary<string, int> ProviderDelays = new()
    {
        { "openai", 1000 },
        { "anthropic", 2000 },
        { "google", 5000 }
    };

    /// <summary>Save results every N new API calls to prevent data loss.</summary>
    private const int IncrementalSaveInterval = 10;

    public EvaluationRunner(
        DatabaseService dbService,
        PromptBuilder promptBuilder,
        TestPromptGenerator testPromptGenerator,
        HallucinationDetector hallucinationDetector,
        MetricsCalculator metricsCalculator,
        int repetitions = 3,
        string outputDir = "Results")
    {
        _dbService = dbService;
        _promptBuilder = promptBuilder;
        _testPromptGenerator = testPromptGenerator;
        _hallucinationDetector = hallucinationDetector;
        _metricsCalculator = metricsCalculator;
        _repetitions = repetitions;
        _outputDir = outputDir;
    }

    /// <summary>
    /// Defines the 4 experimental domains.
    /// </summary>
    public static readonly (string Id, string SubcategoryName)[] DomainDefinitions =
    [
        ("D1", "PCB Design Tool"),
        ("D2", "PCB Design Calculator"),
        ("D3", "Switched-mode power supplies (SMPS), converters, regulators"),
        ("D4", "Transformers")
    ];

    /// <summary>
    /// Runs the experiment for a single LLM client across specified configs and domains.
    /// </summary>
    public async Task<List<QueryResult>> RunExperimentAsync(
        ILlmClient client,
        string[]? specificConfigs = null,
        string[]? specificDomains = null,
        int[]? specificPromptIndices = null,
        int? overrideRepetitions = null)
    {
        var delayMs = ProviderDelays.GetValueOrDefault(client.ProviderId, 2000);
        var effectiveReps = overrideRepetitions ?? _repetitions;

        Console.WriteLine();
        Console.WriteLine($"{'='.ToString().PadLeft(70, '=')}");
        Console.WriteLine($"  MODEL: {client.DisplayName} ({client.ModelId})");
        Console.WriteLine($"  Provider: {client.ProviderId} | Category: {client.Category}");
        Console.WriteLine($"  Delay: {delayMs}ms between calls");
        Console.WriteLine($"{'='.ToString().PadLeft(70, '=')}");

        Directory.CreateDirectory(_outputDir);

        // Load domains
        var domains = new List<Domain>();
        foreach (var (id, subcategoryName) in DomainDefinitions)
        {
            if (specificDomains != null && !specificDomains.Contains(id)) continue;

            Console.Write($"  Loading domain {id} ({subcategoryName})... ");
            var domain = await _dbService.LoadDomainAsync(id, subcategoryName);
            domains.Add(domain);
            Console.WriteLine($"{domain.Tools.Count} tools loaded.");
        }

        var configs = AblationConfig.AllConfigs;
        if (specificConfigs != null)
            configs = configs.Where(c => specificConfigs.Contains(c.Id)).ToArray();

        var allResults = new List<QueryResult>();
        var promptCount = specificPromptIndices?.Length ?? 12;
        var totalCalls = domains.Count * configs.Length * promptCount * effectiveReps;
        var currentCall = 0;

        Console.WriteLine();
        Console.WriteLine($"  Total API calls: {totalCalls}");
        Console.WriteLine($"  Repetitions: {effectiveReps}");
        if (specificPromptIndices != null)
            Console.WriteLine($"  Prompt indices: {string.Join(",", specificPromptIndices)}");
        Console.WriteLine();

        foreach (var domain in domains)
        {
            Console.WriteLine($"\n  {'='.ToString().PadLeft(60, '=')}");
            Console.WriteLine($"  DOMAIN: {domain.Id} - {domain.Name} ({domain.Tools.Count} tools)");
            Console.WriteLine($"  {'='.ToString().PadLeft(60, '=')}");

            var prompts = _testPromptGenerator.GeneratePrompts(domain);
            var primaryTool = domain.Tools[0];

            Console.Write("  Loading related tools (6-stage context builder)... ");
            var relatedTools = await _dbService.GetRelatedToolsAsync(primaryTool.Id);
            Console.WriteLine($"{relatedTools.Count} tools found.");

            foreach (var config in configs)
            {
                var expectedCount = promptCount * effectiveReps;
                var resultFile = Path.Combine(_outputDir, $"results_{client.ProviderId}_{client.Category}_{domain.Id}_{config.Id}.json");

                // Resume support: load existing partial or complete results
                var blockResults = new List<QueryResult>();
                var completedQueryIds = new HashSet<string>();

                if (File.Exists(resultFile))
                {
                    try
                    {
                        var existingJson = await File.ReadAllTextAsync(resultFile);
                        var existingResults = JsonSerializer.Deserialize<List<QueryResult>>(existingJson);
                        if (existingResults != null)
                        {
                            var errorCount = existingResults.Count(r => r.Error != null);

                            // Fully complete with 0 errors → skip entire block
                            if (existingResults.Count >= expectedCount && errorCount == 0)
                            {
                                Console.WriteLine($"\n  Config: {config} -- SKIPPED (already complete: {existingResults.Count} results, 0 errors)");
                                allResults.AddRange(existingResults);
                                currentCall += expectedCount;
                                continue;
                            }

                            // Partial results → resume from where we left off
                            blockResults = existingResults.Where(r => r.Error == null).ToList();
                            completedQueryIds = blockResults.Select(r => r.QueryId).ToHashSet();
                            Console.WriteLine($"\n  Config: {config} -- RESUMING ({blockResults.Count}/{expectedCount} done, skipping completed queries)");
                        }
                    }
                    catch { /* re-run if file is corrupt */ }
                }

                if (!completedQueryIds.Any())
                    Console.WriteLine($"\n  Config: {config}");

                var newCallsInBlock = 0;

                for (int promptIdx = 0; promptIdx < prompts.Count; promptIdx++)
                {
                    if (specificPromptIndices != null && !specificPromptIndices.Contains(promptIdx))
                        continue;

                    var userQuery = prompts[promptIdx];

                    for (int rep = 0; rep < effectiveReps; rep++)
                    {
                        currentCall++;
                        var queryId = $"{client.ProviderId}_{client.Category}_{domain.Id}_{config.Id}_P{promptIdx + 1:D2}_R{rep + 1}";

                        // Skip if already completed successfully
                        if (completedQueryIds.Contains(queryId))
                        {
                            Console.WriteLine($"    [{currentCall}/{totalCalls}] {queryId}... SKIPPED (done)");
                            continue;
                        }

                        Console.Write($"    [{currentCall}/{totalCalls}] {queryId}... ");

                        var result = new QueryResult
                        {
                            QueryId = queryId,
                            DomainId = domain.Id,
                            ConfigId = config.Id,
                            ProviderId = client.ProviderId,
                            ModelId = client.ModelId,
                            ModelCategory = client.Category,
                            PromptIndex = promptIdx,
                            Repetition = rep + 1,
                            Timestamp = DateTime.UtcNow
                        };

                        var systemPrompt = _promptBuilder.BuildPrompt(
                            config, domain, primaryTool, domain.Tools, relatedTools, userQuery);

                        var transientBackoffsSec = new[] { 30, 90, 270, 810 };
                        Exception? lastError = null;
                        var succeeded = false;

                        for (int attempt = 0; attempt <= transientBackoffsSec.Length; attempt++)
                        {
                            try
                            {
                                var llmResponse = await client.ChatAsync(systemPrompt, "");

                                result.RawResponse = llmResponse.Text;
                                result.PromptTokens = llmResponse.InputTokens;
                                result.CompletionTokens = llmResponse.OutputTokens;
                                result.ThinkingTokens = llmResponse.ThinkingTokens;
                                result.LatencyMs = llmResponse.LatencyMs;

                                var analysis = _hallucinationDetector.Analyze(result.RawResponse, domain);
                                result.IsValidJson = analysis.IsValidJson;
                                result.MentionedToolNames = analysis.MentionedToolNames;
                                result.HallucinatedToolNames = analysis.HallucinatedToolNames;
                                result.GroundedToolNames = analysis.GroundedToolNames;
                                result.WorkflowStages = analysis.WorkflowStages;
                                result.WorkflowStepsCount = analysis.WorkflowStages.Count;

                                Console.WriteLine(
                                    $"OK (tools: {result.MentionedToolNames.Count}, " +
                                    $"halluc: {result.HallucinatedToolNames.Count}, " +
                                    $"think: {result.ThinkingTokens}, " +
                                    $"tokens: {result.PromptTokens}+{result.CompletionTokens}, " +
                                    $"{result.LatencyMs:F0}ms" +
                                    (attempt > 0 ? $", retried x{attempt}" : "") +
                                    ")");
                                succeeded = true;
                                break;
                            }
                            catch (Exception ex)
                            {
                                lastError = ex;
                                var msg = ex.Message;
                                var transient =
                                    msg.Contains("429") || msg.Contains("rate") ||
                                    msg.Contains("500") || msg.Contains("502") ||
                                    msg.Contains("503") || msg.Contains("504") ||
                                    msg.Contains("overloaded") ||
                                    msg.Contains("timeout") || msg.Contains("Timeout") ||
                                    ex is TaskCanceledException ||
                                    ex is HttpRequestException;
                                var permanent = msg.Contains("401") || msg.Contains("403") || msg.Contains("404");

                                if (permanent)
                                {
                                    Console.WriteLine($"PERMANENT ERROR (no retry): {msg}");
                                    break;
                                }

                                if (!transient || attempt >= transientBackoffsSec.Length)
                                {
                                    Console.WriteLine($"ERROR ({(transient ? "transient, retries exhausted" : "non-transient")}): {msg}");
                                    break;
                                }

                                var backoff = transientBackoffsSec[attempt];
                                Console.WriteLine($"TRANSIENT ERROR (attempt {attempt + 1}/{transientBackoffsSec.Length + 1}, backoff {backoff}s): {msg}");
                                await Task.Delay(TimeSpan.FromSeconds(backoff));
                            }
                        }

                        if (!succeeded && lastError != null)
                            result.Error = lastError.Message;

                        blockResults.Add(result);
                        newCallsInBlock++;

                        // Incremental save every N new API calls
                        if (newCallsInBlock % IncrementalSaveInterval == 0)
                        {
                            await SaveBlockResultsAsync(blockResults, resultFile);
                            Console.WriteLine($"    ** INCREMENTAL SAVE: {blockResults.Count} results → {Path.GetFileName(resultFile)}");
                        }

                        if (currentCall < totalCalls)
                            await Task.Delay(delayMs);
                    }
                }

                // Final save for this config block
                await SaveBlockResultsAsync(blockResults, resultFile);
                Console.WriteLine($"    ** BLOCK SAVE: {blockResults.Count} results → {Path.GetFileName(resultFile)}");
                allResults.AddRange(blockResults);
            }
        }

        return allResults;
    }

    /// <summary>
    /// Runs the full multi-model experiment across all provided clients.
    /// </summary>
    public async Task RunMultiModelExperimentAsync(
        ILlmClient[] clients,
        string[]? specificConfigs = null,
        string[]? specificDomains = null)
    {
        Console.WriteLine();
        Console.WriteLine("=======================================================================");
        Console.WriteLine("  CADCOMONLINE - MULTI-MODEL ANTI-HALLUCINATION EVALUATION");
        Console.WriteLine("=======================================================================");
        Console.WriteLine();
        Console.WriteLine($"  Models: {clients.Length}");
        foreach (var c in clients)
            Console.WriteLine($"    - {c.DisplayName} ({c.ProviderId}/{c.Category})");
        Console.WriteLine($"  Repetitions: {_repetitions}");
        Console.WriteLine();

        Directory.CreateDirectory(_outputDir);
        var allResults = new List<QueryResult>();

        foreach (var client in clients)
        {
            var clientResults = await RunExperimentAsync(client, specificConfigs, specificDomains);
            allResults.AddRange(clientResults);
        }

        // Compute and save metrics for all models
        Console.WriteLine("\n\n=======================================================================");
        Console.WriteLine("  COMPUTING CROSS-MODEL METRICS");
        Console.WriteLine("=======================================================================\n");

        var allMetrics = new List<MetricsResult>();

        Console.WriteLine(MetricsCalculator.FormatMetricsHeader());

        var domains = allResults.Select(r => r.DomainId).Distinct().OrderBy(d => d).ToList();
        var configs = allResults.Select(r => r.ConfigId).Distinct().OrderBy(c => c).ToList();
        var models = allResults
            .Select(r => (r.ProviderId, r.ModelId, r.ModelCategory))
            .Distinct()
            .OrderBy(m => m.ModelCategory)
            .ThenBy(m => m.ProviderId)
            .ToList();

        foreach (var (providerId, modelId, modelCategory) in models)
        {
            foreach (var domain in domains)
            {
                foreach (var config in configs)
                {
                    var subset = allResults
                        .Where(r => r.DomainId == domain && r.ConfigId == config
                            && r.ProviderId == providerId && r.ModelId == modelId)
                        .ToList();

                    if (!subset.Any()) continue;

                    var metrics = _metricsCalculator.ComputeMetrics(
                        domain, config, subset, providerId, modelId, modelCategory);
                    allMetrics.Add(metrics);

                    Console.WriteLine(MetricsCalculator.FormatMetricsRow(metrics));
                }
            }
        }

        // Save all results and metrics
        await SaveAllResultsAsync(allResults, allMetrics);

        Console.WriteLine($"\n\nExperiment complete. Results saved to: {Path.GetFullPath(_outputDir)}");
    }

    /// <summary>
    /// Runs a single test query for debugging.
    /// </summary>
    public async Task RunSingleTestAsync(ILlmClient client, string domainId, string configId, int promptIndex)
    {
        var domainDef = DomainDefinitions.First(d => d.Id == domainId);
        var domain = await _dbService.LoadDomainAsync(domainDef.Id, domainDef.SubcategoryName);
        var config = AblationConfig.AllConfigs.First(c => c.Id == configId);
        var prompts = _testPromptGenerator.GeneratePrompts(domain);
        var primaryTool = domain.Tools[0];
        var relatedTools = await _dbService.GetRelatedToolsAsync(primaryTool.Id);

        var userQuery = prompts[promptIndex];
        Console.WriteLine($"\nModel: {client.DisplayName} ({client.ProviderId}/{client.Category})");
        Console.WriteLine($"Domain: {domain.Id} ({domain.Name})");
        Console.WriteLine($"Config: {config}");
        Console.WriteLine($"Prompt: {userQuery}\n");

        var systemPrompt = _promptBuilder.BuildPrompt(config, domain, primaryTool, domain.Tools, relatedTools, userQuery);

        Console.WriteLine("--- SYSTEM PROMPT ---");
        Console.WriteLine(systemPrompt[..Math.Min(systemPrompt.Length, 500)] + "...\n");
        Console.WriteLine($"Prompt length: {systemPrompt.Length} chars\n");

        var response = await client.ChatAsync(systemPrompt, "");

        Console.WriteLine("--- RAW RESPONSE ---");
        Console.WriteLine(response.Text);

        var analysis = _hallucinationDetector.Analyze(response.Text, domain);
        Console.WriteLine("\n--- ANALYSIS ---");
        Console.WriteLine($"Valid JSON: {analysis.IsValidJson}");
        Console.WriteLine($"Tool names mentioned: {string.Join(", ", analysis.MentionedToolNames)}");
        Console.WriteLine($"Grounded tools: {string.Join(", ", analysis.GroundedToolNames)}");
        Console.WriteLine($"HALLUCINATED tools: {string.Join(", ", analysis.HallucinatedToolNames)}");
        Console.WriteLine($"Workflow stages: {string.Join(", ", analysis.WorkflowStages)}");
        Console.WriteLine($"Tokens: {response.InputTokens}+{response.OutputTokens} (thinking: {response.ThinkingTokens})");
        Console.WriteLine($"Latency: {response.LatencyMs:F0}ms");
    }

    // ─── Save methods ─────────────────────────────────────────────────────

    /// <summary>
    /// Saves a block of results to a specific file. Used for both incremental and final saves.
    /// </summary>
    private static async Task SaveBlockResultsAsync(List<QueryResult> blockResults, string filePath)
    {
        var json = JsonSerializer.Serialize(blockResults, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(filePath, json);
    }

    private async Task SaveAllResultsAsync(List<QueryResult> allResults, List<MetricsResult> allMetrics)
    {
        // 1. Raw results JSON
        await File.WriteAllTextAsync(
            Path.Combine(_outputDir, "all_results.json"),
            JsonSerializer.Serialize(allResults, new JsonSerializerOptions { WriteIndented = true }));

        // 2. Metrics JSON
        await File.WriteAllTextAsync(
            Path.Combine(_outputDir, "metrics_summary.json"),
            JsonSerializer.Serialize(allMetrics, new JsonSerializerOptions { WriteIndented = true }));

        // 3. Metrics CSV
        var csvLines = new List<string>
        {
            "Domain,Config,Provider,ModelId,Category,HallucinationRate,GroundingRate,WorkflowCoverage,ResponseConsistency,AvgLatencyMs,AvgThinkingTokens,TotalMentioned,TotalHallucinated"
        };
        foreach (var m in allMetrics)
            csvLines.Add($"{m.DomainId},{m.ConfigId},{m.ProviderId},{m.ModelId},{m.ModelCategory},{m.HallucinationRate:F2},{m.GroundingRate:F2},{m.WorkflowCoverage:F2},{m.ResponseConsistency:F2},{m.AvgLatencyMs:F0},{m.AvgThinkingTokens:F1},{m.TotalToolsMentioned},{m.TotalHallucinated}");
        await File.WriteAllLinesAsync(Path.Combine(_outputDir, "metrics_summary.csv"), csvLines);

        // 4. Markdown reports
        await SaveMetricsSummaryMarkdown(allMetrics);
        await SaveHallucinationAnalysisMarkdown(allResults);
        await SaveCrossModelComparisonMarkdown(allMetrics);
        await SaveAblationStudyMarkdown(allMetrics);
    }

    private async Task SaveMetricsSummaryMarkdown(List<MetricsResult> allMetrics)
    {
        var md = new List<string>
        {
            "# Multi-Model Anti-Hallucination Evaluation - Metrics Summary",
            "",
            $"> Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC",
            $"> Repetitions: {_repetitions}",
            "",
            "## Overall Results by Model",
            ""
        };

        var models = allMetrics
            .Select(m => (m.ProviderId, m.ModelId, m.ModelCategory))
            .Distinct().OrderBy(m => m.ModelCategory).ThenBy(m => m.ProviderId).ToList();

        foreach (var (providerId, modelId, modelCategory) in models)
        {
            md.Add($"### {modelId} ({providerId}, {modelCategory})");
            md.Add("");
            md.Add("| Domain | Config | HR (%) | GR (%) | WC (%) | RC (%) | Latency (ms) | Think Tok | Tools | Halluc |");
            md.Add("|--------|--------|--------|--------|--------|--------|-------------|-----------|-------|--------|");

            var modelMetrics = allMetrics
                .Where(m => m.ProviderId == providerId && m.ModelId == modelId)
                .OrderBy(m => m.DomainId).ThenBy(m => m.ConfigId);

            foreach (var m in modelMetrics)
                md.Add($"| {m.DomainId} | {m.ConfigId} | {m.HallucinationRate:F1} | {m.GroundingRate:F1} | {m.WorkflowCoverage:F1} | {m.ResponseConsistency:F1} | {m.AvgLatencyMs:F0} | {m.AvgThinkingTokens:F0} | {m.TotalToolsMentioned} | {m.TotalHallucinated} |");

            md.Add("");
        }

        // Cross-model averages by config
        md.Add("## Cross-Model Averages by Configuration");
        md.Add("");
        md.Add("| Config | Model | Category | Avg HR (%) | Avg GR (%) | Avg WC (%) | Avg RC (%) |");
        md.Add("|--------|-------|----------|-----------|-----------|-----------|-----------|");

        foreach (var config in allMetrics.Select(m => m.ConfigId).Distinct().OrderBy(c => c))
        {
            foreach (var (providerId, modelId, modelCategory) in models)
            {
                var subset = allMetrics
                    .Where(m => m.ConfigId == config && m.ProviderId == providerId && m.ModelId == modelId)
                    .ToList();
                if (!subset.Any()) continue;

                md.Add($"| {config} | {modelId} | {modelCategory} | {subset.Average(m => m.HallucinationRate):F1} | {subset.Average(m => m.GroundingRate):F1} | {subset.Average(m => m.WorkflowCoverage):F1} | {subset.Average(m => m.ResponseConsistency):F1} |");
            }
        }

        md.Add("");
        md.Add("---");
        md.Add("");
        md.Add("### Metric Definitions");
        md.Add("");
        md.Add("- **HR (Hallucination Rate)**: % of mentioned tool names NOT in database. Lower is better.");
        md.Add("- **GR (Grounding Rate)**: % of mentioned tool names found in database (100% - HR). Higher is better.");
        md.Add("- **WC (Workflow Coverage)**: Distinct workflow stages covered / 6 stages. Higher is better.");
        md.Add("- **RC (Response Consistency)**: Jaccard similarity of tool sets across repetitions. Higher is better.");

        await File.WriteAllLinesAsync(Path.Combine(_outputDir, "01-metrics-summary.md"), md);
    }

    private async Task SaveHallucinationAnalysisMarkdown(List<QueryResult> allResults)
    {
        var md = new List<string>
        {
            "# Hallucination Analysis - Cross-Model",
            "",
            $"> Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC",
            ""
        };

        var models = allResults
            .Select(r => (r.ProviderId, r.ModelId, r.ModelCategory))
            .Distinct().OrderBy(m => m.ModelCategory).ThenBy(m => m.ProviderId).ToList();

        foreach (var (providerId, modelId, modelCategory) in models)
        {
            md.Add($"## {modelId} ({providerId}, {modelCategory})");
            md.Add("");

            var modelResults = allResults
                .Where(r => r.ProviderId == providerId && r.ModelId == modelId)
                .ToList();

            var configGroups = modelResults
                .Where(r => r.HallucinatedToolNames.Any())
                .GroupBy(r => r.ConfigId).OrderBy(g => g.Key);

            foreach (var group in configGroups)
            {
                var hallucinatedNames = group
                    .SelectMany(r => r.HallucinatedToolNames)
                    .GroupBy(n => n, StringComparer.OrdinalIgnoreCase)
                    .OrderByDescending(g => g.Count()).ToList();

                md.Add($"### Config {group.Key} ({hallucinatedNames.Count} unique hallucinated names)");
                md.Add("");
                md.Add("| Hallucinated Tool Name | Occurrences |");
                md.Add("|------------------------|-------------|");

                foreach (var name in hallucinatedNames.Take(15))
                    md.Add($"| {name.Key} | {name.Count()} |");

                md.Add("");
            }
        }

        await File.WriteAllLinesAsync(Path.Combine(_outputDir, "02-hallucination-analysis.md"), md);
    }

    private async Task SaveCrossModelComparisonMarkdown(List<MetricsResult> allMetrics)
    {
        var md = new List<string>
        {
            "# Cross-Model Comparison",
            "",
            $"> Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC",
            "",
            "## Standard vs Thinking Models (C0 -- Baseline, No Guardrails)",
            ""
        };

        var c0Metrics = allMetrics.Where(m => m.ConfigId == "C0").ToList();
        if (c0Metrics.Any())
        {
            md.Add("| Model | Category | Provider | Avg HR (%) | Avg GR (%) | Avg Latency (ms) | Avg Think Tok |");
            md.Add("|-------|----------|----------|-----------|-----------|-------------------|---------------|");

            var c0ByModel = c0Metrics
                .GroupBy(m => (m.ProviderId, m.ModelId, m.ModelCategory))
                .OrderBy(g => g.Key.ModelCategory).ThenBy(g => g.Key.ProviderId);

            foreach (var group in c0ByModel)
                md.Add($"| {group.Key.ModelId} | {group.Key.ModelCategory} | {group.Key.ProviderId} | {group.Average(m => m.HallucinationRate):F1} | {group.Average(m => m.GroundingRate):F1} | {group.Average(m => m.AvgLatencyMs):F0} | {group.Average(m => m.AvgThinkingTokens):F0} |");

            md.Add("");
        }

        md.Add("## Standard vs Thinking Models (C5 -- Full Architecture)");
        md.Add("");

        var c5Metrics = allMetrics.Where(m => m.ConfigId == "C5").ToList();
        if (c5Metrics.Any())
        {
            md.Add("| Model | Category | Provider | Avg HR (%) | Avg GR (%) | Avg Latency (ms) | Avg Think Tok |");
            md.Add("|-------|----------|----------|-----------|-----------|-------------------|---------------|");

            var c5ByModel = c5Metrics
                .GroupBy(m => (m.ProviderId, m.ModelId, m.ModelCategory))
                .OrderBy(g => g.Key.ModelCategory).ThenBy(g => g.Key.ProviderId);

            foreach (var group in c5ByModel)
                md.Add($"| {group.Key.ModelId} | {group.Key.ModelCategory} | {group.Key.ProviderId} | {group.Average(m => m.HallucinationRate):F1} | {group.Average(m => m.GroundingRate):F1} | {group.Average(m => m.AvgLatencyMs):F0} | {group.Average(m => m.AvgThinkingTokens):F0} |");

            md.Add("");
        }

        // Architecture effectiveness: C0 vs C5 delta per model
        md.Add("## Architecture Effectiveness (C0 vs C5 HR Reduction)");
        md.Add("");
        md.Add("| Model | Category | C0 HR (%) | C5 HR (%) | Reduction (pp) | Effective? |");
        md.Add("|-------|----------|----------|----------|----------------|-----------|");

        var models = allMetrics
            .Select(m => (m.ProviderId, m.ModelId, m.ModelCategory))
            .Distinct().OrderBy(m => m.ModelCategory).ThenBy(m => m.ProviderId).ToList();

        foreach (var (providerId, modelId, modelCategory) in models)
        {
            var c0 = allMetrics.Where(m => m.ConfigId == "C0" && m.ProviderId == providerId && m.ModelId == modelId).ToList();
            var c5 = allMetrics.Where(m => m.ConfigId == "C5" && m.ProviderId == providerId && m.ModelId == modelId).ToList();

            if (c0.Any() && c5.Any())
            {
                var c0Avg = c0.Average(m => m.HallucinationRate);
                var c5Avg = c5.Average(m => m.HallucinationRate);
                var reduction = c0Avg - c5Avg;
                var effective = reduction > 5 ? "YES" : reduction > 0 ? "Marginal" : "NO";
                md.Add($"| {modelId} | {modelCategory} | {c0Avg:F1} | {c5Avg:F1} | {reduction:F1} | {effective} |");
            }
        }

        await File.WriteAllLinesAsync(Path.Combine(_outputDir, "03-cross-model-comparison.md"), md);
    }

    private async Task SaveAblationStudyMarkdown(List<MetricsResult> allMetrics)
    {
        var configs = allMetrics.Select(m => m.ConfigId).Distinct().OrderBy(c => c).ToList();
        if (configs.Count < 3) return;

        var md = new List<string>
        {
            "# Ablation Study Results - Cross-Model",
            "",
            $"> Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC",
            "",
            "## Configuration Definitions",
            "",
            "| Config | Context Builder | Closed Vocabulary | JSON Enforcement | Description |",
            "|--------|----------------|-------------------|-----------------|-------------|"
        };

        foreach (var config in AblationConfig.AllConfigs)
        {
            if (!configs.Contains(config.Id)) continue;
            md.Add($"| {config.Id} | {(config.UseContextBuilder ? "On" : "Off")} | {(config.UseClosedVocabulary ? "On" : "Off")} | {(config.UseJsonEnforcement ? "On" : "Off")} | {config.Description} |");
        }

        // Per-model ablation
        var models = allMetrics
            .Select(m => (m.ProviderId, m.ModelId, m.ModelCategory))
            .Distinct().OrderBy(m => m.ModelCategory).ThenBy(m => m.ProviderId).ToList();

        foreach (var (providerId, modelId, modelCategory) in models)
        {
            md.Add($"\n## {modelId} ({modelCategory})");
            md.Add("");

            var domains = allMetrics.Where(m => m.ProviderId == providerId && m.ModelId == modelId)
                .Select(m => m.DomainId).Distinct().OrderBy(d => d).ToList();

            var header = "| Config |" + string.Join("|", domains.Select(d => $" HR {d} (%) ")) + "| Avg HR (%) |";
            var sep = "|--------|" + string.Join("|", domains.Select(_ => "----------")) + "|-----------|";
            md.Add(header);
            md.Add(sep);

            foreach (var configId in configs)
            {
                var configMetrics = allMetrics
                    .Where(m => m.ConfigId == configId && m.ProviderId == providerId && m.ModelId == modelId)
                    .ToList();

                if (!configMetrics.Any()) continue;

                var row = $"| {configId} |";
                foreach (var d in domains)
                {
                    var hr = configMetrics.FirstOrDefault(m => m.DomainId == d)?.HallucinationRate ?? 0;
                    row += $" {hr:F1} |";
                }
                row += $" {configMetrics.Average(m => m.HallucinationRate):F1} |";
                md.Add(row);
            }
        }

        await File.WriteAllLinesAsync(Path.Combine(_outputDir, "04-ablation-study.md"), md);
    }
}
