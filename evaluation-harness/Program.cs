using CadcomOnline.Data;
using CadcomOnline.Evaluation.Clients;
using CadcomOnline.Evaluation.Models;
using CadcomOnline.Evaluation.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;

// ═══════════════════════════════════════════════════════════════
// CadcomOnline Multi-Model Anti-Hallucination Evaluation Harness
// For Journal Paper: Cross-Provider Evaluation
// Supports Gen1 (Phase 1+2) and Gen2 (Phase 3) models
// ═══════════════════════════════════════════════════════════════

var configuration = new ConfigurationBuilder()
    .SetBasePath(Directory.GetCurrentDirectory())
    .AddJsonFile("appsettings.json", optional: false)
    .Build();

var connectionString = configuration.GetConnectionString("DBConnection")
    ?? throw new InvalidOperationException("DBConnection not found in appsettings.json");
var repetitions = int.Parse(configuration["Evaluation:Repetitions"] ?? "3");
var outputDir = configuration["Evaluation:OutputDirectory"] ?? "Results";

// API Keys
var openAiKey = configuration["OpenAI:ApiKey"]
    ?? throw new InvalidOperationException("OpenAI:ApiKey not found");
var anthropicKey = configuration["Anthropic:ApiKey"]
    ?? throw new InvalidOperationException("Anthropic:ApiKey not found");
var anthropicModel = configuration["Anthropic:Model"]
    ?? "claude-sonnet-4-5-20250929";
var googleKey = configuration["Google:ApiKey"]
    ?? throw new InvalidOperationException("Google:ApiKey not found");
var googleModel = configuration["Google:Model"]
    ?? "gemini-2.5-flash-lite";

// ═══════════════════════════════════════════════════════════════
// Gen1 Clients: 6 models (3 providers x 2 categories)
// Phase 1: GPT-4.1, Claude Sonnet 4.5, Gemini 2.5 Flash-Lite
// Phase 2: o4-mini, Claude Sonnet 4.5 (thinking), Gemini 2.5 Flash
// ═══════════════════════════════════════════════════════════════

var gen1Clients = new ILlmClient[]
{
    // Standard models
    new OpenAiLlmClient(openAiKey, "gpt-4.1", "standard"),
    new AnthropicLlmClient(anthropicKey, anthropicModel, "standard"),
    new GoogleLlmClient(googleKey, googleModel, "standard"),

    // Thinking models
    new OpenAiLlmClient(openAiKey, "o4-mini", "thinking"),
    new AnthropicLlmClient(anthropicKey, anthropicModel, "thinking"),
    new GoogleLlmClient(googleKey, "gemini-2.5-flash", "thinking"),
};

// ═══════════════════════════════════════════════════════════════
// Gen2 Clients: 6 models (3 providers x 2 categories), tier-aligned with Gen1
// Phase 3a: GPT-5.2, Claude Sonnet 4.6, Gemini 3.1 Flash-Lite (standard)
// Phase 3b: GPT-5.2 (thinking), Sonnet 4.6 (thinking), Flash-Lite 3.1 (thinking)
// ═══════════════════════════════════════════════════════════════

var gen2Clients = new ILlmClient[]
{
    // Standard models (reasoning disabled / minimal thinking)
    new OpenAiHttpLlmClient(openAiKey, "gpt-5.2", "standard",
        displayName: "GPT-5.2", reasoningEffort: "none"),
    new AnthropicLlmClient(anthropicKey, "claude-sonnet-4-6", "standard",
        displayName: "Claude Sonnet 4.6"),
    new GoogleLlmClient(googleKey, "gemini-3.1-flash-lite-preview", "standard",
        displayName: "Gemini 3.1 Flash-Lite", thinkingLevel: "minimal"),

    // Thinking models (reasoning enabled)
    new OpenAiHttpLlmClient(openAiKey, "gpt-5.2", "thinking",
        displayName: "GPT-5.2 (thinking)", reasoningEffort: "high"),
    new AnthropicLlmClient(anthropicKey, "claude-sonnet-4-6", "thinking",
        displayName: "Claude Sonnet 4.6 (thinking)", useAdaptiveThinking: true),
    new GoogleLlmClient(googleKey, "gemini-3.1-flash-lite-preview", "thinking",
        displayName: "Gemini 3.1 Flash-Lite (thinking)", thinkingLevel: "high"),
};

// Combined arrays for convenience
var allClients = gen1Clients.Concat(gen2Clients).ToArray();
var gen1Standard = gen1Clients.Where(c => c.Category == "standard").ToArray();
var gen1Thinking = gen1Clients.Where(c => c.Category == "thinking").ToArray();
var gen2Standard = gen2Clients.Where(c => c.Category == "standard").ToArray();
var gen2Thinking = gen2Clients.Where(c => c.Category == "thinking").ToArray();

// Set up EF Core DataContext
var optionsBuilder = new DbContextOptionsBuilder<DataContext>();
optionsBuilder.UseNpgsql(connectionString);
using var context = new DataContext(optionsBuilder.Options);

// Initialize services
var dbService = new DatabaseService(context);
var promptBuilder = new PromptBuilder();
var testPromptGenerator = new TestPromptGenerator();
var metricsCalculator = new MetricsCalculator();

Console.Write("Loading all tool names from database... ");
var allToolNames = await dbService.GetAllToolNamesAsync();
Console.WriteLine($"{allToolNames.Count} tools found.");

var hallucinationDetector = new HallucinationDetector(allToolNames);

var runner = new EvaluationRunner(
    dbService, promptBuilder, testPromptGenerator,
    hallucinationDetector, metricsCalculator,
    repetitions, outputDir);

// ═══════════════════════════════════════════════════════════════
// CLI Argument Mode (for parallel execution)
// Usage: dotnet run -- --provider openai --category thinking --configs C0,C5 --domains D1 --output phase2 --reps 3 --generation gen2
// ═══════════════════════════════════════════════════════════════

if (args.Length > 0)
{
    var cliProvider = GetArg(args, "--provider");
    var cliCategory = GetArg(args, "--category") ?? "standard";
    var cliConfigs = GetArg(args, "--configs")?.Split(',') ?? ["C0", "C5"];
    var cliDomains = GetArg(args, "--domains")?.Split(',');
    var cliOutput = GetArg(args, "--output") ?? "custom";
    var cliReps = int.TryParse(GetArg(args, "--reps"), out var r) ? r : repetitions;
    var cliGeneration = GetArg(args, "--generation") ?? "gen1";

    if (cliProvider == null)
    {
        Console.WriteLine("ERROR: --provider is required (openai, anthropic, google)");
        return;
    }

    var clientPool = cliGeneration == "gen2" ? gen2Clients : gen1Clients;

    var client = clientPool.FirstOrDefault(c =>
        c.ProviderId == cliProvider && c.Category == cliCategory);

    if (client == null)
    {
        Console.WriteLine($"ERROR: No client found for provider={cliProvider}, category={cliCategory}, generation={cliGeneration}");
        Console.WriteLine("Available: " + string.Join(", ", clientPool.Select(c => $"{c.ProviderId}/{c.Category} ({c.DisplayName})")));
        return;
    }

    Console.WriteLine($"  CLI MODE: {client.DisplayName} [{cliGeneration}] | configs=[{string.Join(",", cliConfigs)}] | domains=[{(cliDomains != null ? string.Join(",", cliDomains) : "ALL")}] | reps={cliReps} | output={cliOutput}");

    var cliRunner = new EvaluationRunner(
        dbService, promptBuilder, testPromptGenerator,
        hallucinationDetector, metricsCalculator,
        cliReps, Path.Combine(outputDir, cliOutput));

    await cliRunner.RunExperimentAsync(client, cliConfigs, cliDomains);
    Console.WriteLine("\n  CLI run complete.");
    return;
}

static string? GetArg(string[] args, string name)
{
    var idx = Array.IndexOf(args, name);
    return idx >= 0 && idx + 1 < args.Length ? args[idx + 1] : null;
}

// ═══════════════════════════════════════════════════════════════
// Interactive Menu
// ═══════════════════════════════════════════════════════════════

while (true)
{
    Console.WriteLine();
    Console.WriteLine("═══════════════════════════════════════════════════════════════");
    Console.WriteLine("  CADCOMONLINE - MULTI-MODEL ANTI-HALLUCINATION EVALUATION");
    Console.WriteLine("  Cross-Provider Tier-Aligned Evaluation");
    Console.WriteLine("═══════════════════════════════════════════════════════════════");
    Console.WriteLine();
    Console.WriteLine("  PILOT / QUICK TESTS:");
    Console.WriteLine("  1. Pilot: C0 vs C5, 1 rep, ALL 12 models (~$25, ~60 min)");
    Console.WriteLine("  2. Single model test (choose model, domain, config)");
    Console.WriteLine();
    Console.WriteLine("  GEN1 EXPERIMENT (GPT-4.1, Claude Sonnet 4.5, Gemini 2.5):");
    Console.WriteLine("  3. Phase 1: Gen1 standard models, full C0-C5 (2,592 calls)");
    Console.WriteLine("  4. Phase 2: Gen1 thinking models, C0 vs C5 (864 calls)");
    Console.WriteLine("  5. Gen1 full: Phase 1 + Phase 2 (3,456 calls)");
    Console.WriteLine();
    Console.WriteLine("  GEN2 EXPERIMENT (GPT-5.2, Sonnet 4.6, Gemini 3.1 Flash-Lite):");
    Console.WriteLine("  6. Phase 3a: Gen2 standard models, full C0-C5 (2,592 calls)");
    Console.WriteLine("  7. Phase 3b: Gen2 thinking models, C0 vs C5 (864 calls)");
    Console.WriteLine("  8. Gen2 full: Phase 3a + Phase 3b (3,456 calls)");
    Console.WriteLine();
    Console.WriteLine("  COMPLETE:");
    Console.WriteLine("  9. ALL phases: Gen1 + Gen2 (6,912 calls)");
    Console.WriteLine();
    Console.WriteLine("  UTILITIES:");
    Console.WriteLine("  10. Custom run (select models, configs, domains)");
    Console.WriteLine("  11. List domains and tools");
    Console.WriteLine("  12. Preview prompts for a domain");
    Console.WriteLine("   0. Exit");
    Console.WriteLine();
    Console.Write("  Select option: ");

    var choice = Console.ReadLine()?.Trim();

    try
    {
        switch (choice)
        {
            case "1":
                await RunPilotAsync();
                break;

            case "2":
                await RunSingleModelTestAsync();
                break;

            case "3":
                Console.WriteLine("\n  Running Phase 1: 3 Gen1 standard models x C0-C5...");
                var p1 = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase1"));
                await p1.RunMultiModelExperimentAsync(gen1Standard);
                break;

            case "4":
                Console.WriteLine("\n  Running Phase 2: 3 Gen1 thinking models x C0 vs C5...");
                var p2 = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase2"));
                await p2.RunMultiModelExperimentAsync(gen1Thinking, specificConfigs: ["C0", "C5"]);
                break;

            case "5":
                Console.WriteLine("\n  Running Gen1 full experiment (Phase 1 + Phase 2)...");
                var g1a = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase1"));
                await g1a.RunMultiModelExperimentAsync(gen1Standard);

                var g1b = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase2"));
                await g1b.RunMultiModelExperimentAsync(gen1Thinking, specificConfigs: ["C0", "C5"]);
                break;

            case "6":
                Console.WriteLine("\n  Running Phase 3a: 3 Gen2 standard models x C0-C5...");
                var p3a = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase3a"));
                await p3a.RunMultiModelExperimentAsync(gen2Standard);
                break;

            case "7":
                Console.WriteLine("\n  Running Phase 3b: 3 Gen2 thinking models x C0 vs C5...");
                var p3b = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase3b"));
                await p3b.RunMultiModelExperimentAsync(gen2Thinking, specificConfigs: ["C0", "C5"]);
                break;

            case "8":
                Console.WriteLine("\n  Running Gen2 full experiment (Phase 3a + Phase 3b)...");
                var g2a = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase3a"));
                await g2a.RunMultiModelExperimentAsync(gen2Standard);

                var g2b = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase3b"));
                await g2b.RunMultiModelExperimentAsync(gen2Thinking, specificConfigs: ["C0", "C5"]);
                break;

            case "9":
                Console.WriteLine("\n  Running ALL phases (Gen1 + Gen2 = 6,912 calls)...");
                Console.WriteLine("  This will take several hours. Continue? (y/n): ");
                if (Console.ReadLine()?.Trim().ToLower() != "y") break;

                Console.WriteLine("\n  === Phase 1: Gen1 Standard (C0-C5) ===");
                var all1 = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase1"));
                await all1.RunMultiModelExperimentAsync(gen1Standard);

                Console.WriteLine("\n  === Phase 2: Gen1 Thinking (C0, C5) ===");
                var all2 = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase2"));
                await all2.RunMultiModelExperimentAsync(gen1Thinking, specificConfigs: ["C0", "C5"]);

                Console.WriteLine("\n  === Phase 3a: Gen2 Standard (C0-C5) ===");
                var all3a = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase3a"));
                await all3a.RunMultiModelExperimentAsync(gen2Standard);

                Console.WriteLine("\n  === Phase 3b: Gen2 Thinking (C0, C5) ===");
                var all3b = new EvaluationRunner(
                    dbService, promptBuilder, testPromptGenerator,
                    hallucinationDetector, metricsCalculator,
                    repetitions, Path.Combine(outputDir, "phase3b"));
                await all3b.RunMultiModelExperimentAsync(gen2Thinking, specificConfigs: ["C0", "C5"]);

                Console.WriteLine("\n  ALL phases complete!");
                break;

            case "10":
                await RunCustomExperimentAsync();
                break;

            case "11":
                await ListDomainsAndToolsAsync(dbService);
                break;

            case "12":
                Console.Write("  Domain (D1/D2/D3/D4): ");
                var previewDomain = Console.ReadLine()?.Trim() ?? "D1";
                await PreviewPromptsAsync(dbService, testPromptGenerator, promptBuilder, previewDomain);
                break;

            case "0":
                Console.WriteLine("  Exiting.");
                return;

            default:
                Console.WriteLine("  Invalid option.");
                break;
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"\n  ERROR: {ex.Message}");
        Console.WriteLine($"  {ex.StackTrace}");
    }
}

// ═══════════════════════════════════════════════════════════════
// Menu Action Methods
// ═══════════════════════════════════════════════════════════════

async Task RunPilotAsync()
{
    Console.WriteLine("\n  PILOT: C0 vs C5, 1 rep, all 12 models (Gen1 + Gen2)");
    Console.Write("  Continue? (y/n): ");
    if (Console.ReadLine()?.Trim().ToLower() != "y") return;

    var pilotRunner = new EvaluationRunner(
        dbService, promptBuilder, testPromptGenerator,
        hallucinationDetector, metricsCalculator,
        repetitions: 1, outputDir: Path.Combine(outputDir, "pilot"));

    await pilotRunner.RunMultiModelExperimentAsync(allClients, specificConfigs: ["C0", "C5"]);
}

async Task RunSingleModelTestAsync()
{
    Console.WriteLine("\n  Available models:");
    Console.WriteLine("  --- Gen1 ---");
    for (int i = 0; i < gen1Clients.Length; i++)
        Console.WriteLine($"    {i + 1}. {gen1Clients[i].DisplayName} ({gen1Clients[i].ProviderId}/{gen1Clients[i].Category})");

    Console.WriteLine("  --- Gen2 ---");
    for (int i = 0; i < gen2Clients.Length; i++)
        Console.WriteLine($"    {gen1Clients.Length + i + 1}. {gen2Clients[i].DisplayName} ({gen2Clients[i].ProviderId}/{gen2Clients[i].Category})");

    Console.Write($"  Select model (1-{allClients.Length}): ");
    var modelIdx = int.Parse(Console.ReadLine()?.Trim() ?? "1") - 1;
    if (modelIdx < 0 || modelIdx >= allClients.Length) { Console.WriteLine("  Invalid."); return; }

    Console.Write("  Domain (D1/D2/D3/D4): ");
    var d = Console.ReadLine()?.Trim() ?? "D1";
    Console.Write("  Config (C0-C5): ");
    var c = Console.ReadLine()?.Trim() ?? "C5";
    Console.Write("  Prompt index (0-11): ");
    var p = int.Parse(Console.ReadLine()?.Trim() ?? "0");

    await runner.RunSingleTestAsync(allClients[modelIdx], d, c, p);
}

async Task RunCustomExperimentAsync()
{
    Console.WriteLine("\n  Select models to include:");
    Console.WriteLine("  --- Gen1 ---");
    for (int i = 0; i < gen1Clients.Length; i++)
        Console.WriteLine($"    {i + 1}. {gen1Clients[i].DisplayName} ({gen1Clients[i].ProviderId}/{gen1Clients[i].Category})");

    Console.WriteLine("  --- Gen2 ---");
    for (int i = 0; i < gen2Clients.Length; i++)
        Console.WriteLine($"    {gen1Clients.Length + i + 1}. {gen2Clients[i].DisplayName} ({gen2Clients[i].ProviderId}/{gen2Clients[i].Category})");

    Console.Write($"  Enter model numbers (comma-separated, e.g., 1,2,3): ");
    var modelInput = Console.ReadLine()?.Trim() ?? "1";
    var selectedClients = modelInput.Split(',')
        .Select(s => int.Parse(s.Trim()) - 1)
        .Where(i => i >= 0 && i < allClients.Length)
        .Select(i => allClients[i])
        .ToArray();

    Console.Write("  Configs (comma-separated, e.g., C0,C5 or all): ");
    var configInput = Console.ReadLine()?.Trim() ?? "C0,C5";
    string[]? configs = configInput.ToLower() == "all" ? null : configInput.Split(',').Select(s => s.Trim()).ToArray();

    Console.Write("  Domains (comma-separated, e.g., D1,D2 or all): ");
    var domainInput = Console.ReadLine()?.Trim() ?? "all";
    string[]? domains = domainInput.ToLower() == "all" ? null : domainInput.Split(',').Select(s => s.Trim()).ToArray();

    Console.Write("  Repetitions (1-5): ");
    var reps = int.Parse(Console.ReadLine()?.Trim() ?? "1");

    Console.Write("  Output subdirectory (default: custom): ");
    var subDir = Console.ReadLine()?.Trim();
    if (string.IsNullOrEmpty(subDir)) subDir = "custom";

    var customRunner = new EvaluationRunner(
        dbService, promptBuilder, testPromptGenerator,
        hallucinationDetector, metricsCalculator,
        reps, Path.Combine(outputDir, subDir));

    await customRunner.RunMultiModelExperimentAsync(selectedClients, configs, domains);
}

// ═══════════════════════════════════════════════════════════════
// Helper Methods
// ═══════════════════════════════════════════════════════════════

static async Task ListDomainsAndToolsAsync(DatabaseService dbService)
{
    foreach (var (id, subcategoryName) in EvaluationRunner.DomainDefinitions)
    {
        var domain = await dbService.LoadDomainAsync(id, subcategoryName);
        Console.WriteLine($"\n  {id}: {domain.Name} ({domain.CategoryName})");
        Console.WriteLine($"  Tools ({domain.Tools.Count}):");
        foreach (var tool in domain.Tools)
        {
            Console.WriteLine($"    - {tool.Name} (ID: {tool.Id}, Features: {tool.Features.Count})");
        }
    }
}

static async Task PreviewPromptsAsync(
    DatabaseService dbService, TestPromptGenerator generator,
    PromptBuilder promptBuilder, string domainId)
{
    var domainDef = EvaluationRunner.DomainDefinitions.First(d => d.Id == domainId);
    var domain = await dbService.LoadDomainAsync(domainDef.Id, domainDef.SubcategoryName);
    var prompts = generator.GeneratePrompts(domain);
    var relatedTools = await dbService.GetRelatedToolsAsync(domain.Tools[0].Id);

    Console.WriteLine($"\n  Domain: {domain.Id} ({domain.Name})");
    Console.WriteLine($"  Primary tool: {domain.Tools[0].Name}");
    Console.WriteLine($"  Related tools: {relatedTools.Count}");
    Console.WriteLine($"\n  12 Test Prompts:");

    for (int i = 0; i < prompts.Count; i++)
    {
        Console.WriteLine($"    P{i + 1:D2}: {prompts[i]}");
    }

    Console.Write("\n  Preview full prompt for which config? (C0-C5, or skip): ");
    var configChoice = Console.ReadLine()?.Trim();
    if (configChoice != null && configChoice.StartsWith("C"))
    {
        var config = AblationConfig.AllConfigs.First(c => c.Id == configChoice);
        var fullPrompt = promptBuilder.BuildPrompt(config, domain, domain.Tools[0], domain.Tools, relatedTools, prompts[0]);
        Console.WriteLine($"\n  --- Full prompt for {config.Id} ({config.Name}) ---");
        Console.WriteLine(fullPrompt);
        Console.WriteLine($"\n  Prompt length: {fullPrompt.Length} characters");
    }
}
