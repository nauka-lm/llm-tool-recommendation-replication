namespace CadcomOnline.Evaluation.Clients;

/// <summary>
/// Unified interface for calling different LLM providers.
/// Each implementation handles one provider, supporting both standard and thinking modes.
/// </summary>
public interface ILlmClient
{
    /// <summary>Provider identifier: "openai", "anthropic", "google"</summary>
    string ProviderId { get; }

    /// <summary>Model identifier: "gpt-4.1", "o4-mini", "claude-sonnet-4-5-20250929", etc.</summary>
    string ModelId { get; }

    /// <summary>Model category: "standard" or "thinking"</summary>
    string Category { get; }

    /// <summary>Display name for reports: "GPT-4.1", "o4-mini", "Claude Sonnet 4.5", etc.</summary>
    string DisplayName { get; }

    /// <summary>
    /// Sends a prompt to the LLM and returns the response with token usage.
    /// For the experiment, systemPrompt contains the full ablation-configured prompt,
    /// and userPrompt is empty (the user query is embedded in systemPrompt by PromptBuilder).
    /// </summary>
    Task<LlmResponse> ChatAsync(string systemPrompt, string userPrompt);
}

/// <summary>
/// Response from any LLM provider, normalized to a common format.
/// </summary>
public record LlmResponse(
    /// <summary>The generated text response</summary>
    string Text,

    /// <summary>Number of input/prompt tokens</summary>
    int InputTokens,

    /// <summary>Number of output/completion tokens (excludes thinking tokens)</summary>
    int OutputTokens,

    /// <summary>Number of thinking/reasoning tokens (0 for standard models)</summary>
    int ThinkingTokens,

    /// <summary>End-to-end latency in milliseconds</summary>
    double LatencyMs,

    /// <summary>Model ID used for the call</summary>
    string ModelId,

    /// <summary>Provider ID: "openai", "anthropic", "google"</summary>
    string ProviderId,

    /// <summary>Category: "standard" or "thinking"</summary>
    string Category
);
