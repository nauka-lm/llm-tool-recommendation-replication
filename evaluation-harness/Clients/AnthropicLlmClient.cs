using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;

namespace CadcomOnline.Evaluation.Clients;

/// <summary>
/// Anthropic LLM client supporting both standard and thinking modes.
/// Gen1: Claude Sonnet 4.5 with thinking.type: "enabled" + budget_tokens.
/// Gen2: Claude Sonnet 4.6 with thinking.type: "adaptive" + output_config.effort.
/// Uses the same model ID for both modes; thinking is enabled via API parameter.
/// </summary>
public class AnthropicLlmClient : ILlmClient
{
    private readonly HttpClient _httpClient;
    private readonly string _modelId;
    private readonly bool _useThinking;
    private readonly bool _useAdaptiveThinking;
    private const string ApiUrl = "https://api.anthropic.com/v1/messages";

    public string ProviderId => "anthropic";
    public string ModelId => _modelId;
    public string Category { get; }
    public string DisplayName { get; }

    /// <param name="apiKey">Anthropic API key</param>
    /// <param name="modelId">Model ID (e.g., "claude-sonnet-4-5-20250929" or "claude-sonnet-4-6")</param>
    /// <param name="category">"standard" or "thinking"</param>
    /// <param name="displayName">Optional display name override for reports</param>
    /// <param name="useAdaptiveThinking">Use Gen2 adaptive thinking format (Sonnet 4.6) instead of legacy budget_tokens</param>
    public AnthropicLlmClient(string apiKey, string modelId, string category,
        string? displayName = null, bool useAdaptiveThinking = false)
    {
        _modelId = modelId;
        Category = category;
        _useThinking = category == "thinking";
        _useAdaptiveThinking = useAdaptiveThinking;

        // Default display names based on model if not provided
        if (displayName != null)
        {
            DisplayName = displayName;
        }
        else
        {
            DisplayName = _useThinking ? "Claude Sonnet 4.5 (thinking)" : "Claude Sonnet 4.5";
        }

        _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(300) };
        _httpClient.DefaultRequestHeaders.Add("x-api-key", apiKey);
        _httpClient.DefaultRequestHeaders.Add("anthropic-version", "2023-06-01");
    }

    public async Task<LlmResponse> ChatAsync(string systemPrompt, string userPrompt)
    {
        var sw = Stopwatch.StartNew();

        // Build the user message - combine system prompt into user content
        // Anthropic's Messages API uses system as a top-level parameter
        var userContent = string.IsNullOrEmpty(userPrompt)
            ? "Please provide your analysis."
            : userPrompt;

        object requestBody;

        if (_useThinking && _useAdaptiveThinking)
        {
            // Sonnet 4.6 adaptive thinking: type="adaptive" + output_config.effort
            requestBody = new
            {
                model = _modelId,
                max_tokens = 16000,
                thinking = new
                {
                    type = "adaptive"
                },
                output_config = new
                {
                    effort = "high"
                },
                system = systemPrompt,
                messages = new[]
                {
                    new { role = "user", content = userContent }
                }
            };
        }
        else if (_useThinking)
        {
            // Sonnet 4.5 legacy thinking: type="enabled" + budget_tokens
            requestBody = new
            {
                model = _modelId,
                max_tokens = 16000,
                thinking = new
                {
                    type = "enabled",
                    budget_tokens = 10000
                },
                system = systemPrompt,
                messages = new[]
                {
                    new { role = "user", content = userContent }
                }
            };
        }
        else
        {
            // Standard mode (no thinking)
            requestBody = new
            {
                model = _modelId,
                max_tokens = 4096,
                system = systemPrompt,
                messages = new[]
                {
                    new { role = "user", content = userContent }
                }
            };
        }

        var response = await _httpClient.PostAsJsonAsync(ApiUrl, requestBody);
        var responseBody = await response.Content.ReadAsStringAsync();
        sw.Stop();

        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException(
                $"Anthropic API error {response.StatusCode}: {responseBody}");

        var result = JsonSerializer.Deserialize<JsonElement>(responseBody);

        // Parse response text and thinking tokens
        // Accumulate across multiple blocks (Gen2 adaptive thinking may use interleaved thinking)
        var text = "";
        var thinkingText = "";

        var contentArray = result.GetProperty("content");
        foreach (var block in contentArray.EnumerateArray())
        {
            var blockType = block.GetProperty("type").GetString();
            if (blockType == "thinking")
            {
                // Thinking block contains internal reasoning
                thinkingText += block.GetProperty("thinking").GetString() ?? "";
            }
            else if (blockType == "text")
            {
                text += block.GetProperty("text").GetString() ?? "";
            }
        }

        var thinkingTokens = thinkingText.Length > 0 ? EstimateTokens(thinkingText) : 0;

        // Parse usage
        var usage = result.GetProperty("usage");
        var inputTokens = usage.GetProperty("input_tokens").GetInt32();
        var outputTokens = usage.GetProperty("output_tokens").GetInt32();

        // For thinking mode, output_tokens includes thinking tokens
        // Subtract to get visible output only
        if (_useThinking && thinkingTokens > 0)
            outputTokens = Math.Max(0, outputTokens - thinkingTokens);

        return new LlmResponse(
            Text: text,
            InputTokens: inputTokens,
            OutputTokens: outputTokens,
            ThinkingTokens: thinkingTokens,
            LatencyMs: sw.Elapsed.TotalMilliseconds,
            ModelId: _modelId,
            ProviderId: ProviderId,
            Category: Category
        );
    }

    /// <summary>
    /// Rough token estimate for thinking text (Anthropic doesn't report thinking token count separately in all cases).
    /// ~4 characters per token is a reasonable approximation.
    /// </summary>
    private static int EstimateTokens(string text) => Math.Max(1, text.Length / 4);
}
