using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;

namespace CadcomOnline.Evaluation.Clients;

/// <summary>
/// HTTP-based OpenAI client for GPT-5.2 with reasoning_effort support.
/// Uses direct HTTP instead of the OpenAI SDK to support the reasoning_effort parameter.
/// GPT-5.2 uses max_completion_tokens (not max_tokens) and top-level reasoning_effort string.
/// </summary>
public class OpenAiHttpLlmClient : ILlmClient
{
    private readonly HttpClient _httpClient;
    private readonly string _modelId;
    private readonly string? _reasoningEffort;
    private const string ApiUrl = "https://api.openai.com/v1/chat/completions";

    public string ProviderId => "openai";
    public string ModelId => _modelId;
    public string Category { get; }
    public string DisplayName { get; }

    /// <param name="apiKey">OpenAI API key</param>
    /// <param name="modelId">Model ID (e.g., "gpt-5.2")</param>
    /// <param name="category">"standard" or "thinking"</param>
    /// <param name="displayName">Display name for reports</param>
    /// <param name="reasoningEffort">Reasoning effort level: "low", "medium", "high". Null or "none" = omit parameter (standard mode).</param>
    public OpenAiHttpLlmClient(string apiKey, string modelId, string category,
        string? displayName = null, string? reasoningEffort = null)
    {
        _modelId = modelId;
        Category = category;
        DisplayName = displayName ?? modelId;
        _reasoningEffort = reasoningEffort;

        _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(300) };
        _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
    }

    public async Task<LlmResponse> ChatAsync(string systemPrompt, string userPrompt)
    {
        var sw = Stopwatch.StartNew();

        var userContent = string.IsNullOrEmpty(userPrompt)
            ? "Please provide your analysis."
            : userPrompt;

        var messages = new List<object>();

        if (!string.IsNullOrEmpty(systemPrompt))
            messages.Add(new { role = "system", content = systemPrompt });

        messages.Add(new { role = "user", content = userContent });

        // Build request body - GPT-5.2 uses max_completion_tokens and top-level reasoning_effort
        // For standard mode (effort=none or null), omit reasoning_effort entirely
        object requestBody;

        if (_reasoningEffort != null && _reasoningEffort != "none")
        {
            requestBody = new
            {
                model = _modelId,
                messages = messages,
                max_completion_tokens = 16000,
                reasoning_effort = _reasoningEffort
            };
        }
        else
        {
            requestBody = new
            {
                model = _modelId,
                messages = messages,
                max_completion_tokens = 4096
            };
        }

        var response = await _httpClient.PostAsJsonAsync(ApiUrl, requestBody);
        var responseBody = await response.Content.ReadAsStringAsync();
        sw.Stop();

        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException(
                $"OpenAI API error {response.StatusCode}: {responseBody}");

        var result = JsonSerializer.Deserialize<JsonElement>(responseBody);

        // Extract text from response
        var text = "";
        try
        {
            text = result
                .GetProperty("choices")[0]
                .GetProperty("message")
                .GetProperty("content")
                .GetString() ?? "";
        }
        catch
        {
            text = "";
        }

        // Parse token usage
        var inputTokens = 0;
        var outputTokens = 0;
        var thinkingTokens = 0;

        if (result.TryGetProperty("usage", out var usage))
        {
            if (usage.TryGetProperty("prompt_tokens", out var promptCount))
                inputTokens = promptCount.GetInt32();

            if (usage.TryGetProperty("completion_tokens", out var completionCount))
                outputTokens = completionCount.GetInt32();

            // GPT-5.2 reports reasoning tokens in completion_tokens_details
            if (usage.TryGetProperty("completion_tokens_details", out var details))
            {
                if (details.TryGetProperty("reasoning_tokens", out var reasoningCount))
                {
                    thinkingTokens = reasoningCount.GetInt32();
                    // Subtract reasoning tokens from output to get visible output only
                    outputTokens = Math.Max(0, outputTokens - thinkingTokens);
                }
            }
        }

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
}
