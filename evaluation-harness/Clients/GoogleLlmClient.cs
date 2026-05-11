using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;

namespace CadcomOnline.Evaluation.Clients;

/// <summary>
/// Google Gemini LLM client supporting both standard and thinking modes.
/// Gen1: Gemini 2.5 Flash-Lite (standard) / Flash (thinking) - uses default config.
/// Gen2: Gemini 3 Flash with thinkingLevel parameter ("minimal" for standard, "high" for thinking).
/// Uses direct HTTP against the Gemini REST API.
/// </summary>
public class GoogleLlmClient : ILlmClient
{
    private readonly HttpClient _httpClient;
    private readonly string _apiKey;
    private readonly string _modelId;
    private readonly string? _thinkingLevel;
    private const string BaseUrl = "https://generativelanguage.googleapis.com/v1beta/models";

    public string ProviderId => "google";
    public string ModelId => _modelId;
    public string Category { get; }
    public string DisplayName { get; }

    /// <param name="apiKey">Google AI API key</param>
    /// <param name="modelId">Model ID (e.g., "gemini-2.5-flash-lite" or "gemini-3.1-flash-lite-preview")</param>
    /// <param name="category">"standard" or "thinking"</param>
    /// <param name="displayName">Optional display name override for reports</param>
    /// <param name="thinkingLevel">Gemini 3 thinking level: "minimal", "low", "medium", "high". Null = omit (gen1 behavior).</param>
    public GoogleLlmClient(string apiKey, string modelId, string category,
        string? displayName = null, string? thinkingLevel = null)
    {
        _apiKey = apiKey;
        _modelId = modelId;
        Category = category;
        _thinkingLevel = thinkingLevel;

        // Default display names based on model if not provided
        if (displayName != null)
        {
            DisplayName = displayName;
        }
        else
        {
            DisplayName = category == "thinking" ? "Gemini 2.5 Flash" : "Gemini 2.5 Flash-Lite";
        }

        _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(300) };
    }

    public async Task<LlmResponse> ChatAsync(string systemPrompt, string userPrompt)
    {
        var sw = Stopwatch.StartNew();

        var userContent = string.IsNullOrEmpty(userPrompt)
            ? "Please provide your analysis."
            : userPrompt;

        // Build request body - differs based on whether thinkingLevel is set (Gemini 3)
        object requestBody;

        if (_thinkingLevel != null)
        {
            // Gemini 3 Flash: use thinkingConfig with thinkingLevel
            requestBody = new
            {
                systemInstruction = new
                {
                    parts = new[] { new { text = systemPrompt } }
                },
                contents = new[]
                {
                    new
                    {
                        role = "user",
                        parts = new[] { new { text = userContent } }
                    }
                },
                generationConfig = new
                {
                    maxOutputTokens = 8192,
                    temperature = 0.7,
                    thinkingConfig = new
                    {
                        thinkingLevel = _thinkingLevel
                    }
                }
            };
        }
        else
        {
            // Gemini 2.5: standard config without thinkingConfig
            requestBody = new
            {
                systemInstruction = new
                {
                    parts = new[] { new { text = systemPrompt } }
                },
                contents = new[]
                {
                    new
                    {
                        role = "user",
                        parts = new[] { new { text = userContent } }
                    }
                },
                generationConfig = new
                {
                    maxOutputTokens = 4096,
                    temperature = 0.7
                }
            };
        }

        var url = $"{BaseUrl}/{_modelId}:generateContent?key={_apiKey}";
        var response = await _httpClient.PostAsJsonAsync(url, requestBody);
        var responseBody = await response.Content.ReadAsStringAsync();
        sw.Stop();

        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException(
                $"Google API error {response.StatusCode}: {responseBody}");

        var result = JsonSerializer.Deserialize<JsonElement>(responseBody);

        // Extract text from response
        var text = "";
        try
        {
            text = result
                .GetProperty("candidates")[0]
                .GetProperty("content")
                .GetProperty("parts")[0]
                .GetProperty("text")
                .GetString() ?? "";
        }
        catch
        {
            // Some responses may have different structure (e.g., safety filtered)
            text = "";
        }

        // Parse token usage from usageMetadata
        var inputTokens = 0;
        var outputTokens = 0;
        var thinkingTokens = 0;

        if (result.TryGetProperty("usageMetadata", out var usage))
        {
            if (usage.TryGetProperty("promptTokenCount", out var promptCount))
                inputTokens = promptCount.GetInt32();

            if (usage.TryGetProperty("candidatesTokenCount", out var candidatesCount))
                outputTokens = candidatesCount.GetInt32();

            // Gemini reports thinking tokens as thoughtsTokenCount
            if (usage.TryGetProperty("thoughtsTokenCount", out var thoughtsCount))
            {
                thinkingTokens = thoughtsCount.GetInt32();
                // Subtract thinking tokens from output to get visible output only
                outputTokens = Math.Max(0, outputTokens - thinkingTokens);
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
