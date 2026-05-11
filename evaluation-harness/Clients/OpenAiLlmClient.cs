using System.Diagnostics;
using OpenAI.Chat;

namespace CadcomOnline.Evaluation.Clients;

/// <summary>
/// OpenAI LLM client supporting both GPT-4.1 (standard) and o4-mini (thinking).
/// Uses the official OpenAI .NET SDK.
/// </summary>
public class OpenAiLlmClient : ILlmClient
{
    private readonly ChatClient _chatClient;

    public string ProviderId => "openai";
    public string ModelId { get; }
    public string Category { get; }
    public string DisplayName { get; }

    public OpenAiLlmClient(string apiKey, string modelId, string category)
    {
        ModelId = modelId;
        Category = category;
        DisplayName = category == "thinking" ? "o4-mini" : "GPT-4.1";
        _chatClient = new ChatClient(modelId, apiKey);
    }

    public async Task<LlmResponse> ChatAsync(string systemPrompt, string userPrompt)
    {
        var sw = Stopwatch.StartNew();

        // Build messages - use system message for standard, developer message concept for thinking
        // The EvaluationRunner puts everything into systemPrompt (the full ablation-configured prompt)
        var messages = new List<ChatMessage>();

        if (!string.IsNullOrEmpty(systemPrompt))
            messages.Add(ChatMessage.CreateSystemMessage(systemPrompt));

        if (!string.IsNullOrEmpty(userPrompt))
            messages.Add(ChatMessage.CreateUserMessage(userPrompt));
        else
            messages.Add(ChatMessage.CreateUserMessage("Please provide your analysis."));

        var response = await _chatClient.CompleteChatAsync(messages);
        sw.Stop();

        var text = "";
        if (response?.Value?.Content?.Count > 0)
            text = response.Value.Content[0].Text ?? "";

        var inputTokens = response?.Value?.Usage?.InputTokenCount ?? 0;
        var outputTokens = response?.Value?.Usage?.OutputTokenCount ?? 0;

        // For o4-mini, reasoning tokens are included in OutputTokenCount
        // They are tracked separately in CompletionTokenDetails
        var thinkingTokens = 0;
        if (Category == "thinking" && response?.Value?.Usage != null)
        {
            var details = response.Value.Usage.OutputTokenDetails;
            if (details != null)
            {
                thinkingTokens = details.ReasoningTokenCount;
                // Subtract reasoning tokens from output to get visible output only
                outputTokens = outputTokens - thinkingTokens;
            }
        }

        return new LlmResponse(
            Text: text,
            InputTokens: inputTokens,
            OutputTokens: outputTokens,
            ThinkingTokens: thinkingTokens,
            LatencyMs: sw.Elapsed.TotalMilliseconds,
            ModelId: ModelId,
            ProviderId: ProviderId,
            Category: Category
        );
    }
}
