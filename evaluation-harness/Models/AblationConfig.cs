namespace CadcomOnline.Evaluation.Models;

/// <summary>
/// Defines the 6 ablation configurations (C0-C5) for the experiment.
/// Each configuration toggles anti-hallucination mechanisms on/off.
/// </summary>
public class AblationConfig
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public bool UseContextBuilder { get; set; }
    public bool UseClosedVocabulary { get; set; }
    public bool UseJsonEnforcement { get; set; }

    public static readonly AblationConfig[] AllConfigs =
    [
        new()
        {
            Id = "C0",
            Name = "Ungrounded",
            Description = "Baseline (no anti-hallucination)",
            UseContextBuilder = false,
            UseClosedVocabulary = false,
            UseJsonEnforcement = false
        },
        new()
        {
            Id = "C1",
            Name = "Context only",
            Description = "DB context provided but no vocabulary constraint",
            UseContextBuilder = true,
            UseClosedVocabulary = false,
            UseJsonEnforcement = false
        },
        new()
        {
            Id = "C2",
            Name = "Vocabulary only",
            Description = "Tool whitelist but no rich context",
            UseContextBuilder = false,
            UseClosedVocabulary = true,
            UseJsonEnforcement = false
        },
        new()
        {
            Id = "C3",
            Name = "JSON only",
            Description = "Structured output but no grounding content",
            UseContextBuilder = false,
            UseClosedVocabulary = false,
            UseJsonEnforcement = true
        },
        new()
        {
            Id = "C4",
            Name = "Context + Vocabulary",
            Description = "Both grounding mechanisms, no format constraint",
            UseContextBuilder = true,
            UseClosedVocabulary = true,
            UseJsonEnforcement = false
        },
        new()
        {
            Id = "C5",
            Name = "Full architecture",
            Description = "All three mechanisms (our approach)",
            UseContextBuilder = true,
            UseClosedVocabulary = true,
            UseJsonEnforcement = true
        }
    ];

    public override string ToString() =>
        $"{Id}: {Name} [Context={UseContextBuilder}, Vocab={UseClosedVocabulary}, JSON={UseJsonEnforcement}]";
}
