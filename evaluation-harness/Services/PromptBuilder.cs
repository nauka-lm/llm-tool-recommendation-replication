using CadcomOnline.Evaluation.Models;

namespace CadcomOnline.Evaluation.Services;

/// <summary>
/// Builds prompts for all 6 ablation configurations (C0-C5).
/// Adapted from OpenAIService.BuildEnhancedPrompt() in production code.
/// </summary>
public class PromptBuilder
{
    /// <summary>
    /// Builds a system prompt for the given configuration, domain, and user query.
    /// </summary>
    public string BuildPrompt(
        AblationConfig config,
        Domain domain,
        ToolInfo primaryTool,
        List<ToolInfo> rankedTools,
        List<RelatedToolInfo> relatedTools,
        string userQuery)
    {
        return config.Id switch
        {
            "C0" => BuildC0Ungrounded(domain, primaryTool, userQuery),
            "C1" => BuildC1ContextOnly(domain, primaryTool, rankedTools, relatedTools, userQuery),
            "C2" => BuildC2VocabularyOnly(domain, primaryTool, rankedTools, relatedTools, userQuery),
            "C3" => BuildC3JsonOnly(domain, primaryTool, userQuery),
            "C4" => BuildC4ContextAndVocabulary(domain, primaryTool, rankedTools, relatedTools, userQuery),
            "C5" => BuildC5Full(domain, primaryTool, rankedTools, relatedTools, userQuery),
            _ => throw new ArgumentException($"Unknown config: {config.Id}")
        };
    }

    /// <summary>
    /// C0: Ungrounded baseline. No whitelist, no context, no JSON enforcement.
    /// Generic assistant with minimal info.
    /// </summary>
    private string BuildC0Ungrounded(Domain domain, ToolInfo primaryTool, string userQuery)
    {
        return $@"You are an engineering assistant helping with tool selection.

The user is working on a {domain.SubcategoryName} project and has selected {primaryTool.Name} as their primary tool.

User query: {userQuery}

Please provide your recommendation including:
- A recommended tool and why
- Strengths and limitations
- A workflow with complementary tools
- Practical tips for the user";
    }

    /// <summary>
    /// C1: Context only. Rich DB context provided but no vocabulary constraint.
    /// LLM sees all tool data but isn't told to restrict output to these tools.
    /// </summary>
    private string BuildC1ContextOnly(
        Domain domain, ToolInfo primaryTool, List<ToolInfo> rankedTools,
        List<RelatedToolInfo> relatedTools, string userQuery)
    {
        var expertDesc = BuildExpertDescription(domain);

        var prompt = $@"You are an expert in {expertDesc}.

The user is working on a {domain.SubcategoryName} project.

User query: {userQuery}

Here are the tools available in our platform:

PRIMARY TOOLS (ranked by user criteria):
";
        foreach (var tool in rankedTools)
        {
            prompt += FormatToolDetails(tool);
        }

        if (relatedTools.Any())
        {
            prompt += "\nRELATED TOOLS:\n";
            foreach (var tool in relatedTools)
            {
                prompt += FormatRelatedToolDetails(tool);
            }
        }

        prompt += @"

Please provide your recommendation including:
- A recommended tool and why
- Strengths and limitations
- A workflow with complementary tools
- Practical tips for the user";

        return prompt;
    }

    /// <summary>
    /// C2: Vocabulary only. Tool whitelist but no rich context.
    /// LLM is told ONLY to use these tool names, but doesn't get features/functions.
    /// </summary>
    private string BuildC2VocabularyOnly(
        Domain domain, ToolInfo primaryTool, List<ToolInfo> rankedTools,
        List<RelatedToolInfo> relatedTools, string userQuery)
    {
        var allToolNames = rankedTools.Select(t => t.Name)
            .Concat(relatedTools.Select(t => t.Name))
            .Distinct().ToList();
        var toolNamesList = string.Join(", ", allToolNames);

        return $@"You are an engineering assistant helping with tool selection for {domain.SubcategoryName}.

CRITICAL INSTRUCTION - VERIFIED TOOL DATABASE:
You MUST ONLY reference tools from our verified database listed below. DO NOT mention, suggest, or recommend ANY tools that are not explicitly listed here.

AVAILABLE TOOLS IN DATABASE: [{toolNamesList}]

Any tool name you mention in your response MUST be from this list. Do not invent or suggest tools outside this database.

The user selected {primaryTool.Name} as their primary tool.

User query: {userQuery}

STRICT: ONLY use tool names from: [{toolNamesList}]
Do NOT invent or suggest tools outside the database.

Please provide your recommendation including:
- A recommended tool and why
- Strengths and limitations
- A workflow with complementary tools
- Practical tips for the user";
    }

    /// <summary>
    /// C3: JSON only. Structured output but no grounding content.
    /// LLM must respond in JSON format but has no tool whitelist or rich context.
    /// </summary>
    private string BuildC3JsonOnly(Domain domain, ToolInfo primaryTool, string userQuery)
    {
        return $@"You are an engineering assistant helping with tool selection.

The user is working on a {domain.SubcategoryName} project and has selected {primaryTool.Name} as their primary tool.

User query: {userQuery}

Respond ONLY in this JSON format:

{{
  ""recommendation"": {{
    ""toolName"": ""The recommended tool"",
    ""verdict"": ""2-3 sentence expert verdict"",
    ""strengths"": [""strength 1"", ""strength 2"", ""strength 3"", ""strength 4""],
    ""limitations"": [""limitation 1"", ""limitation 2""],
    ""bestFor"": ""Ideal use case""
  }},
  ""comparison"": {{
    ""summary"": ""Compare relevant tools"",
    ""toolNotes"": [{{""tool"": ""Tool name"", ""note"": ""Key differentiator""}}]
  }},
  ""workflow"": {{
    ""title"": ""Recommended Design Workflow"",
    ""description"": ""Overview"",
    ""steps"": [
      {{""phase"": ""Phase name"", ""tool"": ""Tool name"", ""action"": ""Description""}},
      {{""phase"": ""Phase name"", ""tool"": ""Tool name"", ""action"": ""Description""}}
    ]
  }},
  ""tips"": [""tip 1"", ""tip 2"", ""tip 3"", ""tip 4""],
  ""integration"": {{
    ""fileFormats"": ""Supported formats"",
    ""compatibility"": ""Tool compatibility"",
    ""dataExchange"": ""Data exchange tips""
  }}
}}

Return ONLY valid JSON - no markdown, no code blocks, no extra text.";
    }

    /// <summary>
    /// C4: Context + Vocabulary. Both grounding mechanisms, no format constraint.
    /// LLM gets rich context AND is told to use only these tools, but free-form response.
    /// </summary>
    private string BuildC4ContextAndVocabulary(
        Domain domain, ToolInfo primaryTool, List<ToolInfo> rankedTools,
        List<RelatedToolInfo> relatedTools, string userQuery)
    {
        var expertDesc = BuildExpertDescription(domain);
        var allToolNames = rankedTools.Select(t => t.Name)
            .Concat(relatedTools.Select(t => t.Name))
            .Distinct().ToList();
        var toolNamesList = string.Join(", ", allToolNames);

        var prompt = $@"You are an expert in {expertDesc}.

CRITICAL INSTRUCTION - VERIFIED TOOL DATABASE:
You MUST ONLY reference tools from our verified database listed below. DO NOT mention, suggest, or recommend ANY tools that are not explicitly listed here. This is a curated database of tested and verified tools.

AVAILABLE TOOLS IN DATABASE: [{toolNamesList}]

Any tool name you mention in your response MUST be from this list. Do not invent or suggest tools outside this database.

PRIMARY TOOLS (ranked from best to worst match based on user criteria):
";
        foreach (var tool in rankedTools)
        {
            prompt += FormatToolDetails(tool);
        }

        if (relatedTools.Any())
        {
            prompt += "\nRELATED TOOLS FROM DATABASE (for workflow completion):\n";
            foreach (var tool in relatedTools)
            {
                prompt += FormatRelatedToolDetails(tool);
            }
        }

        prompt += $@"

User query: {userQuery}

STRICT: ONLY use tool names from: [{toolNamesList}]
Do NOT invent or suggest tools outside the database.

Please provide your recommendation including:
- A recommended tool and why (reference actual features)
- Strengths and limitations
- A workflow with complementary tools (from the list above)
- Practical tips for the user";

        return prompt;
    }

    /// <summary>
    /// C5: Full architecture. All three mechanisms (our production approach).
    /// Replicates OpenAIService.BuildEnhancedPrompt() exactly.
    /// </summary>
    private string BuildC5Full(
        Domain domain, ToolInfo primaryTool, List<ToolInfo> rankedTools,
        List<RelatedToolInfo> relatedTools, string userQuery)
    {
        var expertDesc = BuildExpertDescription(domain);
        var allToolNames = rankedTools.Select(t => t.Name)
            .Concat(relatedTools.Select(t => t.Name))
            .Distinct().ToList();
        var toolNamesList = string.Join(", ", allToolNames);

        var prompt = $@"You are an expert in {expertDesc}.

CRITICAL INSTRUCTION - VERIFIED TOOL DATABASE:
You MUST ONLY reference tools from our verified database listed below. DO NOT mention, suggest, or recommend ANY tools that are not explicitly listed here. This is a curated database of tested and verified tools.

AVAILABLE TOOLS IN DATABASE: [{toolNamesList}]

Any tool name you mention in your response MUST be from this list. Do not invent or suggest tools outside this database.

═══════════════════════════════════════════════════════════════════════════════
PRIMARY TOOLS (ranked from best to worst match based on user criteria):
═══════════════════════════════════════════════════════════════════════════════
";

        int rank = 1;
        foreach (var tool in rankedTools)
        {
            prompt += $@"
[{rank}] {tool.Name} (Score: {1.0 - (rank - 1) * 0.1:F3})
    Type: {tool.TypeOfTool}
    Category: {tool.Category} > {tool.Subcategory}
    Functions: {tool.Functions}
    Abstraction Level: {tool.AbstractionLevel}
    I/O Connections: {tool.IOConnections}
    Verification: {tool.VerificationTool}
    Features: {string.Join(", ", tool.Features)}
";
            rank++;
        }

        if (relatedTools.Any())
        {
            prompt += @"
═══════════════════════════════════════════════════════════════════════════════
RELATED TOOLS FROM DATABASE (for workflow completion):
═══════════════════════════════════════════════════════════════════════════════
";
            foreach (var tool in relatedTools)
            {
                prompt += $@"
• {tool.Name} [{tool.RelationshipType}]
    Type: {tool.TypeOfTool}
    Category: {tool.Category} > {tool.Subcategory}
    Functions: {tool.Functions}
";
            }
        }

        prompt += $@"

═══════════════════════════════════════════════════════════════════════════════
USER QUERY:
═══════════════════════════════════════════════════════════════════════════════

{userQuery}

═══════════════════════════════════════════════════════════════════════════════
ANALYSIS REQUIREMENTS:
═══════════════════════════════════════════════════════════════════════════════

As an expert engineer, provide detailed and actionable analysis based ONLY on the tools listed above.

Respond in this JSON format:

{{
  ""recommendation"": {{
    ""toolName"": ""MUST be from: [{toolNamesList}]"",
    ""verdict"": ""2-3 sentence expert verdict explaining why this tool is the best choice based on the Features and Functions listed above"",
    ""strengths"": [
      ""Specific strength based on the tool's listed Features"",
      ""Specific strength based on the tool's Functions"",
      ""Specific strength based on the tool's I/O Connections or Abstraction Level"",
      ""Specific strength based on the tool's Type or Category fit""
    ],
    ""limitations"": [
      ""Realistic limitation to be aware of"",
      ""Realistic limitation to be aware of""
    ],
    ""bestFor"": ""Describe the ideal use case based on the tool's Features and Functions""
  }},
  ""comparison"": {{
    ""summary"": ""Compare the PRIMARY TOOLS listed above - explain score differences and when to choose each"",
    ""toolNotes"": [
      {{""tool"": ""Tool from PRIMARY TOOLS list"", ""note"": ""Key differentiator based on its Features""}}
    ]
  }},
  ""workflow"": {{
    ""title"": ""Recommended Design Workflow"",
    ""description"": ""Overview using ONLY tools from the database lists above"",
    ""steps"": [
      {{""phase"": ""Phase name"", ""tool"": ""MUST be from database"", ""action"": ""Detailed description referencing actual tool functions""}},
      {{""phase"": ""Phase name"", ""tool"": ""MUST be from database"", ""action"": ""Detailed description referencing actual tool functions""}},
      {{""phase"": ""Phase name"", ""tool"": ""MUST be from database"", ""action"": ""Detailed description referencing actual tool functions""}}
    ]
  }},
  ""tips"": [
    ""Practical tip referencing specific Features from the tools above"",
    ""Practical tip about using the Functions effectively"",
    ""Practical tip about I/O Connections or integration"",
    ""Practical tip about verification or workflow optimization""
  ],
  ""integration"": {{
    ""fileFormats"": ""Based on I/O Connections listed in the tools above"",
    ""compatibility"": ""Based on the tools' actual Features and Functions"",
    ""dataExchange"": ""Tips for data exchange between the specific tools listed above""
  }}
}}

STRICT REQUIREMENTS:
1. ONLY use tool names from: [{toolNamesList}]
2. Reference actual Features, Functions, and I/O Connections from the data provided
3. Do NOT invent or suggest tools outside the database
4. Include at least 4 strengths and 4 tips
5. Make workflow steps reference actual tool capabilities
6. Return ONLY valid JSON - no markdown, no code blocks, no extra text";

        return prompt;
    }

    private string FormatToolDetails(ToolInfo tool)
    {
        return $@"
  {tool.Name}
    Type: {tool.TypeOfTool}
    Category: {tool.Category} > {tool.Subcategory}
    Functions: {tool.Functions}
    Abstraction Level: {tool.AbstractionLevel}
    I/O Connections: {tool.IOConnections}
    Verification: {tool.VerificationTool}
    Features: {string.Join(", ", tool.Features)}
";
    }

    private string FormatRelatedToolDetails(RelatedToolInfo tool)
    {
        return $@"  • {tool.Name} [{tool.RelationshipType}]
    Type: {tool.TypeOfTool}
    Category: {tool.Category} > {tool.Subcategory}
    Functions: {tool.Functions}
";
    }

    private string BuildExpertDescription(Domain domain)
    {
        var categoryExpertDomains = new Dictionary<string, string>
        {
            { "Printed circuit board design (PCB)", "PCB design" },
            { "RF and microwave design (RF/MW)", "RF and microwave design" },
            { "Antenna design (Ant)", "antenna design and simulation" },
            { "Audio design (Aud)", "audio circuit design" },
            { "Analog design (A)", "analog circuit design" },
            { "Interface circuit design of ADC,DAC, etc. (Int)", "ADC/DAC interface design" },
            { "Digital design (D)", "digital circuit design" },
            { "Power Supply Design (PS)", "power supply design" },
            { "Elements", "electronic components and elements" },
            { "Electric installation design (EI)", "electrical installation design" },
            { "Nanotechnology design (Nano)", "nanotechnology design" },
            { "Outcome to prototype: FPGA/CPLDs, USRPs, Arduino", "prototyping and implementation" }
        };

        var subcategoryExpertDomains = new Dictionary<string, string>
        {
            { "PCB Design Tool", "PCB layout and routing tools" },
            { "PCB Design Calculator", "PCB design calculations and analysis" },
            { "Filters", "analog filter design" },
            { "Active Filters", "active filter design and analysis" },
            { "Resistors", "resistor selection and calculation" },
            { "Capacitors", "capacitor selection and analysis" },
            { "Inductors", "inductor design and selection" },
            { "Transformers", "transformer design and selection" },
            { "Crystals oscillators", "crystal oscillator design" },
            { "Heat-sinks", "thermal management and heat-sink design" },
            { "Interface ADC/DAC", "data converter interface design" },
            { "Digital", "digital logic and circuit design" },
            { "RF and microwave", "RF and microwave circuit design" },
            { "Antenna", "antenna design and modeling" },
            { "Audio", "audio circuit and system design" },
            { "Switched-mode power supplies (SMPS), converters, regulators", "SMPS and power converter design" },
            { "Arduino controller", "Arduino-based prototyping" },
            { "USRPs", "software-defined radio platforms" },
            { "FPGA/CPLDs", "FPGA and CPLD development" },
            { "PCBs", "PCB prototyping and manufacturing" },
            { "Electric installation", "electrical installation planning" },
            { "Nano Technology", "nanotechnology applications" }
        };

        if (subcategoryExpertDomains.TryGetValue(domain.SubcategoryName, out var subcatExpert))
            return subcatExpert;
        if (categoryExpertDomains.TryGetValue(domain.CategoryName, out var catExpert))
            return catExpert;

        return "electronic design tools and CAD software";
    }
}
