using CadcomOnline.Evaluation.Models;

namespace CadcomOnline.Evaluation.Services;

/// <summary>
/// Generates the 12 test prompts per domain, following the paper's evaluation design (Section 5.1).
/// Each prompt template is parameterized with the domain's actual tool names.
/// </summary>
public class TestPromptGenerator
{
    /// <summary>
    /// Generates 12 test prompts for a domain, using the domain's actual tool names.
    /// </summary>
    public List<string> GeneratePrompts(Domain domain)
    {
        if (domain.Tools.Count == 0)
            throw new InvalidOperationException($"Domain {domain.Id} has no tools loaded.");

        var topTool = domain.Tools[0].Name;
        var secondTool = domain.Tools.Count > 1 ? domain.Tools[1].Name : topTool;

        // Use natural language descriptions instead of raw subcategory names
        var domainDesc = GetDomainDescription(domain);
        var domainNoun = GetDomainNoun(domain);
        var projectType = GetProjectType(domain);
        var specificFeature = GetSpecificFeature(domain);
        var analysisType = GetAnalysisType(domain);
        var workflowDesc = GetWorkflowDescription(domain);

        return new List<string>
        {
            // P01: Workflow completion
            $"I selected {topTool} for {domainDesc}. What complementary tools do I need for a complete design workflow?",

            // P02: Tool comparison
            $"Compare {topTool} with {secondTool} for a {projectType} project.",

            // P03: Complete toolchain
            $"I'm starting a new {domainDesc} project. Recommend a complete toolchain.",

            // P04: Simulation tools
            $"What simulation tools work well with {topTool}?",

            // P05: Design verification (fixed: no "design design")
            $"I need to verify my {domainNoun}. What tools should I use?",

            // P06: Prototyping
            $"Recommend tools for {domainDesc} prototyping after design completion.",

            // P07: Calculators
            $"What calculators complement {topTool} for {domainDesc}?",

            // P08: Alternatives
            $"I need an alternative to {topTool} with better {specificFeature}.",

            // P09: End-to-end workflow (domain-aware phrasing)
            $"Build me a {workflowDesc}.",

            // P10: Open-source focus
            $"What open-source alternatives exist for {domainDesc}?",

            // P11: Student/beginner
            $"I'm a student learning {domainDesc}. What tools should I start with?",

            // P12: Specific analysis (domain-aware to avoid redundancy)
            $"What tools help with {analysisType} analysis in my {domainDesc} workflow?"
        };
    }

    /// <summary>
    /// Maps domain ID to natural-language description for use in prompts.
    /// </summary>
    private string GetDomainDescription(Domain domain) => domain.Id switch
    {
        "D1" => "PCB design",
        "D2" => "PCB trace calculation and impedance analysis",
        "D3" => "SMPS and power converter design",
        "D4" => "transformer design and calculation",
        _ => domain.SubcategoryName.ToLower()
    };

    /// <summary>
    /// Maps domain ID to a noun phrase for P05 (avoids "design design").
    /// </summary>
    private string GetDomainNoun(Domain domain) => domain.Id switch
    {
        "D1" => "PCB layout and routing",
        "D2" => "PCB trace calculations and impedance results",
        "D3" => "SMPS power converter circuit",
        "D4" => "transformer design calculations",
        _ => domain.SubcategoryName.ToLower() + " results"
    };

    /// <summary>
    /// Domain-aware workflow description for P09 (avoids "schematic to manufacturing" for calculators).
    /// </summary>
    private string GetWorkflowDescription(Domain domain) => domain.Id switch
    {
        "D1" => "complete workflow from schematic capture to PCB manufacturing",
        "D2" => "complete PCB analysis workflow covering trace width, impedance, and thermal calculations",
        "D3" => "complete SMPS design workflow from specification to prototype",
        "D4" => "complete transformer design workflow from requirements to component selection",
        _ => $"complete workflow for {domain.SubcategoryName.ToLower()}"
    };

    private string GetProjectType(Domain domain) => domain.Id switch
    {
        "D1" => "4-layer PCB with mixed-signal components",
        "D2" => "high-speed PCB with controlled impedance traces",
        "D3" => "12V to 3.3V buck converter",
        "D4" => "step-down power transformer for 50Hz application",
        _ => "engineering design"
    };

    private string GetSpecificFeature(Domain domain) => domain.Id switch
    {
        "D1" => "autorouting capabilities",
        "D2" => "impedance calculation accuracy",
        "D3" => "efficiency optimization features",
        "D4" => "core loss calculation",
        _ => "analysis capabilities"
    };

    private string GetAnalysisType(Domain domain) => domain.Id switch
    {
        "D1" => "impedance and thermal",
        "D2" => "trace width and thermal",
        "D3" => "power efficiency and thermal",
        "D4" => "core loss and winding",
        _ => "performance"
    };
}
