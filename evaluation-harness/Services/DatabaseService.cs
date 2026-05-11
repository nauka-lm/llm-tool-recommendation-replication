using CadcomOnline.Data;
using CadcomOnline.Evaluation.Models;
using Microsoft.EntityFrameworkCore;
using System.Text.RegularExpressions;

namespace CadcomOnline.Evaluation.Services;

/// <summary>
/// Fetches tool data from the PostgreSQL database for evaluation.
/// Replicates the data loading logic from DecisionMakingController.
/// </summary>
public class DatabaseService
{
    private readonly DataContext _context;

    public DatabaseService(DataContext context)
    {
        _context = context;
    }

    /// <summary>
    /// Loads domain data: tools, features, and related tools for a given subcategory.
    /// </summary>
    public async Task<Domain> LoadDomainAsync(string domainId, string subcategoryName)
    {
        var subcategory = await _context.Subcategory
            .Include(s => s.Category)
                .ThenInclude(c => c.Panel)
            .FirstOrDefaultAsync(s => s.Name == subcategoryName && !s.IsDeleted && s.IsActive);

        if (subcategory == null)
            throw new InvalidOperationException($"Subcategory '{subcategoryName}' not found in database.");

        var domain = new Domain
        {
            Id = domainId,
            Name = subcategoryName,
            SubcategoryName = subcategoryName,
            CategoryName = subcategory.Category?.Name ?? "Unknown",
            SubcategoryId = subcategory.Id,
            CategoryId = subcategory.CategoryId
        };

        // Load all tools in this subcategory with their full details
        var toolPassports = await _context.ToolPassport
            .Include(tp => tp.TypeOfTool)
            .Include(tp => tp.Category)
            .Include(tp => tp.Subcategory)
            .Where(tp => !tp.IsDeleted && tp.IsActive && tp.SubcategoryId == subcategory.Id)
            .ToListAsync();

        foreach (var tp in toolPassports)
        {
            // Load features for this tool
            var features = await _context.ToolFeature
                .Include(tf => tf.Criteria)
                .Include(tf => tf.CriteriaValue)
                .Where(tf => !tf.IsDeleted && tf.IsActive && tf.ToolPassportId == tp.Id)
                .ToListAsync();

            var featureStrings = features
                .Where(f => f.Criteria != null && f.CriteriaValue != null)
                .Select(f => $"{f.Criteria!.Name}: {f.CriteriaValue!.Value}")
                .ToList();

            domain.Tools.Add(new ToolInfo
            {
                Id = tp.Id,
                Name = tp.Name,
                TypeOfTool = tp.TypeOfTool?.Name ?? "Unknown",
                Category = tp.Category?.Name ?? "Unknown",
                Subcategory = tp.Subcategory?.Name ?? "Unknown",
                Functions = RemoveHtmlTags(tp.Function ?? ""),
                AbstractionLevel = RemoveHtmlTags(tp.AbstractionLevel ?? ""),
                IOConnections = RemoveHtmlTags(tp.IOConnection ?? ""),
                VerificationTool = RemoveHtmlTags(tp.VerificationTool ?? ""),
                ApplicationArea = RemoveHtmlTags(tp.ApplicationAreaOrEquivalenceCoverage ?? ""),
                Features = featureStrings
            });
        }

        return domain;
    }

    /// <summary>
    /// Gets the complete list of all 82 tool names in the entire database (for global hallucination detection).
    /// </summary>
    public async Task<HashSet<string>> GetAllToolNamesAsync()
    {
        var names = await _context.ToolPassport
            .Where(tp => !tp.IsDeleted && tp.IsActive)
            .Select(tp => tp.Name)
            .ToListAsync();

        return new HashSet<string>(names, StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Replicates the 6-stage context builder from DecisionMakingController.GetRelatedToolsAsync().
    /// </summary>
    public async Task<List<RelatedToolInfo>> GetRelatedToolsAsync(int primaryToolId)
    {
        var relatedTools = new List<RelatedToolInfo>();

        var primaryTool = await _context.ToolPassport
            .Include(tp => tp.Subcategory)
            .Include(tp => tp.Category)
                .ThenInclude(c => c.Panel)
            .Include(tp => tp.TypeOfTool)
            .FirstOrDefaultAsync(tp => tp.Id == primaryToolId && !tp.IsDeleted && tp.IsActive);

        if (primaryTool == null) return relatedTools;

        var addedToolIds = new HashSet<int> { primaryToolId };
        var primaryPanelId = primaryTool.PanelId ?? primaryTool.Category?.PanelId ?? 1;
        var primarySubcategoryName = primaryTool.Subcategory?.Name ?? "";

        // STAGE 0: Tool Passport Relationships
        await AddToolPassportRelatedTools(primaryTool, relatedTools, addedToolIds);

        // STAGE 1: Same Category, Different Subcategory
        var sameCategoryTools = await _context.ToolPassport
            .Include(tp => tp.TypeOfTool).Include(tp => tp.Category).Include(tp => tp.Subcategory)
            .Where(tp => !tp.IsDeleted && tp.IsActive
                        && tp.CategoryId == primaryTool.CategoryId
                        && tp.SubcategoryId != primaryTool.SubcategoryId)
            .Take(5).ToListAsync();

        foreach (var tool in sameCategoryTools)
        {
            if (addedToolIds.Add(tool.Id))
                relatedTools.Add(CreateRelatedToolInfo(tool, ClassifyToolRelationship(tool, primarySubcategoryName)));
        }

        // STAGE 2: Same Panel, Different Category
        var samePanelTools = await _context.ToolPassport
            .Include(tp => tp.TypeOfTool).Include(tp => tp.Category).Include(tp => tp.Subcategory)
            .Where(tp => !tp.IsDeleted && tp.IsActive
                        && tp.PanelId == primaryPanelId
                        && tp.CategoryId != primaryTool.CategoryId)
            .Take(6).ToListAsync();

        foreach (var tool in samePanelTools)
        {
            if (addedToolIds.Add(tool.Id))
            {
                var panelName = tool.Category?.Panel?.Name ??
                    (primaryPanelId == 1 ? "Structural/Behavioral" : "Physical/Topological");
                relatedTools.Add(CreateRelatedToolInfo(tool, $"Related {panelName} ({tool.Category?.Name})"));
            }
        }

        // STAGE 3: Cross-Panel Tools
        var crossPanelTools = await _context.ToolPassport
            .Include(tp => tp.TypeOfTool).Include(tp => tp.Category).ThenInclude(c => c.Panel)
            .Include(tp => tp.Subcategory)
            .Where(tp => !tp.IsDeleted && tp.IsActive && tp.PanelId != primaryPanelId)
            .Take(8).ToListAsync();

        foreach (var tool in crossPanelTools)
        {
            if (addedToolIds.Add(tool.Id))
                relatedTools.Add(CreateRelatedToolInfo(tool, DetermineWorkflowStage(tool, primaryPanelId)));
        }

        // STAGE 4: Calculator/Analysis Tools
        var calculatorTools = await _context.ToolPassport
            .Include(tp => tp.TypeOfTool).Include(tp => tp.Category).Include(tp => tp.Subcategory)
            .Where(tp => !tp.IsDeleted && tp.IsActive
                        && (tp.Subcategory.Name.ToLower().Contains("calculator")
                            || tp.TypeOfTool.Name.ToLower().Contains("calculator")
                            || tp.Name.ToLower().Contains("calculator")
                            || tp.Name.ToLower().Contains("toolkit")))
            .Take(5).ToListAsync();

        foreach (var tool in calculatorTools)
        {
            if (addedToolIds.Add(tool.Id))
                relatedTools.Add(CreateRelatedToolInfo(tool, $"Calculator/Analysis ({tool.Category?.Name})"));
        }

        // STAGE 5: Simulation Tools
        var simulationTools = await _context.ToolPassport
            .Include(tp => tp.TypeOfTool).Include(tp => tp.Category).Include(tp => tp.Subcategory)
            .Where(tp => !tp.IsDeleted && tp.IsActive
                        && (tp.Subcategory.Name.ToLower().Contains("simulator")
                            || tp.Subcategory.Name.ToLower().Contains("simulation")
                            || tp.Category.Name.ToLower().Contains("simulator")
                            || tp.Name.ToLower().Contains("spice")
                            || tp.Name.ToLower().Contains("simulator")
                            || tp.Name.ToLower().Contains("simulation")))
            .Take(4).ToListAsync();

        foreach (var tool in simulationTools)
        {
            if (addedToolIds.Add(tool.Id))
                relatedTools.Add(CreateRelatedToolInfo(tool, $"Simulation ({tool.Category?.Name})"));
        }

        // STAGE 6: Prototyping Tools
        var prototypeTools = await _context.ToolPassport
            .Include(tp => tp.TypeOfTool).Include(tp => tp.Category).Include(tp => tp.Subcategory)
            .Where(tp => !tp.IsDeleted && tp.IsActive
                        && (tp.Subcategory.Name.ToLower().Contains("prototype")
                            || tp.Subcategory.Name.ToLower().Contains("fpga")
                            || tp.Subcategory.Name.ToLower().Contains("arduino")
                            || tp.Subcategory.Name.ToLower().Contains("usrp")
                            || tp.Category.Name.ToLower().Contains("prototype")
                            || tp.Category.Name.ToLower().Contains("outcome")))
            .Take(3).ToListAsync();

        foreach (var tool in prototypeTools)
        {
            if (addedToolIds.Add(tool.Id))
                relatedTools.Add(CreateRelatedToolInfo(tool, $"Prototyping ({tool.Subcategory?.Name})"));
        }

        return relatedTools;
    }

    private async Task AddToolPassportRelatedTools(
        CadcomOnline.Models.ToolPassport primaryTool,
        List<RelatedToolInfo> relatedTools,
        HashSet<int> addedToolIds)
    {
        var ioConnections = ExtractToolNames(primaryTool.IOConnection);
        var verificationTools = ExtractToolNames(primaryTool.VerificationTool);
        var equivalentTools = ExtractToolNames(primaryTool.ApplicationAreaOrEquivalenceCoverage);

        var allRelatedNames = ioConnections.Concat(verificationTools).Concat(equivalentTools).Distinct().ToList();
        if (!allRelatedNames.Any()) return;

        var matchingTools = await _context.ToolPassport
            .Include(tp => tp.TypeOfTool).Include(tp => tp.Category).Include(tp => tp.Subcategory)
            .Where(tp => !tp.IsDeleted && tp.IsActive)
            .ToListAsync();

        foreach (var tool in matchingTools)
        {
            if (addedToolIds.Contains(tool.Id)) continue;
            var toolNameLower = tool.Name.ToLower();

            foreach (var relatedName in allRelatedNames)
            {
                if (toolNameLower.Contains(relatedName.ToLower()) || relatedName.ToLower().Contains(toolNameLower))
                {
                    if (addedToolIds.Add(tool.Id))
                    {
                        string relationshipType;
                        if (ioConnections.Any(n => toolNameLower.Contains(n.ToLower()) || n.ToLower().Contains(toolNameLower)))
                            relationshipType = "I/O Compatible";
                        else if (verificationTools.Any(n => toolNameLower.Contains(n.ToLower()) || n.ToLower().Contains(toolNameLower)))
                            relationshipType = "Verification Tool";
                        else
                            relationshipType = "Equivalent/Similar";

                        relatedTools.Add(CreateRelatedToolInfo(tool, relationshipType));
                    }
                    break;
                }
            }
        }
    }

    private string ClassifyToolRelationship(CadcomOnline.Models.ToolPassport tool, string primarySubcategoryName)
    {
        var subcategoryName = tool.Subcategory?.Name?.ToLower() ?? "";
        var toolName = tool.Name?.ToLower() ?? "";
        var typeOfTool = tool.TypeOfTool?.Name?.ToLower() ?? "";

        if (subcategoryName.Contains("calculator") || toolName.Contains("calculator") || typeOfTool.Contains("calculator"))
            return $"Calculator ({tool.Category?.Name})";
        if (subcategoryName.Contains("simulator") || subcategoryName.Contains("simulation") ||
            toolName.Contains("spice") || toolName.Contains("simulator"))
            return $"Simulation ({tool.Category?.Name})";
        if (subcategoryName.Contains("verification") || subcategoryName.Contains("drc") ||
            toolName.Contains("verification") || toolName.Contains("checker"))
            return "Verification Tool";

        return $"Complementary ({tool.Subcategory?.Name})";
    }

    private string DetermineWorkflowStage(CadcomOnline.Models.ToolPassport tool, int primaryPanelId)
    {
        var subcategoryName = tool.Subcategory?.Name?.ToLower() ?? "";
        var categoryName = tool.Category?.Name?.ToLower() ?? "";
        var toolName = tool.Name?.ToLower() ?? "";
        var typeOfToolName = tool.TypeOfTool?.Name?.ToLower() ?? "";
        var panelName = tool.Category?.Panel?.Name ?? "";
        var baseStage = primaryPanelId == 1 ? "Physical/Topological" : "Structural";

        if (subcategoryName.Contains("calculator") || typeOfToolName.Contains("calculator") ||
            toolName.Contains("calculator") || toolName.Contains("toolkit"))
            return $"{baseStage} Calculator ({tool.Category?.Name})";
        if (subcategoryName.Contains("simulator") || subcategoryName.Contains("simulation") ||
            toolName.Contains("spice") || toolName.Contains("simulator") || toolName.Contains("simulation"))
            return $"{baseStage} Simulation ({tool.Category?.Name})";
        if (subcategoryName.Contains("prototype") || subcategoryName.Contains("fpga") ||
            subcategoryName.Contains("arduino") || subcategoryName.Contains("usrp") ||
            categoryName.Contains("prototype") || categoryName.Contains("outcome"))
            return $"Prototyping ({tool.Subcategory?.Name})";
        if (categoryName.Contains("element") ||
            subcategoryName.Contains("resistor") || subcategoryName.Contains("capacitor") ||
            subcategoryName.Contains("inductor") || subcategoryName.Contains("transformer") ||
            subcategoryName.Contains("crystal") || subcategoryName.Contains("heat"))
            return $"Component/Sourcing ({tool.Subcategory?.Name})";
        if (subcategoryName.Contains("verification") || subcategoryName.Contains("drc") ||
            toolName.Contains("verification") || toolName.Contains("checker"))
            return "Verification Tool";

        return !string.IsNullOrEmpty(panelName) ? $"{panelName} ({tool.Category?.Name})" : $"{baseStage} ({tool.Category?.Name})";
    }

    private RelatedToolInfo CreateRelatedToolInfo(CadcomOnline.Models.ToolPassport tool, string relationshipType)
    {
        return new RelatedToolInfo
        {
            Id = tool.Id,
            Name = tool.Name,
            TypeOfTool = tool.TypeOfTool?.Name ?? "Unknown",
            Category = tool.Category?.Name ?? "Unknown",
            Subcategory = tool.Subcategory?.Name ?? "Unknown",
            Functions = RemoveHtmlTags(tool.Function ?? ""),
            RelationshipType = relationshipType
        };
    }

    private List<string> ExtractToolNames(string? fieldValue)
    {
        if (string.IsNullOrWhiteSpace(fieldValue)) return [];
        var cleanValue = RemoveHtmlTags(fieldValue);
        return cleanValue
            .Split([',', ';', '\n', '\r'], StringSplitOptions.RemoveEmptyEntries)
            .Select(s => s.Trim())
            .Where(s => !string.IsNullOrWhiteSpace(s) && s.Length > 2)
            .ToList();
    }

    private static string RemoveHtmlTags(string html)
    {
        if (string.IsNullOrEmpty(html)) return html;
        return Regex.Replace(html, "<.*?>", string.Empty);
    }
}
