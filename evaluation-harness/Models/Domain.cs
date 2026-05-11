namespace CadcomOnline.Evaluation.Models;

/// <summary>
/// Represents one of the 4 evaluation domains from the paper.
/// Each domain maps to a specific subcategory in the database.
/// </summary>
public class Domain
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string SubcategoryName { get; set; } = string.Empty;
    public string CategoryName { get; set; } = string.Empty;
    public int SubcategoryId { get; set; }
    public int CategoryId { get; set; }
    public List<ToolInfo> Tools { get; set; } = [];

    /// <summary>
    /// All valid tool names in this domain (the "ground truth" for hallucination detection).
    /// </summary>
    public HashSet<string> ValidToolNames => new(Tools.Select(t => t.Name), StringComparer.OrdinalIgnoreCase);
}

public class ToolInfo
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string TypeOfTool { get; set; } = string.Empty;
    public string Category { get; set; } = string.Empty;
    public string Subcategory { get; set; } = string.Empty;
    public string Functions { get; set; } = string.Empty;
    public string AbstractionLevel { get; set; } = string.Empty;
    public string IOConnections { get; set; } = string.Empty;
    public string VerificationTool { get; set; } = string.Empty;
    public string ApplicationArea { get; set; } = string.Empty;
    public List<string> Features { get; set; } = [];
}

public class RelatedToolInfo
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string TypeOfTool { get; set; } = string.Empty;
    public string Category { get; set; } = string.Empty;
    public string Subcategory { get; set; } = string.Empty;
    public string Functions { get; set; } = string.Empty;
    public string RelationshipType { get; set; } = string.Empty;
}
