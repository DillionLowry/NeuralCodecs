namespace NeuralCodecs.Core.Models;

/// <summary>
/// Represents a file in a Hugging Face repository
/// </summary>
public class RepoFile
{
    public string Path { get; set; } = "";
    public string Type { get; set; } = "";
    public long Size { get; set; }
    public string? Sha { get; set; }
    public DateTime LastModified { get; set; }
}