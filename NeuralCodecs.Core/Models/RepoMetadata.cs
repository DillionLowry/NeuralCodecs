using NeuralCodecs.Core.Loading;

namespace NeuralCodecs.Core.Models;

/// <summary>
/// Repository metadata from Hugging Face
/// </summary>
public class RepoMetadata
{
    public string Id { get; set; } = "";
    public string ModelId { get; set; } = "";
    public string Author { get; set; } = "";
    public List<string> Tags { get; set; } = new();
    public DateTime LastModified { get; set; }
    public bool Private { get; set; }
    public List<CachedFile> Files { get; set; } = new();
}