using NeuralCodecs.Core.Loading.Cache;

namespace NeuralCodecs.Core.Loading.Repository;

/// <summary>
/// Represents metadata information for a model repository.
/// </summary>
public class RepositoryMetadata
{
    /// <summary>
    /// Gets or sets the unique identifier for the repository.
    /// </summary>
    public string Id { get; set; } = "";

    /// <summary>
    /// Gets or sets the model identifier associated with this repository.
    /// </summary>
    public string ModelId { get; set; } = "";

    /// <summary>
    /// Gets or sets the author of the model.
    /// </summary>
    public string Author { get; set; } = "";

    /// <summary>
    /// Gets or sets the list of tags associated with the model.
    /// </summary>
    public List<string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets the last modification timestamp of the repository.
    /// </summary>
    public DateTime LastModified { get; set; }

    /// <summary>
    /// Gets or sets whether the repository is private.
    /// </summary>
    public bool Private { get; set; }

    /// <summary>
    /// Gets or sets the list of files contained in the repository.
    /// </summary>
    public List<RepositoryFile> Files { get; set; } = new();
}