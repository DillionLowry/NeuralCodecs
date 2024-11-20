namespace NeuralCodecs.Core.Loading.Repository;

/// <summary>
/// Represents a file within a model repository.
/// </summary>
public class RepositoryFile
{
    /// <summary>
    /// Gets or sets the relative path of the file within the repository.
    /// </summary>
    public string Path { get; set; } = "";

    /// <summary>
    /// Gets or sets the type/format of the file.
    /// </summary>
    public string Type { get; set; } = "";

    /// <summary>
    /// Gets or sets the size of the file in bytes.
    /// </summary>
    public long Size { get; set; }

    /// <summary>
    /// Gets or sets the SHA hash of the file content.
    /// </summary>
    public string? Sha { get; set; }

    /// <summary>
    /// Gets or sets the last modification timestamp of the file.
    /// </summary>
    public DateTime LastModified { get; set; }
}