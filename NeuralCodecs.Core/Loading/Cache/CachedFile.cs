namespace NeuralCodecs.Core.Loading.Cache
{
    /// <summary>
    /// Represents a file in the model cache.
    /// </summary>
    public class CachedFile
    {
        /// <summary>
        /// Gets or sets the relative path of the cached file.
        /// </summary>
        public string Path { get; set; } = "";

        /// <summary>
        /// Gets or sets the hash of the file content.
        /// </summary>
        public string Hash { get; set; } = "";

        /// <summary>
        /// Gets or sets the full path of the cached file.
        /// </summary>
        public string FullName { get; set; }

        /// <summary>
        /// Gets or sets the name of the cached file.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the size of the cached file in bytes.
        /// </summary>
        public int Size { get; set; }
    }
}