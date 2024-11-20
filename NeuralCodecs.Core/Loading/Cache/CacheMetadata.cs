namespace NeuralCodecs.Core.Loading.Cache
{
    /// <summary>
    /// Represents metadata for a cached model.
    /// </summary>
    public class CacheMetadata
    {
        /// <summary>
        /// Gets or sets the model identifier.
        /// </summary>
        public string ModelId { get; set; } = "";

        /// <summary>
        /// Gets or sets the revision/version of the model.
        /// </summary>
        public string Revision { get; set; } = "";

        /// <summary>
        /// Gets or sets when the model was cached.
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of days to keep the cached model.
        /// </summary>
        public int MaxAgeInDays { get; set; }

        /// <summary>
        /// Gets or sets the list of files included in the cache.
        /// </summary>
        public List<CachedFile> Files { get; set; } = new();
    }
}