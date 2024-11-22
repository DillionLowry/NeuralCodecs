namespace NeuralCodecs.Core.Loading
{
    /// <summary>
    /// Provides caching functionality for ML models and their associated files.
    /// </summary>
    public interface IModelCache
    {
        /// <summary>
        /// Gets the cached path for a specific model and revision.
        /// </summary>
        /// <param name="modelId">The unique identifier of the model.</param>
        /// <param name="revision">The revision or version of the model.</param>
        /// <returns>A task that resolves to the cached path of the model.</returns>
        Task<string> GetCachedPath(string modelId, string revision);

        /// <summary>
        /// Caches a model and its associated files.
        /// </summary>
        /// <param name="modelId">The unique identifier of the model.</param>
        /// <param name="sourcePath">The source path of the model files.</param>
        /// <param name="revision">The revision or version of the model.</param>
        /// <param name="targetFileName">The target filename for the model.</param>
        /// <param name="targetConfigFileName">The target filename for the model's configuration.</param>
        /// <param name="additionalMetadata">Optional additional metadata to store with the model.</param>
        /// <returns>A task that resolves to the cached path of the model.</returns>
        Task<string> CacheModel(string modelId,
            string sourcePath,
            string revision,
            string targetFileName,
            string targetConfigFileName,
            IDictionary<string, string>? additionalMetadata = null);

        /// <summary>
        /// Clears the cache for a specific model or all models if no modelId is provided.
        /// </summary>
        /// <param name="modelId">Optional model ID to clear specific model cache. If null, clears all cached models.</param>
        void ClearCache(string modelId = null);

        /// <summary>
        /// Gets the default directory path used for caching.
        /// </summary>
        /// <returns>The default cache directory path.</returns>
        string GetDefaultCacheDirectory();

        /// <summary>
        /// Gets the current cache directory path.
        /// </summary>
        /// <returns>The current cache directory path.</returns>
        string GetCacheDirectory();
    }
}