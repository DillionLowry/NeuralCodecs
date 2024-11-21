namespace NeuralCodecs.Core.Loading
{
    public interface IModelCache
    {
        Task<string> GetCachedPath(string modelId, string revision);

        Task<string> CacheModel(string modelId,
            string sourcePath,
            string revision,
            string targetFileName,
            string targetConfigFileName,
            IDictionary<string, string>? additionalMetadata = null);

        void ClearCache(string modelId = null);

        string GetDefaultCacheDirectory();

        string GetCacheDirectory();
    }
}