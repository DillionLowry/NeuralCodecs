namespace NeuralCodecs.Core.Loading.Repository
{
    /// <summary>
    /// Defines operations for interacting with a model repository.
    /// </summary>
    public interface IModelRepository
    {
        /// <summary>
        /// Gets the path to a model file within the repository.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="revision">The revision/version of the model.</param>
        /// <returns>The path to the model file.</returns>
        Task<string> GetModelPath(string modelId, string revision);

        /// <summary>
        /// Retrieves metadata information about a model.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <returns>The model's metadata.</returns>
        Task<ModelMetadata> GetModelInfo(string modelId, string revision);

        /// <summary>
        /// Downloads a model from the repository.
        /// </summary>
        /// <param name="modelId">The identifier of the model to download.</param>
        /// <param name="targetPath">The local path where the model should be saved.</param>
        /// <param name="progress">A progress reporter for download status.</param>
        Task DownloadModel(string modelId, string targetPath, IProgress<double> progress, ModelLoadOptions options);
    }
}