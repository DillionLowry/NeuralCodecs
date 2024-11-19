using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Models;
// TODO: Cleanup and refactor
namespace NeuralCodecs.Core.Interfaces
{
    /// <summary>
    /// Base interface for backend-specific model loaders
    /// </summary>
    public interface IModelLoader
    {
        /// <summary>
        /// Event raised when a model loading error occurs
        /// </summary>
        event EventHandler<ModelLoadErrorEventArgs> OnError;

        /// <summary>
        /// Event raised to report model loading progress
        /// </summary>
        event EventHandler<ModelLoadProgressEventArgs> OnProgress;


        /// <summary>
        /// Registers a validator for a specific model configuration type
        /// </summary>
        /// <typeparam name="TConfig">Type of model configuration</typeparam>
        /// <param name="validator">Validator implementation</param>
        void RegisterValidator<TConfig>(IModelValidator<TConfig> validator) where TConfig : IModelConfig;

        /// <summary>
        /// Loads a model from a path with optional configuration
        /// </summary>
        /// <typeparam name="TModel">Type of model to load</typeparam>
        /// <typeparam name="TConfig">Type of config to load</typeparam>
        /// <param name="path">Path to model file</param>
        /// <param name="config">Optional model configuration</param>
        /// <returns>Loaded model instance</returns>
        Task<TModel> LoadModelAsync<TModel, TConfig>(string path, TConfig? config = default, ModelLoadOptions? options = null)
            where TModel : class, INeuralCodec
            where TConfig : class, IModelConfig;

        /// <summary>
        /// Loads a model using a custom factory function and configuration
        /// </summary>
        /// <typeparam name="TModel">Type of model to load</typeparam>
        /// <typeparam name="TConfig">Type of config to load</typeparam>
        /// <param name="path">Path to model file</param>
        /// <param name="modelFactory">Factory function to create the model instance</param>
        /// <param name="config">Model configuration</param>
        /// <returns>Loaded model instance</returns>
        Task<TModel> LoadModelAsync<TModel, TConfig>(
            string path,
            Func<IModelConfig, TModel> modelFactory,
            TConfig config,
            ModelLoadOptions? options = null)
            where TModel : class, INeuralCodec
            where TConfig : class, IModelConfig;

        ///// <summary>
        ///// Loads a model from a local file path
        ///// </summary>
        ///// <typeparam name="TModel">Type of model to load</typeparam>
        ///// <param name="path">Path to model file</param>
        ///// <param name="options">Loading options</param>
        ///// <returns>Loaded model instance</returns>
        //    Task<TModel> LoadLocalModel<TModel>(string path, ModelLoadOptions options) where TModel : INeuralCodec;

        ///// <summary>
        ///// Loads a model from a remote source
        ///// </summary>
        ///// <typeparam name="TModel">Type of model to load</typeparam>
        ///// <param name="source">Remote source identifier (e.g., HuggingFace model ID)</param>
        ///// <param name="options">Loading options</param>
        ///// <returns>Loaded model instance</returns>
        //Task<TModel> LoadRemoteModel<TModel>(string source, ModelLoadOptions options) where TModel : INeuralCodec;

        ///// <summary>
        ///// Creates a new model instance from configuration
        ///// </summary>
        ///// <typeparam name="TModel">Type of model to create</typeparam>
        ///// <param name="config">Model configuration</param>
        ///// <param name="device">Target device for the model</param>
        ///// <returns>Created model instance</returns>
        //TModel CreateModel<TModel>(IModelConfig config, Device? device = null) where TModel : INeuralCodec;

        /// <summary>
        /// Gets information about a model without loading it
        /// </summary>
        /// <param name="source">Model source (local path or remote identifier)</param>
        /// <returns>Model information if available, null otherwise</returns>
        Task<ModelInfo?> GetModelInfo(string source);

        /// <summary>
        /// Checks if a source string represents a local file path
        /// </summary>
        /// <param name="source">Source string to check</param>
        /// <returns>True if source is a local path, false otherwise</returns>
        bool IsLocalPath(string source);

        /// <summary>
        /// Gets the default cache directory for this model loader
        /// </summary>
        /// <returns>Path to default cache directory</returns>
        string GetDefaultCacheDirectory();

        /// <summary>
        /// Clears the model cache
        /// </summary>
        /// <param name="modelId">Optional specific model ID to clear, or null to clear all</param>
        void ClearCache(string? modelId = null);
    }
}