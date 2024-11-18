using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Models;
using System.Text.Json;

namespace NeuralCodecs.Torch
{
    //public interface IModelFactory
    //{
    //    INeuralCodec CreateModel(ModelConfig config, Device? device = null);
    //}

    //public class SNACModelFactory : IModelFactory
    //{
    //    public INeuralCodec CreateModel(ModelConfig config, Device? device = null)
    //    {
    //        if (config is not SNACConfig snacConfig)
    //            throw new ArgumentException("Invalid config type for SNAC model");

    //        return new SNAC(snacConfig, ConvertDevice(device));
    //    }
    //}

    /// <summary>
    /// Primary interface for working with neural audio codecs
    /// </summary>
    ///
    public class NeuralCodecs
    {
        private readonly CacheManager Cache = new();
        private readonly IModelLoader _modelLoader;
        private ModelLoadOptions _options;

        public NeuralCodecs(IModelLoader modelLoader)
        {
            _modelLoader = modelLoader ?? throw new ArgumentNullException(nameof(modelLoader));
            _options = new ModelLoadOptions();
        }

        /// <summary>
        /// Loads a SNAC model from a local file or Hugging Face repository
        /// </summary>
        /// <param name="source">Local path or Hugging Face repo ID</param>
        /// <param name="options">Loading options</param>
        /// <returns>Loaded SNAC model</returns>
        public async Task<TModel> LoadModelAsync<TModel>(string source,
            ModelLoadOptions? options = null) where TModel : INeuralCodec
        {
            return await _modelLoader.LoadLocalModel<TModel>(source, options ?? new ModelLoadOptions());
        }

        /// <summary>
        /// Gets the default model cache location
        /// </summary>
        public string GetDefaultCacheDirectory()
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "torch", "neural_codecs");
        }

        /// <summary>
        /// Clears the model cache
        /// </summary>
        /// <param name="modelId">Optional specific model to clear</param>
        public void ClearCache(string? modelId = null)
        {
            Cache.ClearCache(modelId);
        }




        public INeuralCodec CreateModel(ModelConfig config, Core.Models.Device? device = null)
        {
            return _modelLoader.CreateModel<INeuralCodec>(config, device);
        }

        public TModel CreateModel<TModel>(ModelConfig config, Core.Models.Device? device = null) where TModel : INeuralCodec
        {
            return _modelLoader.CreateModel<TModel>(config, device);
        }


        /// <summary>
        /// Gets information about a cached model
        /// </summary>
        public async Task<ModelInfo?> GetModelInfo(string source)
        {
            try
            {
                if (IsLocalPath(source))
                {
                    var config = await LoadConfig(source);
                    return new ModelInfo
                    {
                        Source = source,
                        Config = config,
                        IsCached = true,
                        LastModified = File.GetLastWriteTimeUtc(source)
                    };
                }
                else
                {
                    var loader = new HuggingFaceLoader();
                    var metadata = await loader.GetRepositoryMetadata(source);
                    var cachedPath = Cache.GetCachedModel(source);

                    return new ModelInfo
                    {
                        Source = source,
                        Config = cachedPath != null ? await LoadConfig(cachedPath) : null,
                        IsCached = cachedPath != null,
                        LastModified = metadata.LastModified,
                        Author = metadata.Author,
                        Tags = metadata.Tags
                    };
                }
            }
            catch
            {
                return null;
            }
        }
        //todo
        private async Task<ModelConfig?> LoadConfig(string modelPath)
        {
            var configPath = Path.ChangeExtension(modelPath, ".json");
            if (!File.Exists(configPath))
                return null;

            var json = await File.ReadAllTextAsync(configPath);
            return JsonSerializer.Deserialize<ModelConfig>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
        }

        private bool IsLocalPath(string source)
        {
            // Check if source looks like a file path
            return source.Contains(Path.DirectorySeparatorChar) ||
                   source.Contains(Path.AltDirectorySeparatorChar) ||
                   File.Exists(source);
        }
    }
}