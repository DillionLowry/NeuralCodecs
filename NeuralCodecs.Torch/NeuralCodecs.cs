using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Models;
using NeuralCodecs.Torch.Loading;
using System.Text.Json;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch
{
    /// <summary>
    /// Primary interface for working with neural audio codecs
    /// </summary>
    public static class NeuralCodecs
    {
        private static readonly CacheManager Cache = new();
        private static readonly TorchModelLoader _modelLoader = new();

        /// <summary>
        /// Loads a SNAC model from a local file or Hugging Face repository
        /// </summary>
        /// <param name="source">Local path or Hugging Face repo ID</param>
        /// <param name="options">Loading options</param>
        /// <returns>Loaded SNAC model</returns>
        public static async Task<TModel> LoadModelAsync<TModel>(string source,
            ModelLoadOptions? options = null) where TModel : INeuralCodec
        {
            return await _modelLoader.LoadLocalModel<TModel>(source, options ?? new ModelLoadOptions());
        }

        /// <summary>
        /// Gets the default model cache location
        /// </summary>
        public static string GetDefaultCacheDirectory()
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "torch", "neural_codecs");
        }

        /// <summary>
        /// Clears the model cache
        /// </summary>
        /// <param name="modelId">Optional specific model to clear</param>
        public static void ClearCache(string? modelId = null)
        {
            Cache.ClearCache(modelId);
        }

        /// <summary>
        /// Creates a new SNAC model with the specified configuration
        /// </summary>
        public static SNAC CreateSNACModel(SNACConfig config)
        {
            config.Validate();

            return new SNAC(
                samplingRate: config.SamplingRate,
                encoderDim: config.EncoderDim,
                encoderRates: config.EncoderRates,
                decoderDim: config.DecoderDim,
                decoderRates: config.DecoderRates,
                attnWindowSize: config.AttnWindowSize,
                codebookSize: config.CodebookSize,
                codebookDim: config.CodebookDim,
                vqStrides: config.VQStrides,
                noise: config.Noise,
                depthwise: config.Depthwise);
        }

        /// <summary>
        /// Saves a model to the specified path
        /// </summary>
        public static void SaveModel(
            Module<Tensor, (Tensor, List<Tensor>)> model,
            string path,
            ModelConfig config)
        {
            var directory = Path.GetDirectoryName(path);
            if (directory == null)
                throw new ArgumentException("Invalid path", nameof(path));

            Directory.CreateDirectory(directory);

            // Save model weights
            model.save(path);

            // Save config
            var configPath = Path.ChangeExtension(path, ".json");
            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            File.WriteAllText(configPath, json);
        }

        /// <summary>
        /// Gets information about a cached model
        /// </summary>
        public static async Task<ModelInfo?> GetModelInfo(string source)
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

        private static async Task<ModelConfig?> LoadConfig(string modelPath)
        {
            var configPath = Path.ChangeExtension(modelPath, ".json");
            if (!File.Exists(configPath))
                return null;

            var json = await File.ReadAllTextAsync(configPath);
            return JsonSerializer.Deserialize<SNACConfig>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
        }

        private static bool IsLocalPath(string source)
        {
            // Check if source looks like a file path
            return source.Contains(Path.DirectorySeparatorChar) ||
                   source.Contains(Path.AltDirectorySeparatorChar) ||
                   File.Exists(source);
        }
    }
}