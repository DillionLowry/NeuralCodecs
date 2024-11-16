using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Models;
using System.Diagnostics;
using System.Text.Json;

namespace NeuralCodecs.Core.Interfaces
{
    /// <summary>
    /// Base class for backend-specific model loaders
    /// </summary>
    public interface IModelLoader
    {
        /// <summary>
        /// Loads a model from a local path or remote source
        /// </summary>
        public async Task<TModel> LoadModel<TModel>(string source, ModelLoadOptions? options = null) where TModel : INeuralCodec
        {
            options ??= new ModelLoadOptions();

            try
            {
                if (IsLocalPath(source))
                {
                    return await LoadLocalModel<TModel>(source, options);
                }
                else
                {
                    return await LoadRemoteModel<TModel>(source, options);
                }
            }
            catch (Exception ex) when (ex is not ModelLoadException)
            {
                throw new ModelLoadException($"Failed to load model from {source}", ex);
            }
        }

        /// <summary>
        /// Creates a new model instance from config
        /// </summary>
        public TModel CreateModel<TModel>(ModelConfig config, Device? device = null) where TModel : INeuralCodec;

        public void ClearCache();

        /// <summary>
        /// Saves a model to the specified path
        /// </summary>
        public void SaveModel<TModel>(TModel model, string path) where TModel : INeuralCodec
        {
            var directory = Path.GetDirectoryName(path)
                ?? throw new ArgumentException("Invalid path", nameof(path));

            Directory.CreateDirectory(directory);

            // Save model weights
            model.Save(path);

            // Save config
            var configPath = Path.ChangeExtension(path, ".json");
            SaveConfig(model.Config, configPath);
        }

        /// <summary>
        /// Gets information about a model without loading it
        /// </summary>
        public async Task<ModelInfo?> GetModelInfo(string source)
        {
            try
            {
                if (IsLocalPath(source))
                {
                    var config = await LoadConfig<ModelConfig>(source);
                    return new ModelInfo
                    {
                        Source = source,
                        Config = config,
                        IsCached = true,
                        LastModified = File.GetLastWriteTimeUtc(source),
                    };
                }

                return await GetRemoteModelInfo(source);
            }
            catch
            {
                return null;
            }
        }

        public string GetDefaultCacheDirectory();
        //{
        //    return Path.Combine(
        //        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        //        ".cache", "neural_codecs", BackendName.ToLowerInvariant());
        //}

        public bool IsLocalPath(string source)
        {
            return source.Contains(Path.DirectorySeparatorChar) ||
                   source.Contains(Path.AltDirectorySeparatorChar) ||
                   File.Exists(source);
        }

        public async Task<TModel> LoadLocalModel<TModel>(string path, ModelLoadOptions options) where TModel : INeuralCodec
        {
            if (!File.Exists(path))
                throw new ModelLoadException($"Model file not found at {path}");

            var configPath = Path.ChangeExtension(path, ".json");
            if (!File.Exists(configPath))
            {
                configPath = Path.Combine(Path.GetDirectoryName(path) ?? "", "config.json");
            }
            if (!File.Exists(configPath) && options.HasConfigFile)
            {
                throw new ModelLoadException($"Config file not found at {configPath}");
            }

            try
            {
                var config = await LoadConfig<ModelConfig>(configPath);
                var model = CreateModel<TModel>(config, options.Device);
                model.LoadWeights(path);

                if (options.ValidateModel && !ValidateModel<TModel>(model))
                    throw new ModelLoadException("Model failed validation after loading");

                return model;
            }
            catch (Exception ex) when (ex is not ModelLoadException)
            {
                throw new ModelLoadException($"Failed to load model from {path}", ex);
            }
        }

        public Task<TModel> LoadRemoteModel<TModel>(string source, ModelLoadOptions options) where TModel : INeuralCodec;

        public Task<ModelInfo?> GetRemoteModelInfo(string source);

        public bool ValidateModel<TModel>(TModel model) where TModel : INeuralCodec
        {
            // Basic validation - override for backend-specific checks
            return true;
        }

        public async Task<T> LoadConfig<T>(string path) where T : ModelConfig
        {
            try
            {
                if (!File.Exists(path))
                    throw new FileNotFoundException($"Config file not found at {path}");
                var json = await File.ReadAllTextAsync(path);
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    ReadCommentHandling = JsonCommentHandling.Skip
                };

                var config = JsonSerializer.Deserialize<T>(json, options)
                    ?? throw new ModelLoadException("Failed to deserialize config");

                //ValidateConfig(config);
                return config;
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex);
                throw new ModelLoadException($"Failed to load config from {path}", ex);
            }
        }

        public void SaveConfig(ModelConfig config, string path)
        {
            try
            {
                ValidateConfig(config);
                var json = JsonSerializer.Serialize(config, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });
                File.WriteAllText(path, json);
            }
            catch (Exception ex)
            {
                throw new ModelLoadException($"Failed to save config to {path}", ex);
            }
        }

        public void ValidateConfig(ModelConfig config)
        {
            if (string.IsNullOrEmpty(config.Architecture))
                throw new ModelConfigException("Missing architecture type");
        }

        public Task<TModel> LoadHuggingFaceModel<TModel>(string repoId, ModelLoadOptions options) where TModel : INeuralCodec;
    }
}