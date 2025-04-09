using NeuralCodecs.Core;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Core.Events;
using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Loading.Cache;
using NeuralCodecs.Core.Loading.Repository;
using NeuralCodecs.Core.Validation;
using NeuralCodecs.Torch.Config.DAC;
using NeuralCodecs.Torch.Config.Encodec;
using NeuralCodecs.Torch.Config.SNAC;
using NeuralCodecs.Torch.Models;
using NeuralCodecs.Torch.Modules.Encodec;
using System.Text.Json;

namespace NeuralCodecs.Torch;

/// <summary>
/// Provides functionality to load and manage PyTorch-based neural network models.
/// Handles both local and remote model loading, caching, and validation.
/// </summary>
public class TorchModelLoader : IModelLoader
{
    #region Fields

    private readonly IModelCache _cache;
    private readonly ModelRegistry _registry;
    private readonly Dictionary<Type, object> _validators = new();

    private static readonly JsonSerializerOptions _jsonSerializerOptions = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        Converters = { new ModelConfigJsonConverter<IModelConfig>() }
    };

    #endregion Fields

    /// <summary>
    /// Initializes a new instance of the TorchModelLoader class.
    /// </summary>
    /// <param name="cache">Optional custom model cache implementation.</param>
    /// <param name="repository">Optional custom model repository implementation.</param>
    /// <param name="validator">Optional model validator for configuration validation.</param>
    public TorchModelLoader(
        IModelCache? cache = null,
        IModelRepository? repository = null,
        IModelValidator<IModelConfig>? validator = null)
    {
        _cache = cache ?? new DefaultModelCache();
        if (validator != null)
        {
            RegisterValidator(validator);
        }
        _registry = CreateDefaultRegistry();
    }

    /// <summary>
    /// Event raised when an error occurs during model loading or processing.
    /// </summary>
    public event EventHandler<LoadErrorEventArgs> OnError;

    /// <summary>
    /// Event raised to report progress during model loading operations.
    /// </summary>
    public event EventHandler<LoadProgressEventArgs> OnProgress;

    /// <summary>
    /// Clears the model cache for a specific model or all models.
    /// </summary>
    /// <param name="modelId">Optional model ID to clear specific model cache. If null, clears entire cache.</param>
    public void ClearCache(string? modelId = null)
    {
        try
        {
            _cache.ClearCache(modelId);
        }
        catch (Exception ex)
        {
            OnError?.Invoke(this, new LoadErrorEventArgs(
                modelId ?? "all models",
                new LoadException("Failed to clear cache", ex)));
        }
    }

    /// <summary>
    /// Gets the default directory path used for caching models.
    /// </summary>
    /// <returns>The default cache directory path.</returns>
    public string GetDefaultCacheDirectory()
    {
        return _cache.GetDefaultCacheDirectory();
    }

    /// <summary>
    /// Retrieves information about a model from either a local or remote source.
    /// </summary>
    /// <param name="source">The path or identifier of the model.</param>
    /// <returns>Model information if found; otherwise, null.</returns>
    public async Task<ModelMetadata?> GetModelInfo(string source)
    {
        try
        {
            if (IsLocalPath(source))
            {
                return GetLocalModelInfo(source);
            }
            else
            {
                return await GetRemoteModelInfo(source);
            }
        }
        catch (Exception ex)
        {
            OnError?.Invoke(this, new LoadErrorEventArgs(source, ex));
            return null;
        }
    }

    /// <summary>
    /// Determines if the provided source represents a local file path.
    /// </summary>
    /// <param name="source">The path to check.</param>
    /// <returns>True if the source is a local path; otherwise, false.</returns>
    public bool IsLocalPath(string source)
    {
        // Hugging Face model
        if (source.Count(c => c == '/') == 1 &&
            !source.Contains(':') &&  // No drive letter
            !source.StartsWith('/') && // Not absolute path
            !source.StartsWith('\\')) // Not UNC path
        {
            return false;
        }

        // Check if the source is a valid URI
        if (Uri.TryCreate(source, UriKind.Absolute, out var uriResult) &&
            (uriResult.Scheme == Uri.UriSchemeHttp || uriResult.Scheme == Uri.UriSchemeHttps))
        {
            return false;
        }

        // Check if the path is rooted (indicating a local path)
        return Path.IsPathRooted(source) || File.Exists(source);
    }

    /// <summary>
    /// Loads a model asynchronously from either a local or remote source.
    /// </summary>
    /// <typeparam name="TModel">The type of model to load.</typeparam>
    /// <typeparam name="TConfig">The type of configuration for the model.</typeparam>
    /// <param name="path">The path or identifier of the model to load.</param>
    /// <param name="config">Optional model configuration.</param>
    /// <param name="options">Optional loading options.</param>
    /// <returns>The loaded model instance.</returns>
    public async Task<TModel> LoadModelAsync<TModel, TConfig>(
        string path,
        TConfig? config = default,
        ModelLoadOptions? options = null)
        where TModel : class, INeuralCodec
        where TConfig : class, IModelConfig
    {
        options ??= new ModelLoadOptions
        {
            Device = config?.Device,
            ValidateModel = config is null
        };
        config ??= await LoadAndValidateConfig<TConfig>(GetConfigPath(path));

        if (IsLocalPath(path))
        {
            return await LoadLocalModel<TModel, TConfig>(path, config, options);
        }
        else
        {
            return await LoadRemoteModel<TModel, TConfig>(path, config, options);
        }
    }

    /// <summary>
    /// Loads a model using a custom factory method and configuration.
    /// </summary>
    /// <typeparam name="TModel">The type of model to load.</typeparam>
    /// <typeparam name="TConfig">The type of configuration for the model.</typeparam>
    /// <param name="path">The path or identifier of the model to load.</param>
    /// <param name="modelFactory">Factory function to create the model instance.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="options">Optional loading options.</param>
    /// <returns>The loaded model instance.</returns>
    public async Task<TModel> LoadModelAsync<TModel, TConfig>(
        string path,
        Func<IModelConfig, TModel> modelFactory,
        TConfig config,
        ModelLoadOptions? options = null)
        where TModel : class, INeuralCodec
        where TConfig : class, IModelConfig
    {
        try
        {
            var model = modelFactory(config);

            await LoadWeights(model, path, options?.ValidateModel ?? false);

            if (_validators.TryGetValue(config.GetType(), out var validatorObj))
            {
                var validator = validatorObj as IModelValidator<IModelConfig>
                    ?? throw new ConfigurationException("Invalid model validator");

                var validationResult = await validator.ValidateModel(model, config);

                if (!validationResult.IsValid)
                {
                    throw new LoadException(
                        $"Model validation failed: {string.Join(", ", validationResult.Errors)}");
                }
            }

            return model;
        }
        catch (Exception ex)
        {
            OnError?.Invoke(this, new LoadErrorEventArgs(path, ex));
            throw new LoadException($"Failed to load model using custom factory: {path}", ex);
        }
    }

    /// <summary>
    /// Registers a validator for a specific model configuration type.
    /// </summary>
    /// <typeparam name="TConfig">The type of configuration to validate.</typeparam>
    /// <param name="validator">The validator instance.</param>
    public void RegisterValidator<TConfig>(IModelValidator<TConfig> validator)
        where TConfig : IModelConfig
    {
        _validators[typeof(TConfig)] = validator;
    }

    private static ModelRegistry CreateDefaultRegistry()
    {
        var registry = new ModelRegistry();

        registry.RegisterModel<SNAC, SNACConfig>((config) => new SNAC(config));
        registry.RegisterModel<DAC, DACConfig>((config) => new DAC(config));
        registry.RegisterModel<Encodec, EncodecConfig>((config) => new Encodec(config));
        registry.RegisterModel<EncodecLanguageModel, EncodecLanguageModelConfig>((config) => new EncodecLanguageModel(config));

        return registry;
    }

    private string GetConfigPath(string modelPath)
    {
        var configPath = Path.ChangeExtension(modelPath, ".json");
        if (!File.Exists(configPath))
        {
            configPath = Path.Combine(
                Path.GetDirectoryName(modelPath) ?? "",
                "config.json");
        }
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"Config file not found at {configPath}");
        }
        return configPath;
    }

    private ModelMetadata? GetLocalModelInfo(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        var fileInfo = new FileInfo(path);
        return new ModelMetadata
        {
            Source = path,
            IsCached = false, // Local file isn't considered cached
            LastModified = fileInfo.LastWriteTimeUtc,
            Size = fileInfo.Length,
            Backend = "Torch"
        };
    }

    private async Task<ModelMetadata?> GetRemoteModelInfo(string source, string revision = "main")
    {
        try
        {
            var repository = GetRepositoryForSource(source);
            var modelInfo = await repository.GetModelInfo(source, revision);
            var cachedPath = await _cache.GetCachedPath(source, revision);

            return new ModelMetadata
            {
                Source = source,
                IsCached = cachedPath != null,
                LastModified = modelInfo.LastModified,
                Author = modelInfo.Author,
                Tags = modelInfo.Tags,
                Backend = "Torch",
                Size = modelInfo.Size
            };
        }
        catch (Exception ex)
        {
            OnError?.Invoke(this, new LoadErrorEventArgs(source, ex));
            return null;
        }
    }

    private async Task<TConfig> LoadAndValidateConfig<TConfig>(string path)
        where TConfig : IModelConfig
    {
        if (!File.Exists(path))
        {
            throw new LoadException($"Config file not found at {path}");
        }

        var config = await LoadConfig<TConfig>(path);

        if (_validators.TryGetValue(config.GetType(), out var validatorObj))
        {
            var validator = validatorObj as IModelValidator<IModelConfig>
                ?? throw new ConfigurationException("Invalid model validator");

            var configResult = validator.ValidateConfig(config);

            if (!configResult.IsValid)
            {
                throw new ConfigurationException(
                    $"Invalid model configuration: {string.Join(", ", configResult.Errors)}");
            }
        }

        return config;
    }

    private async Task<TConfig> LoadConfig<TConfig>(string path) where TConfig : IModelConfig
    {
        try
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Config file not found at {path}");

            var json = await File.ReadAllTextAsync(path);
            var config = JsonSerializer.Deserialize<TConfig>(json, _jsonSerializerOptions)
                ?? throw new LoadException("Failed to deserialize config");

            return config;
        }
        catch (Exception ex) when (ex is not LoadException)
        {
            throw new LoadException($"Failed to load config from {path}", ex);
        }
    }

    private async Task<TModel> LoadLocalModel<TModel, TConfig>(
        string path, TConfig config, ModelLoadOptions options)
        where TModel : class, INeuralCodec
        where TConfig : class, IModelConfig
    {
        if (!File.Exists(path))
        {
            if (path.Contains(".cache"))
            {
                _cache.ClearCache();
                throw new CacheException($"Model file not found at {path}. Clearing Cache.");
            }
            throw new LoadException($"Model file not found at {path}");
        }

        try
        {
            var model = _registry.CreateModel<TModel, TConfig>(config);

            await LoadWeights(model, path, options.ValidateModel);

            return model;
        }
        catch (Exception ex) when (ex is not (LoadException or CacheException))
        {
            OnError?.Invoke(this, new LoadErrorEventArgs(path, ex));
            throw new LoadException($"Failed to load local model: {path}. {ex.Message}", ex);
        }
    }

    private async Task<TModel> LoadRemoteModel<TModel, TConfig>(
        string source, TConfig config, ModelLoadOptions options)
        where TModel : class, INeuralCodec
        where TConfig : class, IModelConfig
    {
        try
        {
            var modelPath = !options.ForceReload
                ? await _cache.GetCachedPath(source, options.Revision)
                : null;

            if (modelPath == null)
            {
                // Get the appropriate repository for this source
                var repository = GetRepositoryForSource(source);
                
                // If it's GitHub, set the revision from config
                if (repository is GitHubRepository && !string.IsNullOrEmpty(config.Version))
                {
                    options.Revision = config.Version;
                }
                
                // Path of the model file in the repository
                var modelMetadata = await repository.GetModelInfo(source, options.Revision);

                var tempDir = Path.Combine(Path.GetTempPath(), $"neural_codecs_{Guid.NewGuid()}");
                Directory.CreateDirectory(tempDir);

                try
                {
                    var progress = new Progress<double>(p =>
                        OnProgress?.Invoke(this, new LoadProgressEventArgs(source, p)));

                    await repository.DownloadModel(source, tempDir, progress, options);

                    modelPath = await _cache.CacheModel(
                        modelMetadata.Source,
                        tempDir,
                        options.Revision,
                        modelMetadata.FileName,
                        modelMetadata.ConfigFileName
                        );
                }
                finally
                {
                    if (Directory.Exists(tempDir))
                    {
                        try
                        {
                            Directory.Delete(tempDir, recursive: true);
                        }
                        catch { }
                    }
                }
            }

            return await LoadLocalModel<TModel, TConfig>(modelPath, config, options);
        }
        catch (Exception ex) when (ex is not LoadException)
        {
            OnError?.Invoke(this, new LoadErrorEventArgs(source, ex));
            _cache.ClearCache(source); // Clean up failed download
            throw new LoadException($"Failed to load remote model: {source}. {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Gets the appropriate model repository for the given source.
    /// </summary>
    /// <param name="source">The model source URL or identifier.</param>
    /// <returns>An IModelRepository implementation suitable for the source.</returns>
    private IModelRepository GetRepositoryForSource(string source)
    {
        if (Uri.TryCreate(source, UriKind.Absolute, out var uri))
        {
            // Direct URL repositories by domain
            if (uri.Host.Equals("github.com", StringComparison.OrdinalIgnoreCase))
            {
                return new GitHubRepository();
            }
            
            // For dl.fbaipublicfiles.com or any other direct URL
            var directUrlRepo = new DirectUrlRepository();
            if (directUrlRepo.CanHandleUrl(source))
            {
                return directUrlRepo;
            }
        }
        
        // Handle Hugging Face model ID format (owner/repo)
        if (source.Count(c => c == '/') == 1 && !source.Contains(':'))
        {
            return new HuggingFaceRepository();
        }

        throw new InvalidDataException($"Unsupported model source: {source}");
    }

    private async Task LoadWeights(INeuralCodec model, string path, bool validate)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(90));

        await Task.Run(() =>
        {
            model.LoadWeights(path);
            cts.Token.ThrowIfCancellationRequested();
        }, cts.Token);

        if (validate && _validators.TryGetValue(model.Config.GetType(), out var validatorObj) 
            && validatorObj is IModelValidator<IModelConfig> validator)
        {
            var validationResult = await validator.ValidateModel(model, model.Config);
            if (!validationResult.IsValid)
            {
                throw new LoadException($"Model validation failed: {string.Join(", ", validationResult.Errors)}");
            }
        }
    }

    /// <summary>
    /// Loads a model asynchronously without requiring a configuration.
    /// The model will be created with its built-in class defaults.
    /// </summary>
    /// <typeparam name="TModel">The type of model to load.</typeparam>
    /// <param name="path">The path or identifier of the model to load.</param>
    /// <param name="options">Optional loading options.</param>
    /// <returns>The loaded model instance.</returns>
    public async Task<TModel> LoadModelAsync<TModel>(string path, ModelLoadOptions? options = null)
        where TModel : class, INeuralCodec
    {
        options ??= new ModelLoadOptions { ValidateModel = false };
        
        // Get the model path (handle remote models)
        string modelPath = path;
        if (!IsLocalPath(path))
        {
            modelPath = await GetRemoteModelPath(path, options);
        }
        
        // Verify the model file exists
        if (!File.Exists(modelPath))
        {
            if (modelPath.Contains(".cache"))
            {
                _cache.ClearCache();
                throw new CacheException($"Model file not found at {modelPath}. Clearing Cache.");
            }
            throw new LoadException($"Model file not found at {modelPath}");
        }

        try
        {
            // Instantiate the model using its default constructor
            var model = Activator.CreateInstance<TModel>();
            
            // Load the weights
            await LoadWeights(model, modelPath, options.ValidateModel);
            
            return model;
        }
        catch (MissingMethodException)
        {
            throw new LoadException($"Model type {typeof(TModel).Name} must have a parameterless constructor to load without configuration");
        }
        catch (Exception ex) when (ex is not (LoadException or CacheException))
        {
            OnError?.Invoke(this, new LoadErrorEventArgs(path, ex));
            throw new LoadException($"Failed to load model: {path}. {ex.Message}", ex);
        }
    }

    ///// <summary>
    ///// Loads a model asynchronously by detecting its type from the file structure.
    ///// No configuration is required as the model will use its built-in defaults.
    ///// </summary>
    ///// <param name="path">The path or identifier of the model to load.</param>
    ///// <param name="options">Optional loading options.</param>
    ///// <returns>The loaded neural codec model instance.</returns>
    //public async Task<INeuralCodec> LoadModelWithoutConfigAsync(string path, ModelLoadOptions? options = null)
    //{
    //    options ??= new ModelLoadOptions 
    //    { 
    //        ValidateModel = false,
    //        RequireConfig = false
    //    };
        
    //    // Get the model path (handle remote models)
    //    string modelPath = path;
    //    if (!IsLocalPath(path))
    //    {
    //        modelPath = await GetRemoteModelPath(path, options);
    //    }
        
    //    // Create model instance based on file inspection
    //    var modelInstance = await ModelInstantiator.CreateModelInstanceFromFile(modelPath);
        
    //    // Load the weights
    //    await LoadWeights(modelInstance, modelPath, options.ValidateModel);
        
    //    return modelInstance;
    //}

    /// <summary>
    /// Downloads a remote model and returns the local path
    /// </summary>
    private async Task<string> GetRemoteModelPath(string source, ModelLoadOptions options)
    {
        var modelPath = !options.ForceReload
            ? await _cache.GetCachedPath(source, options.Revision)
            : null;

        if (modelPath == null)
        {
            // Get the appropriate repository for this source
            var repository = GetRepositoryForSource(source);
            
            // Path of the model file in the repository
            var modelMetadata = await repository.GetModelInfo(source, options.Revision);

            var tempDir = Path.Combine(Path.GetTempPath(), $"neural_codecs_{Guid.NewGuid()}");
            Directory.CreateDirectory(tempDir);

            try
            {
                var progress = new Progress<double>(p =>
                    OnProgress?.Invoke(this, new LoadProgressEventArgs(source, p)));

                await repository.DownloadModel(source, tempDir, progress, options);

                // If we don't require config files, we can skip passing the config filename
                string configFileName = options.RequireConfig ? modelMetadata.ConfigFileName : "";
                
                modelPath = await _cache.CacheModel(
                    modelMetadata.Source,
                    tempDir,
                    options.Revision,
                    modelMetadata.FileName,
                    configFileName
                    );
            }
            finally
            {
                if (Directory.Exists(tempDir))
                {
                    try
                    {
                        Directory.Delete(tempDir, recursive: true);
                    }
                    catch { }
                }
            }
        }

        return modelPath;
    }


}