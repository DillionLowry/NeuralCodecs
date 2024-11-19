using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Models;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using DeviceType = NeuralCodecs.Core.Models.DeviceType;

namespace NeuralCodecs.Torch;

// TODO: Cleanup and refactor
public class TorchModelLoader : IModelLoader
{
    #region Fields

    private readonly IModelCache _cache;
    private readonly ModelRegistry _registry;
    private readonly IModelRepository _repository;
    private readonly Dictionary<Type, object> _validators = new();
    #endregion Fields
    public TorchModelLoader(
        IModelCache? cache = null,
        IModelRepository? repository = null,
        IModelValidator<IModelConfig>? validator = null)
    {
        _cache = cache ?? new DefaultModelCache();
        _repository = repository ?? new HuggingFaceRepository();
        if (validator != null)
        {
            RegisterValidator(validator);
        }
        _registry = CreateDefaultRegistry();
    }

    public event EventHandler<ModelLoadErrorEventArgs> OnError;

    public event EventHandler<ModelLoadProgressEventArgs> OnProgress;
    public void ClearCache(string? modelId = null)
    {
        try
        {
            _cache.ClearCache(modelId);
        }
        catch (Exception ex)
        {
            OnError?.Invoke(this, new ModelLoadErrorEventArgs(
                modelId ?? "all models",
                new ModelLoadException("Failed to clear cache", ex)));
        }
    }

    public string GetDefaultCacheDirectory()
    {
        return _cache.GetDefaultCacheDirectory();
    }

    public async Task<ModelInfo?> GetModelInfo(string source)
    {
        try
        {
            if (IsLocalPath(source))
            {
                return await GetLocalModelInfo(source);
            }
            else
            {
                return await GetRemoteModelInfo(source);
            }
        }
        catch (Exception ex)
        {
            OnError?.Invoke(this, new ModelLoadErrorEventArgs(source, ex));
            return null;
        }
    }

    public bool IsLocalPath(string source) =>
        source.Contains(Path.DirectorySeparatorChar) ||
        source.Contains(Path.AltDirectorySeparatorChar) ||
        File.Exists(source);

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

        if (IsLocalPath(path))
        {
            return await LoadLocalModel<TModel, TConfig>(path, options);
        }
        else
        {
            return await LoadRemoteModel<TModel, TConfig>(path, options);
        }
    }

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

            await Task.Run(() => model.LoadWeights(path));

            if (_validators.TryGetValue(config.GetType(), out var validatorObj))
            {
                var validator = validatorObj as IModelValidator<TConfig>;
                var validationResult = await validator.ValidateModel(model, config);

                if (!validationResult.IsValid)
                {
                    throw new ModelLoadException(
                        $"Model validation failed: {string.Join(", ", validationResult.Errors)}");
                }
            }

            return model;
        }
        catch (Exception ex)
        {
            OnError?.Invoke(this, new ModelLoadErrorEventArgs(path, ex));
            throw new ModelLoadException($"Failed to load model using custom factory: {path}", ex);
        }
    }

    public void RegisterValidator<TConfig>(IModelValidator<TConfig> validator)
        where TConfig : IModelConfig
    {
        _validators[typeof(TConfig)] = validator;
    }

    private static torch.Device ConvertDevice(Core.Models.Device device)
    {
        if (device == null)
            return torch.CPU;

        return device.Type switch
        {
            DeviceType.CPU => torch.CPU,
            DeviceType.CUDA => cuda.is_available() ?
                torch.CUDA :
                throw new InvalidOperationException("CUDA device requested but not available"),
            _ => throw new ArgumentException($"Unsupported device type: {device.Type}")
        };
    }

    private static ModelRegistry CreateDefaultRegistry()
    {
        var registry = new ModelRegistry();

        registry.RegisterModel<SNAC, SNACConfig>((config) =>
            new SNAC(config, ConvertDevice(config.Device)));

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
        return configPath;
    }

    private async Task<ModelInfo?> GetLocalModelInfo(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        var fileInfo = new FileInfo(path);
        return new ModelInfo
        {
            Source = path,
            IsCached = false, // Local file isn't considered cached
            LastModified = fileInfo.LastWriteTimeUtc,
            Size = fileInfo.Length,
            Backend = "Torch"
        };
    }

    private async Task<ModelInfo?> GetRemoteModelInfo(string source)
    {
        try
        {
            // Get repository metadata from Hugging Face
            var modelInfo = await _repository.GetModelInfo(source);
            var cachedPath = await _cache.GetCachedPath(source, "main");

            return new ModelInfo
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
            OnError?.Invoke(this, new ModelLoadErrorEventArgs(source, ex));
            return null;
        }
    }

    private async Task<TConfig> LoadAndValidateConfig<TConfig>(string path, ModelLoadOptions options) where TConfig : IModelConfig
    {
        var configPath = GetConfigPath(path);
        if (!File.Exists(configPath) && options.RequireConfig)
        {
            throw new ModelLoadException($"Config file not found at {configPath}");
        }

        var config = await LoadConfig<TConfig>(configPath);

        if (_validators.TryGetValue(config.GetType(), out var validatorObj))
        {
            var validator = validatorObj as IModelValidator<IModelConfig>;
            var configResult = validator.ValidateConfig(config); // todo

            if (!configResult.IsValid)
            {
                throw new ModelConfigException(
                    $"Invalid model configuration: {string.Join(", ", configResult.Errors)}");
            }
        }

        return config;
    }

    private static readonly JsonSerializerOptions _jsonSerializerOptions = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        Converters = { new ModelConfigJsonConverter<IModelConfig>() }
    };

    private async Task<TConfig> LoadConfig<TConfig>(string path) where TConfig : IModelConfig
    {
        try
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Config file not found at {path}");

            var json = await File.ReadAllTextAsync(path);
            var config = JsonSerializer.Deserialize<TConfig>(json, _jsonSerializerOptions)
                ?? throw new ModelLoadException("Failed to deserialize config");

            return config;
        }
        catch (Exception ex) when (ex is not ModelLoadException)
        {
            throw new ModelLoadException($"Failed to load config from {path}", ex);
        }
    }
    //private async Task<TConfig> LoadConfig<TConfig>(string path) where TConfig : IModelConfig
    //{
    //    try
    //    {
    //        if (!File.Exists(path))
    //            throw new FileNotFoundException($"Config file not found at {path}");

    //        var json = await File.ReadAllTextAsync(path);
    //        var options = new JsonSerializerOptions
    //        {
    //            PropertyNameCaseInsensitive = true,
    //            ReadCommentHandling = JsonCommentHandling.Skip,
    //            Converters = { new ModelConfigJsonConverter<TConfig>() }
    //        };

    //        var config = JsonSerializer.Deserialize<TConfig>(json, options)
    //            ?? throw new ModelLoadException("Failed to deserialize config");

    //        return config;
    //    }
    //    catch (Exception ex) when (ex is not ModelLoadException)
    //    {
    //        throw new ModelLoadException($"Failed to load config from {path}", ex);
    //    }
    //}

    private async Task<TModel> LoadLocalModel<TModel, TConfig>(
                            string path, ModelLoadOptions options)
        where TModel : class, INeuralCodec
        where TConfig : class, IModelConfig
    {
        if (!File.Exists(path))
            throw new ModelLoadException($"Model file not found at {path}");

        try
        {
            var config = await LoadAndValidateConfig<TConfig>(path, options);
            var model = _registry.CreateModel<TModel, TConfig>(config);

            await LoadWeights(model, path, options.ValidateModel);

            return (TModel)model;
        }
        catch (Exception ex)
        {
            OnError?.Invoke(this, new ModelLoadErrorEventArgs(path, ex));
            throw new ModelLoadException($"Failed to load model from {path}", ex);
        }
    }

    private async Task<TModel> LoadRemoteModel<TModel, TConfig>(
        string source, ModelLoadOptions options)
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
                // path of the model file in the repository
                var repoModelPath = await _repository.GetModelPath(source, options.Revision);

                var tempDir = Path.Combine(Path.GetTempPath(), $"neural_codecs_{Guid.NewGuid()}");
                Directory.CreateDirectory(tempDir);

                try
                {
                    var progress = new Progress<double>(p =>
                        OnProgress?.Invoke(this, new ModelLoadProgressEventArgs(source, p)));

                    await _repository.DownloadModel(source, tempDir, progress);

                    modelPath = await _cache.CacheModel(
                        source,
                        Path.Combine(tempDir, repoModelPath),
                        options.Revision);
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

            return await LoadLocalModel<TModel, TConfig>(modelPath, options);
        }
        catch (Exception ex)
        {
            _cache.ClearCache(source); // Clean up failed download
            OnError?.Invoke(this, new ModelLoadErrorEventArgs(source, ex));
            throw new ModelLoadException($"Failed to load remote model: {source}", ex);
        }
    }

    private async Task LoadWeights(INeuralCodec model, string path, bool validate)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(90));

        await Task.Run(() =>
        {
            model.LoadWeights(path);
            cts.Token.ThrowIfCancellationRequested();
        }, cts.Token);

        if (validate && _validators.TryGetValue(model.Config.GetType(), out var validatorObj))
        {
            var validator = validatorObj as IModelValidator<IModelConfig>;
            var validationResult = await validator.ValidateModel(model, model.Config);

            if (!validationResult.IsValid)
            {
                throw new ModelLoadException(
                    $"Model validation failed: {string.Join(", ", validationResult.Errors)}");
            }
        }
    }
}