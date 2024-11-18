using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Models;
using System.Diagnostics;
using System.Text.Json;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler.impl.CyclicLR;

namespace NeuralCodecs.Torch.Loading;

public class TorchModelLoader : IModelLoader
{
    #region Fields

    private readonly HuggingFaceLoader _hubLoader;
    internal readonly CacheManager Cache;
    //protected readonly string BackendName;

    #endregion Fields

    #region Constructors

    public TorchModelLoader(string? authToken = null, string? cacheDir = null)
    {
        _hubLoader = new HuggingFaceLoader(authToken);
    }

    #endregion Constructors

    #region Events

    public event EventHandler<ModelLoadErrorEventArgs>? OnError;

    public event EventHandler<ModelLoadProgressEventArgs>? OnProgress;

    public event EventHandler<ModelLoadWarningEventArgs>? OnWarning;

    #endregion Events

    #region Methods
    public async Task<TModel> LoadModelAsync<TModel>(string path, ModelConfig config) where TModel : class, INeuralCodec
    {
        var torchDevice = ConvertDevice(config.Device);
        // Implementation for predefined models (e.g., SNAC)

        if (typeof(TModel) == typeof(SNAC)) {
            var model = new SNAC(config as SNACConfig ?? new SNACConfig(), torchDevice);
            model.LoadWeights(path);
            return model as TModel;
        }
        throw new NotSupportedException($"Unsupported model type: {nameof(TModel)}");

    }

    public async Task<TModel> LoadModelAsync<TModel>(
        string path,
        Func<ModelConfig, TModel> modelFactory,
        ModelConfig config) where TModel : class, INeuralCodec
    {
        var model = modelFactory(config);
        model.LoadWeights(path);
        return model;
    }

//private readonly Dictionary<string, IModelFactory> _modelFactories = new();

//public void RegisterModelFactory(string modelType, IModelFactory factory)
//{
//    _modelFactories[modelType] = factory;
//}

//public TModel CreateModel<TModel>(ModelConfig config, Device? device = null) where TModel : INeuralCodec
//{
//    if (!_modelFactories.TryGetValue(config.Architecture, out var factory))
//        throw new ArgumentException($"No factory registered for model type: {config.Architecture}");

//    return (TModel)factory.CreateModel(config, device);
//}
// Return INeuralCodec or TMODEL?
public TModel CreateModel<TModel>(ModelConfig config, Core.Models.Device? device = null) where TModel : INeuralCodec
    {
        var torchDevice = ConvertDevice(device);
        /* TODO: figure out how I want to do this
         * Interface method? factory? Static class method?
        return new TModel(config, device, torchDevice);
        */

        // PLACEHOLDER
        return (TModel)(new SNAC((SNACConfig)config, torchDevice) as INeuralCodec);
    }

    public async Task<ModelInfo?> GetRemoteModelInfo(string source)
    {
        try
        {
            var metadata = await _hubLoader.GetRepositoryMetadata(source);
            var cachedPath = Cache.GetCachedModel(source);

            ModelConfig? config = null;
            if (cachedPath != null)
            {
                var configPath = Path.ChangeExtension(cachedPath, ".json");
                if (File.Exists(configPath))
                {
                    config = await LoadConfig<ModelConfig>(configPath);
                }
            }

            var lastModified = metadata.LastModified;
            var modelFile = metadata.Files.FirstOrDefault(f =>
                Path.GetExtension(f.FullName) is ".bin" or ".pt" or ".pth");

            var size = modelFile?.Size ?? 0;

            return new ModelInfo
            {
                Source = source,
                Config = config,
                IsCached = cachedPath != null,
                LastModified = lastModified,
                Author = metadata.Author,
                Tags = metadata.Tags,
                //Backend = BackendName,
                Size = size,
            };
        }
        catch (Exception ex)
        {
            OnError?.Invoke(this, new ModelLoadErrorEventArgs(source, ex));
            return null;
        }
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
            var config = await this.LoadConfig<ModelConfig>(configPath);

            var model = CreateModel<TModel>(config, options.Device);

            using (var scope = torch.NewDisposeScope())
            {
                // Load weights with timeout
                using var cts = new CancellationTokenSource(
                    TimeSpan.FromSeconds(90));

                await Task.Run(() =>
                {
                    model.LoadWeights(path);
                    cts.Token.ThrowIfCancellationRequested();
                }, cts.Token);
            }

            if (false && options.ValidateModel) //TODO
            {
                config.Validate();
                await ValidateLoadedModel(model, path);
            }

            return model;
        }
        catch (OperationCanceledException)
        {
            throw new ModelLoadException($"Loading model timed out: {path}");
        }
        catch (Exception ex) when (ex is not ModelLoadException)
        {
            throw new ModelLoadException($"Failed to load model from {path}", ex);
        }
    }

    public async Task<TModel> LoadRemoteModel<TModel>(string source, ModelLoadOptions options) where TModel : INeuralCodec
    {
        var modelPath = !options.ForceReload
            ? Cache.GetCachedModel(source, options.Revision)
            : null;

        if (modelPath == null)
        {
            try
            {
                // Download model files
                var progress = new Progress<double>(p =>
                    OnProgress?.Invoke(this, new ModelLoadProgressEventArgs(source, p)));

                var files = await _hubLoader.DownloadSnapshot(
                    source,
                    Cache.GetModelCacheDir(source, options.Revision),
                    allowedPatterns: new[] { "*.bin", "*.pt", "*.pth", "*.json" },
                    progress: progress);

                modelPath = files.FirstOrDefault(f =>
                    Path.GetExtension(f) is ".bin" or ".pt" or ".pth")
                    ?? throw new ModelLoadException("No model file found in download");

                await ValidateDownload(files, modelPath);
            }
            catch (Exception ex)
            {
                Cache.ClearCache(source); // Clean up failed download
                throw new ModelLoadException($"Failed to download model: {source}", ex);
            }
        }

        return await LoadLocalModel<TModel>(modelPath, options);
    }

    protected void ValidateConfig(ModelConfig config)
    {
        switch (config)
        {
            case SNACConfig snacConfig:
                //ValidateSNACConfig(snacConfig); Todo: Implement this
                break;

            default:
                throw new ModelConfigException(
                    $"Unsupported config type for TorchSharp: {config.GetType().Name}");
        }
    }

    private static void LoadPyTorchWeights(SNAC model, string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"PyTorch weights not found at {modelPath}");

        try
        {
            model.load_py(modelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load PyTorch weights from {modelPath}", ex);
        }
    }

    private static void LoadTorchSharpWeights(SNAC model, string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"TorchSharp weights not found at {modelPath}");

        try
        {
            model.load(modelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load TorchSharp weights from {modelPath}", ex);
        }
    }

    private static torch.Device ConvertDevice(Core.Models.Device? device)
    {
        if (device == null)
            return torch.CPU;

        return device.Type.ToLowerInvariant() switch
        {
            "cpu" => torch.CPU,
            "cuda" => torch.CUDA,
            _ => throw new ArgumentException($"Unsupported device type: {device.Type}")
        };
    }

    private static Tensor CreateDummyInput(ModelConfig config, torch.Device device)
    {
        switch (config)
        {
            case SNACConfig snacConfig:
                return torch.randn(1, 1, snacConfig.SamplingRate).to(device);

            default:
                throw new ArgumentException($"Unsupported config type: {config.GetType().Name}");
        }
    }

    private static bool IsLocalPath(string source)
    {
        // Check if source looks like a file path
        return source.Contains(Path.DirectorySeparatorChar) ||
               source.Contains(Path.AltDirectorySeparatorChar) ||
               File.Exists(source);
    }

    private static bool IsPyTorchModelFile(byte[] bytes)
    {
        // Check for PyTorch magic numbers/signatures
        if (bytes.Length < 8) return false;

        // Look for common PyTorch model signatures
        return bytes[0] == 0x80 && bytes[1] == 0x02 || // Pickle protocol
               (bytes[0] == 'P' && bytes[1] == 'K'); // ZIP archive
    }

    public async Task<T> LoadConfig<T>(string path) where T : ModelConfig
    {
        try
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Config file not found at {path}");
            var json = await File.ReadAllTextAsync(path);
            //var options = new JsonSerializerOptions
            //{
            //    PropertyNameCaseInsensitive = true,
            //    ReadCommentHandling = JsonCommentHandling.Skip
            //};

            var config = JsonSerializer.Deserialize<SNACConfig>(json)
                ?? throw new ModelLoadException("Failed to deserialize config");

            //ValidateConfig(config);
            return config as T;
        }
        catch (Exception ex)
        {
            Debug.WriteLine(ex);
            throw new ModelLoadException($"Failed to load config from {path}", ex);
        }
    }

    private async Task ValidateDownload(IEnumerable<string> files, string modelPath)
    {
        // Verify all required files exist
        var configPath = Path.ChangeExtension(modelPath, ".json");
        if (!File.Exists(configPath))
            throw new ModelLoadException("Config file missing from download");

        // Verify model file size
        var modelInfo = new FileInfo(modelPath);
        if (modelInfo.Length < 1000) // Arbitrary minimum size
            throw new ModelLoadException("Model file appears to be invalid (too small)");

        // Load and validate config
        var config = await LoadConfig<ModelConfig>(configPath);
        ValidateConfig(config);

        // Verify model file format
        try
        {
            var modelBytes = await File.ReadAllBytesAsync(modelPath);
            if (!IsPyTorchModelFile(modelBytes))
                throw new ModelLoadException("Invalid PyTorch model file format");
        }
        catch (Exception ex)
        {
            throw new ModelLoadException("Failed to validate model file", ex);
        }
    }

    private async Task ValidateLoadedModel(INeuralCodec model, string path)
    {
        try
        {
            if (!ValidateModel(model))
                throw new ModelLoadException("Model failed basic validation");

            // Test inference with dummy input
            //var dummy = CreateDummyInput(model.Config, model.Device);
            //await Task.Run(() => model.Forward(dummy));
        }
        catch (Exception ex)
        {
            throw new ModelLoadException($"Model validation failed for {path}", ex);
        }
    }

    private bool ValidateModel(INeuralCodec model)
    {
        throw new NotImplementedException();
    }

    public void ClearCache()
    {
        throw new NotImplementedException();
    }

    public string GetDefaultCacheDirectory()
    {
        throw new NotImplementedException();
    }

    public Task<TModel> LoadHuggingFaceModel<TModel>(string repoId, ModelLoadOptions options) where TModel : INeuralCodec
    {
        throw new NotImplementedException();
    }

    public Task<TModel> LoadModel<TModel>(string source, ModelLoadOptions? options = null) where TModel : INeuralCodec
    {
        throw new NotImplementedException();
    }

    public void SaveModel<TModel>(TModel model, string path) where TModel : INeuralCodec
    {
        throw new NotImplementedException();
    }

    public Task<ModelInfo?> GetModelInfo(string source)
    {
        throw new NotImplementedException();
    }

    public void SaveConfig(ModelConfig config, string path)
    {
        throw new NotImplementedException();
    }

    void IModelLoader.ValidateConfig(ModelConfig config)
    {
        throw new NotImplementedException();
    }

    #endregion Methods
}