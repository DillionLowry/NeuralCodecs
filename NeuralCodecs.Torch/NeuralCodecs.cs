using NeuralCodecs.Core.Loading;
using NeuralCodecs.Torch.Config.DAC;
using NeuralCodecs.Torch.Config.Dia;
using NeuralCodecs.Torch.Config.Encodec;
using NeuralCodecs.Torch.Config.SNAC;
using NeuralCodecs.Torch.Models;
using NeuralCodecs.Torch.Modules.Dia;

namespace NeuralCodecs.Torch;

/// <summary>
/// Provides methods for creating and loading neural network models using Torch.
/// </summary>
public static partial class NeuralCodecs
{
    /// <summary>
    /// Creates an instance of TorchModelLoader.
    /// </summary>
    /// <returns>A new instance of TorchModelLoader.</returns>
    public static TorchModelLoader CreateTorchLoader()
    {
        return new TorchModelLoader();
    }

    /// <summary>
    /// Asynchronously creates an instance of the <see cref="SNAC"/> model by loading it from the specified file path.
    /// </summary>
    /// <remarks>The returned <see cref="SNAC"/> model is set to evaluation mode after loading. This method is
    /// designed for scenarios where the model needs to be loaded asynchronously, such as in applications with
    /// non-blocking I/O requirements.</remarks>
    /// <param name="path">The file path to the model to be loaded. This cannot be null or empty.</param>
    /// <param name="config">An optional configuration object of type <see cref="SNACConfig"/> to customize the model's behavior. If null,
    /// default configuration settings will be used.</param>
    /// <param name="options">Optional model loading options of type <see cref="ModelLoadOptions"/> to control how the model is loaded. If
    /// null, default options will be applied.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the loaded <see cref="SNAC"/> model
    /// instance.</returns>
    public static async Task<SNAC> CreateSNACAsync(string path, SNACConfig? config = null, ModelLoadOptions? options = null)
    {
        var loader = new TorchModelLoader();
        var model = await loader.LoadModelAsync<SNAC, SNACConfig>(path, config, options);
        model.eval();
        return model;
    }

    /// <summary>
    /// Asynchronously creates a new instance of the <see cref="DAC"/> model by loading it from the specified file path.
    /// </summary>
    /// <remarks>The method uses a <see cref="TorchModelLoader"/> to load the model and sets it to evaluation
    /// mode after loading.</remarks>
    /// <param name="path">The file path to the model to be loaded. This cannot be null or empty.</param>
    /// <param name="config">An optional configuration object of type <see cref="DACConfig"/> to customize the model loading process. If
    /// null, default configuration settings will be used.</param>
    /// <param name="options">Optional model loading options of type <see cref="ModelLoadOptions"/>. If null, default options will be applied.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the loaded <see cref="DAC"/> model.</returns>
    public static async Task<DAC> CreateDACAsync(string path, DACConfig? config = null, ModelLoadOptions? options = null)
    {
        var loader = new TorchModelLoader();
        options ??= new ModelLoadOptions() { HasConfigFile = false, ValidateModel = false };
        var model = await loader.LoadModelAsync<DAC, DACConfig>(path, config, options);
        model.eval();
        return model;
    }
    /// <summary>
    /// Asynchronously creates an instance of the <see cref="Encodec"/> model using the specified file path and optional
    /// configuration.
    /// </summary>
    /// <param name="path">The file path to the model file. This cannot be null or empty.</param>
    /// <param name="config">An optional <see cref="EncodecConfig"/> object to configure the model. If null, default configuration is used.</param>
    /// <param name="options">Optional <see cref="ModelLoadOptions"/> to customize the model loading process. If null, default options are
    /// applied.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the initialized <see
    /// cref="Encodec"/> model.</returns>
    public static async Task<Encodec> CreateEncodecAsync(string path, EncodecConfig? config = null, ModelLoadOptions? options = null)
    {
        var loader = new TorchModelLoader();
        var model = await loader.LoadModelAsync<Encodec, EncodecConfig>(path, config, options);
        model.eval();
        return model;
    }

    /// <summary>
    /// Asynchronously creates a new instance of the <see cref="Dia"/> model from the specified file path.
    /// </summary>
    /// <remarks>The method uses a <see cref="TorchModelLoader"/> to load the model asynchronously. The model
    /// is returned in evaluation mode, ready for inference.</remarks>
    /// <param name="path">The file path to the model file to be loaded. This cannot be null or empty.</param>
    /// <param name="config">An optional configuration object of type <see cref="DiaConfig"/> to customize the model loading process. If
    /// null, default configuration settings will be used.</param>
    /// <param name="options">Optional model loading options of type <see cref="ModelLoadOptions"/>. If null, default options will be applied.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the loaded <see cref="Dia"/> model.</returns>
    public static async Task<Dia> CreateDiaAsync(string path, DiaConfig? config = null, ModelLoadOptions? options = null)
    {
        var loader = new TorchModelLoader();
        options ??= new ModelLoadOptions() { HasConfigFile = false, ValidateModel = false };
        var model = await loader.LoadModelAsync<Dia, DiaConfig>(path, config, options);
        model.eval();
        return model;
    }
}