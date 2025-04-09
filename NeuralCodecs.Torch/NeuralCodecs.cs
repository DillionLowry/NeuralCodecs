using NeuralCodecs.Core.Loading;
using NeuralCodecs.Torch.Config.DAC;
using NeuralCodecs.Torch.Config.Encodec;
using NeuralCodecs.Torch.Config.SNAC;
using NeuralCodecs.Torch.Models;

namespace NeuralCodecs.Torch
{
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
        /// Asynchronously creates an instance of SNAC using the specified path, configuration, and options.
        /// </summary>
        /// <param name="path">The path to the model file.</param>
        /// <param name="config">The configuration for the SNAC model.</param>
        /// <param name="options">The options for loading the model.</param>
        public static async Task<SNAC> CreateSNACAsync(string path, SNACConfig? config = null, ModelLoadOptions? options = null)
        {
            var loader = new TorchModelLoader();
            var model = await loader.LoadModelAsync<SNAC, SNACConfig>(path, config, options);
            model.eval();
            return model;
        }

        /// <summary>
        /// Asynchronously creates an instance of DAC using the specified path, configuration, and options.
        /// </summary>
        /// <param name="path">The path to the model file.</param>
        /// <param name="config">The configuration for the DAC model.</param>
        /// <param name="options">The options for loading the model.</param>
        /// <returns>A new instance of DAC.</returns>
        public static async Task<DAC> CreateDACAsync(string path, DACConfig? config = null, ModelLoadOptions? options = null)
        {
            var loader = new TorchModelLoader();
            options ??= new ModelLoadOptions() { HasConfigFile = false, ValidateModel = false };
            var model = await loader.LoadModelAsync<DAC, DACConfig>(path, config, options);
            model.eval();
            return model;
        }
        public static async Task<Encodec> CreateEncodecAsync(string path, EncodecConfig? config = null, ModelLoadOptions? options = null)
        {
            var loader = new TorchModelLoader();
            var model = await loader.LoadModelAsync<Encodec, EncodecConfig>(path, config, options);
            model.eval();
            return model;
        }

    }
}