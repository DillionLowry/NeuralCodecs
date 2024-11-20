using NeuralCodecs.Core.Configuration;

namespace NeuralCodecs.Core
{
    /// <summary>
    /// Represents a neural codec model implementation that can be used for audio processing.
    /// </summary>
    public interface INeuralCodec : IDisposable
    {
        /// <summary>
        /// Gets the configuration settings for the neural codec model.
        /// </summary>
        IModelConfig Config { get; }

        /// <summary>
        /// Loads the model weights from the specified file path.
        /// </summary>
        /// <param name="path">The file path containing the model weights.</param>
        void LoadWeights(string path);
    }
}