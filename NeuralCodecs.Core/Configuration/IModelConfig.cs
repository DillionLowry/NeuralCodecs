namespace NeuralCodecs.Core.Configuration
{
    /// <summary>
    /// Defines the configuration interface for neural codec models.
    /// </summary>
    public interface IModelConfig
    {
        /// <summary>
        /// Gets or sets the device configuration for model execution.
        /// </summary>
        DeviceConfiguration Device { get; set; }

        /// <summary>
        /// Gets or sets the sampling rate in Hz for audio processing.
        /// </summary>
        int SampleRate { get; set; }

        /// <summary>
        /// Gets or sets the architecture name of the model.
        /// </summary>
        string Architecture { get; set; }

        /// <summary>
        /// Gets or sets the version identifier of the model.
        /// </summary>
        string Version { get; set; }

        /// <summary>
        /// Gets or sets additional metadata associated with the model.
        /// </summary>
        IDictionary<string, string> Metadata { get; set; }
    }
}