using NeuralCodecs.Core.Configuration;

namespace NeuralCodecs.Core.Loading
{
    /// <summary>
    /// Configuration options for model loading
    /// </summary>
    public record ModelLoadOptions
    {
        /// <summary>
        /// Target device for the model. Defaults to CPU.
        /// </summary>
        public DeviceConfiguration? Device { get; init; }

        /// <summary>
        /// Model revision/tag to load. Defaults to "main".
        /// </summary>
        public string Revision { get; init; } = "main";

        /// <summary>
        /// Whether to validate model state after loading. Default true.
        /// </summary>
        public bool ValidateModel { get; init; } = true;

        /// <summary>
        /// Whether to force redownload even if cached. Default false.
        /// </summary>
        public bool ForceReload { get; init; } = false;

        /// <summary>
        /// Optional authentication token for private models
        /// </summary>
        public string? AuthToken { get; init; }

        // TODO
        public bool HasConfigFile { get; init; }
        public bool RequireConfig { get; init; }
    }
}