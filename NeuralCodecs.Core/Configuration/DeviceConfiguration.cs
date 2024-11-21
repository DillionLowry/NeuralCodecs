namespace NeuralCodecs.Core.Configuration
{
    /// <summary>
    /// Represents the configuration for a computation device.
    /// </summary>
    public record DeviceConfiguration
    {
        /// <summary>
        /// Gets the type of the device (CPU or CUDA).
        /// </summary>
        public DeviceType Type { get; init; }

        /// <summary>
        /// Gets the device index for multi-device setups.
        /// </summary>
        public int Index { get; init; }

        /// <summary>
        /// Gets a default CPU device configuration.
        /// </summary>
        public static DeviceConfiguration CPU => new() { Type = DeviceType.CPU, Index = 0 };

        /// <summary>
        /// Creates a CUDA device configuration with the specified device index.
        /// </summary>
        /// <param name="index">The CUDA device index. Defaults to 0.</param>
        /// <returns>A new CUDA device configuration.</returns>
        public static DeviceConfiguration CUDA(int index = 0) => new() { Type = DeviceType.CUDA, Index = index };
    }
}