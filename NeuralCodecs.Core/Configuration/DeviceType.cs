namespace NeuralCodecs.Core.Configuration
{
    /// <summary>
    /// Specifies the type of device for model execution.
    /// </summary>
    public enum DeviceType
    {
        /// <summary>
        /// CPU-based execution.
        /// </summary>
        CPU,

        /// <summary>
        /// CUDA-based GPU execution.
        /// </summary>
        CUDA
    }
}