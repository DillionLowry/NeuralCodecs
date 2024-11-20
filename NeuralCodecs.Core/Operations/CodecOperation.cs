namespace NeuralCodecs.Core.Operations
{
    /// <summary>
    /// Defines the different types of operations that can be performed by a neural codec.
    /// </summary>
    public enum CodecOperation
    {
        /// <summary>
        /// Model initialization operation.
        /// </summary>
        Initialization,

        /// <summary>
        /// Audio encoding operation.
        /// </summary>
        Encoding,

        /// <summary>
        /// Audio decoding operation.
        /// </summary>
        Decoding,

        /// <summary>
        /// Model loading operation.
        /// </summary>
        ModelLoading
    }
}