namespace NeuralCodecs.Core.Exceptions
{
    /// <summary>
    /// Exception thrown when model loading fails.
    /// </summary>
    public class LoadException : NeuralCodecException
    {
        /// <summary>
        /// Gets the model ID
        /// </summary>
        public string? ModelId { get; }

        /// <summary>
        /// Gets the path to the model
        /// </summary>
        public string? ModelPath { get; }

        /// <summary>
        /// Gets the model revision
        /// </summary>
        public string? Revision { get; }

        /// <summary>
        /// Creates a new load exception
        /// </summary>
        public LoadException(string message) : base(message) { }

        /// <summary>
        /// Creates a new load exception with a message and inner exception
        /// </summary>
        public LoadException(string message, Exception innerException)
            : base(message, innerException) { }

        /// <summary>
        /// Creates a new load exception with context
        /// </summary>
        public LoadException(string message, string modelId,
            string? modelPath = null, string? revision = null)
            : base(message)
        {
            ModelId = modelId;
            ModelPath = modelPath;
            Revision = revision;

            if (modelId != null) WithDiagnostic("ModelId", modelId);
            if (modelPath != null) WithDiagnostic("ModelPath", modelPath);
            if (revision != null) WithDiagnostic("Revision", revision);
        }

        /// <summary>
        /// Creates a new load exception with context and inner exception
        /// </summary>
        public LoadException(string message, Exception innerException,
            string modelId, string? modelPath = null, string? revision = null)
            : base(message, innerException)
        {
            ModelId = modelId;
            ModelPath = modelPath;
            Revision = revision;
            if (modelId != null) WithDiagnostic("ModelId", modelId);
            if (modelPath != null) WithDiagnostic("ModelPath", modelPath);
            if (revision != null) WithDiagnostic("Revision", revision);
        }
    }
}