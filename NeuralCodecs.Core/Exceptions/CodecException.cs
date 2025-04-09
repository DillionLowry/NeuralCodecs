using NeuralCodecs.Core.Operations;

namespace NeuralCodecs.Core.Exceptions
{
    /// <summary>
    /// Exception thrown when a codec operation fails
    /// </summary>
    public class CodecException : NeuralCodecException
    {
        /// <summary>
        /// Gets the name of the codec
        /// </summary>
        public string CodecName { get; }

        /// <summary>
        /// Gets the operation that failed
        /// </summary>
        public CodecOperation Operation { get; }

        /// <summary>
        /// Gets additional operational context
        /// </summary>
        public string? OperationContext { get; }

        /// <summary>
        /// Creates a new codec exception
        /// </summary>
        public CodecException(string message, string codecName, CodecOperation operation,
            string? operationContext = null, Exception? innerException = null)
            : base(message, innerException)
        {
            CodecName = codecName;
            Operation = operation;
            OperationContext = operationContext;

            WithDiagnostic("CodecName", codecName);
            WithDiagnostic("Operation", operation.ToString());
            if (operationContext != null) WithDiagnostic("OperationContext", operationContext);
        }

    }
}