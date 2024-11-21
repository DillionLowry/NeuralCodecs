using NeuralCodecs.Core.Operations;

namespace NeuralCodecs.Core.Exceptions
{
    /// <summary>
    /// Exception thrown when a codec operation fails.
    /// </summary>
    public class CodecException : Exception
    {
        /// <summary>
        /// Gets the name of the codec that caused the exception.
        /// </summary>
        public string CodecName { get; }

        /// <summary>
        /// Gets the operation that was being performed when the exception occurred.
        /// </summary>
        public CodecOperation Operation { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="CodecException"/> class with specified details.
        /// </summary>
        /// <param name="message">The message that describes the error.</param>
        /// <param name="codecName">The name of the codec that caused the exception.</param>
        /// <param name="operation">The operation that was being performed.</param>
        /// <param name="innerException">The exception that is the cause of the current exception.</param>
        public CodecException(string message, string codecName, CodecOperation operation, Exception innerException = null)
            : base(message, innerException)
        {
            CodecName = codecName;
            Operation = operation;
        }
    }
}