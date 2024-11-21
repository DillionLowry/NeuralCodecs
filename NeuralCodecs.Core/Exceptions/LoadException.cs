namespace NeuralCodecs.Core.Exceptions
{
    /// <summary>
    /// Exception thrown when model loading fails.
    /// </summary>
    public class LoadException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LoadException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The message that describes the error.</param>
        public LoadException(string message) : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LoadException"/> class with a specified error message and a reference to the inner exception.
        /// </summary>
        /// <param name="message">The message that describes the error.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public LoadException(string message, Exception inner) : base(message, inner)
        {
        }
    }
}