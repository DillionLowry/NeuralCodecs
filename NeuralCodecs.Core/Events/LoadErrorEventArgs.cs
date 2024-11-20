namespace NeuralCodecs.Core.Events
{
    /// <summary>
    /// Provides data for load error events.
    /// </summary>
    public class LoadErrorEventArgs : EventArgs
    {
        /// <summary>
        /// Gets the source identifier where the error occurred.
        /// </summary>
        public string Source { get; }

        /// <summary>
        /// Gets the error that occurred during loading.
        /// </summary>
        public Exception Error { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="LoadErrorEventArgs"/> class.
        /// </summary>
        /// <param name="source">The source identifier where the error occurred.</param>
        /// <param name="error">The error that occurred during loading.</param>
        public LoadErrorEventArgs(string source, Exception error)
        {
            Source = source;
            Error = error;
        }
    }
}