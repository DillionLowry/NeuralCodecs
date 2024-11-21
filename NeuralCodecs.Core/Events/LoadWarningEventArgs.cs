namespace NeuralCodecs.Core.Events
{
    /// <summary>
    /// Provides data for load warning events.
    /// </summary>
    public class LoadWarningEventArgs : EventArgs
    {
        /// <summary>
        /// Gets the warning message.
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="LoadWarningEventArgs"/> class.
        /// </summary>
        /// <param name="message">The warning message.</param>
        public LoadWarningEventArgs(string message)
        {
            Message = message;
        }
    }
}