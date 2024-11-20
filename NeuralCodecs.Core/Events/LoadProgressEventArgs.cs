namespace NeuralCodecs.Core.Events
{
    /// <summary>
    /// Provides data for load progress events.
    /// </summary>
    public class LoadProgressEventArgs : EventArgs
    {
        /// <summary>
        /// Gets the source identifier for the load operation.
        /// </summary>
        public string Source { get; }

        /// <summary>
        /// Gets the progress value between 0 and 1.
        /// </summary>
        public double Progress { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="LoadProgressEventArgs"/> class.
        /// </summary>
        /// <param name="source">The source identifier for the load operation.</param>
        /// <param name="progress">The progress value between 0 and 1.</param>
        public LoadProgressEventArgs(string source, double progress)
        {
            Source = source;
            Progress = progress;
        }
    }
}