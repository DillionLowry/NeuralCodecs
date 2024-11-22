namespace NeuralCodecs.Core.Loading
{
    /// <summary>
    /// Represents metadata information for a neural network model.
    /// </summary>
    public class ModelMetadata
    {
        /// <summary>
        /// Gets or sets the original source location of the model.
        /// </summary>
        public string Source { get; set; } = "";

        /// <summary>
        /// Gets or sets a value indicating whether the model is cached locally.
        /// </summary>
        public bool IsCached { get; set; }

        /// <summary>
        /// Gets or sets the last modification timestamp of the model.
        /// </summary>
        public DateTime LastModified { get; set; }

        /// <summary>
        /// Gets or sets the author of the model.
        /// </summary>
        public string Author { get; set; } = "";

        /// <summary>
        /// Gets or sets the list of tags associated with the model.
        /// </summary>
        public List<string> Tags { get; set; } = new();

        /// <summary>
        /// Gets or sets the backend framework used by the model.
        /// </summary>
        public string Backend { get; set; } = "";

        /// <summary>
        /// Gets or sets the size of the model in bytes.
        /// </summary>
        public long Size { get; internal set; }

        /// <summary>
        /// Gets or sets the filename of the model.
        /// </summary>
        public string FileName { get; set; }

        /// <summary>
        /// Gets or sets the filename of the model's configuration file.
        /// </summary>
        public string ConfigFileName { get; set; }
    }
}