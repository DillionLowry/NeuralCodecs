using NeuralCodecs.Core.Models;
using System.Text.Json.Serialization;

namespace NeuralCodecs.Core.Loading
{
    public abstract class ModelConfig
    {
        [JsonConstructor]
        public ModelConfig()
        { }
        public Device Device { get; set; }
        /// <summary>
        /// The sampling rate of the audio the model was trained on
        /// </summary>
        public int SamplingRate { get; set; }

        /// <summary>
        /// Model architecture type identifier
        /// </summary>
        public string Architecture { get; set; } = "";

        /// <summary>
        /// Model version string
        /// </summary>
        public string Version { get; set; } = "1.0";

        /// <summary>
        /// Additional metadata key-value pairs
        /// </summary>
        public Dictionary<string, string> Metadata { get; set; } = new();

        public abstract bool Validate();
    }
}