using NeuralCodecs.Core.Interfaces;

namespace NeuralCodecs.Core.Models
{
    /// <summary>
    /// Model information
    /// </summary>
    public class ModelInfo
    {
        public string Source { get; set; } = "";
        //public IModelConfig? Config { get; set; }
        public bool IsCached { get; set; }
        public DateTime LastModified { get; set; }
        public string Author { get; set; } = "";
        public List<string> Tags { get; set; } = new();
        public string Backend { get; set; } = "";
        public long Size { get; internal set; }
    }
}