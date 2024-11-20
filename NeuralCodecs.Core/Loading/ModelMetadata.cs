namespace NeuralCodecs.Core.Loading
{
    public class ModelMetadata
    {
        public string Source { get; set; } = "";
        public bool IsCached { get; set; }
        public DateTime LastModified { get; set; }
        public string Author { get; set; } = "";
        public List<string> Tags { get; set; } = new();
        public string Backend { get; set; } = "";
        public long Size { get; internal set; }
        public string FileName { get; set; }
        public string ConfigFileName { get; set; }
    }
}