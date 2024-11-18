namespace NeuralCodecs.Core.Loading
{
    public class CacheMetadata
    {
        public string ModelId { get; set; } = "";
        public string Revision { get; set; } = "";
        public DateTime Timestamp { get; set; }
        public int MaxAgeInDays { get; set; }
        public List<CachedFile> Files { get; set; } = new();
    }
}