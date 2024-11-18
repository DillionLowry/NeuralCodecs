namespace NeuralCodecs.Core.Loading
{
    public class CachedFile
    {
        public string Path { get; set; } = "";
        public string Hash { get; set; } = "";
        public string FullName { get; set; }
        public string Name { get; set; }
        public int Size { get; set; }
    }
}