// NeuralCodecs.Diagnostics/CodecDiagnostics.cs
namespace NeuralCodecs.Diagnostics
{
    /// <summary>
    /// Statistics tracked for each module
    /// </summary>
    public class ModuleStats
    {
        public int CallCount { get; set; }
        public long TotalExecutionTime { get; set; }
        public long MaxMemoryBytes { get; set; }
        public double AverageMemoryBytes { get; set; }
        public List<TensorStat> TensorStats { get; } = new();
        public List<(long[] InputShape, long[] OutputShape)> ShapeHistory { get; } = new();
        public Dictionary<string, List<double>> GradientHistory { get; } = new();
        public List<ErrorRecord> Errors { get; } = new List<ErrorRecord>();

    }
}