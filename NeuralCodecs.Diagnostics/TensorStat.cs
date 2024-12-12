// NeuralCodecs.Diagnostics/CodecDiagnostics.cs
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Diagnostics
{
    public class TensorStat
    {
        public string Name { get; set; }
        public string Shape { get; set; }
        public double Min { get; set; }
        public double Max { get; set; }
        public double Mean { get; set; }
        public bool HasNaN { get; set; }
        public bool HasInf { get; set; }
        public int Step { get; set; }
    }
}