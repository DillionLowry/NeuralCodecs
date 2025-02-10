// NeuralCodecs.Diagnostics/CodecDiagnostics.cs
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

public class ErrorRecord
{
    public string Message { get; set; }
    public Exception Exception { get; set; }
    public DateTime Timestamp { get; set; }
    public int Step { get; set; }
}