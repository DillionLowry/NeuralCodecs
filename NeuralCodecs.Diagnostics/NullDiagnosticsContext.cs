// NeuralCodecs.Diagnostics/CodecDiagnostics.cs
using NeuralCodecs.Diagnostics;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

/// <summary>
/// No-op implementation for when diagnostics are disabled
/// </summary>
public class NullDiagnosticsContext : IDiagnosticsContext
{
    public bool IsEnabled => false;
    private static readonly IDisposable NullScope = new DummyScope();

    public void Dispose() { }
    public void GenerateComparisonScript(string path) { }
    public void LogError(string moduleName, string message, Exception ex = null) { }
    public void LogMessage(string moduleName, string message) { }
    public void LogModuleExecution(string moduleName, long executionTimeMs, long memoryBytes, long[] inputShape = null, long[] outputShape = null) { }
    public void LogModuleOutput(string moduleName, string label, Tensor output) { }
    public void LogModuleOutput(string moduleName, string label, IEnumerable<Tensor> outputs) { }
    public void LogTensor(string moduleName, string tensorName, Tensor tensor) { }
    public void DetectAnomalies(string moduleName, Tensor tensor, double threshold = 5.0) { }
    public void PrintSummary() { }
    public IDisposable TrackScope(string moduleName, Tensor input) => NullScope;

    private class DummyScope : IDisposable
    {
        public void Dispose() { }
    }
}