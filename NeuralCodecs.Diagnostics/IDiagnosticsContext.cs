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
/// Interface defining diagnostics operations
/// </summary>
public interface IDiagnosticsContext : IDisposable
{
    void LogModuleExecution(string moduleName, long executionTimeMs, long memoryBytes, long[] inputShape = null, long[] outputShape = null);
    void LogTensor(string moduleName, string tensorName, Tensor tensor);
    void LogModuleOutput(string moduleName, string label, Tensor output);
    void LogModuleOutput(string moduleName, string label, IEnumerable<Tensor> outputs);
    void LogMessage(string moduleName, string message);
    void LogError(string moduleName, string message, Exception ex = null);
    void DetectAnomalies(string moduleName, Tensor tensor, double threshold = 5.0);
    void PrintSummary();
    void GenerateComparisonScript(string path);
    IDisposable TrackScope(string moduleName, Tensor input);
    bool IsEnabled { get; }
}