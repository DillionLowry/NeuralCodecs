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
/// Factory for creating diagnostics contexts
/// </summary>
public static class DiagnosticsFactory
{
    private static readonly IDiagnosticsContext NullContext = new NullDiagnosticsContext();

    public static IDiagnosticsContext Create(
        bool enabled = true,
        string logFile = null,
        string tensorboardDir = null,
        string tensorOutputDir = null,
        IEnumerable<string> modulesToTrack = null)
    {
        return enabled
            ? (IDiagnosticsContext)new DiagnosticsContext(enabled, logFile, tensorboardDir, tensorOutputDir, modulesToTrack)
            : NullContext;
    }
}