// NeuralCodecs.Diagnostics/CodecDiagnostics.cs
using System.Diagnostics.Tracing;

namespace NeuralCodecs.Diagnostics
{
    // EventSource for ETW events
    [EventSource(Name = "NeuralCodecs-Diagnostics")]
    public sealed class CodecEventSource : EventSource
    {
        public static readonly CodecEventSource Log = new CodecEventSource();

        [Event(1, Level = EventLevel.Informational)]
        public void ModuleExecution(string moduleName, long executionTimeMs, long memoryBytes)
            => WriteEvent(1, moduleName, executionTimeMs, memoryBytes);

        [Event(2, Level = EventLevel.Verbose)]
        public void TensorStats(string moduleName, string tensorName, double minValue, double maxValue, string shape)
            => WriteEvent(2, moduleName, tensorName, minValue, maxValue, shape);

        [Event(3, Level = EventLevel.Warning)]
        public void AnomalyDetected(string moduleName, string description)
            => WriteEvent(3, moduleName, description);
    }
}