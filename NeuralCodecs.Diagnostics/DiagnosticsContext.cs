// NeuralCodecs.Diagnostics/CodecDiagnostics.cs
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Diagnostics
{
    /// <summary>
    /// Main diagnostics context for neural codec modules
    /// </summary>
    public class DiagnosticsContext : IDiagnosticsContext
    {
        private readonly bool _enabled;
        private readonly string _logFile;
        private readonly string _tensorboardDir;
        private readonly string _tensorOutputDir;
        private readonly HashSet<string> _modulesToTrack;
        private readonly ConcurrentDictionary<string, ModuleStats> _stats;
        private readonly Stopwatch _globalTimer;
        private readonly StringBuilder _logBuffer;
        private readonly TextWriter _logWriter;
        private int _step;
        private readonly object _lock = new object();
        private readonly TensorLogger _tensorLogger;
        public bool IsEnabled => _enabled;

        public DiagnosticsContext(
            bool enabled = true,
            string logFile = null,
            string tensorboardDir = null,
            string tensorOutputDir = null,
            IEnumerable<string> modulesToTrack = null)
        {
            _enabled = enabled;
            _logFile = logFile;
            _tensorboardDir = tensorboardDir;
            _tensorOutputDir = tensorOutputDir;
            _modulesToTrack = modulesToTrack?.ToHashSet() ?? new HashSet<string>();
            _stats = new ConcurrentDictionary<string, ModuleStats>();
            _logBuffer = new StringBuilder();
            _globalTimer = Stopwatch.StartNew();

            if (!string.IsNullOrEmpty(_logFile))
            {
                var logDir = Path.GetDirectoryName(_logFile);
                if (!string.IsNullOrEmpty(logDir))
                    Directory.CreateDirectory(logDir);
                _logWriter = new StreamWriter(_logFile, true);
            }

            if (!string.IsNullOrEmpty(_tensorboardDir))
            {
                Directory.CreateDirectory(_tensorboardDir);
            }
            //if (!string.IsNullOrEmpty(tensorOutputDir))
            //{
            //    _tensorLogger = new TensorLogger(tensorOutputDir);
            //    if (!enabled) _tensorLogger.Disable();
            //}
        }
        public void LogTensor(string moduleName, string tensorName, Tensor tensor)
        {
            if (!_enabled || !ShouldTrackModule(moduleName) || tensor.IsInvalid)
                return;

            try
            {
                using var noGrad = torch.no_grad();

                // Basic tensor stats
                double min = double.NaN, max = double.NaN, mean = double.NaN;
                bool hasNaN = false, hasInf = false;

                tensor = tensor.detach();
                if (tensor.numel() > 0)
                {
                    min = tensor.min().ToDouble();
                    max = tensor.max().ToDouble();
                    mean = tensor.mean().ToDouble();

                    hasNaN = tensor.isnan().any().ToBoolean();
                    hasInf = tensor.isinf().any().ToBoolean();
                }

                // Format shape info
                var shapeStr = string.Join("x", tensor.shape);

                // Save tensor stats
                lock (_lock)
                {
                    if (!_stats.ContainsKey(moduleName))
                        _stats[moduleName] = new ModuleStats();

                    var stats = _stats[moduleName];
                    stats.TensorStats.Add(new TensorStat
                    {
                        Name = tensorName,
                        Shape = shapeStr,
                        Min = min,
                        Max = max,
                        Mean = mean,
                        HasNaN = hasNaN,
                        HasInf = hasInf,
                        Step = _step
                    });
                }

                // Build log message
                var msg = new StringBuilder();
                msg.AppendLine($"Tensor: {moduleName}/{tensorName}");
                msg.AppendLine($"  Shape: {shapeStr}");
                msg.AppendLine($"  Stats: min={min:E3}, max={max:E3}, mean={mean:E3}");
                if (hasNaN || hasInf)
                    msg.AppendLine($"  Issues: NaN={hasNaN}, Inf={hasInf}");

                LogMessage(moduleName, msg.ToString());

                // Save tensor data if output directory specified
                if (!string.IsNullOrEmpty(_tensorOutputDir))
                {
                    var fileName = $"{_step:D6}_{moduleName}_{tensorName}.npz";
                    var path = Path.Combine(_tensorOutputDir, fileName);

                    try
                    {
                        using var fs = File.Create(path);
                        using var writer = new BinaryWriter(fs);

                        // Write shape
                        writer.Write(tensor.shape.Length);
                        foreach (var dim in tensor.shape)
                            writer.Write(dim);

                        // Write data
                        var flatTensor = tensor.cpu().to(ScalarType.Float32).flatten();
                        var data = flatTensor.data<float>().ToArray();
                        foreach (var value in data)
                            writer.Write(value);
                    }
                    catch (Exception ex)
                    {
                        LogError(moduleName, $"Failed to save tensor to {path}", ex);
                    }
                }
            }
            catch (Exception ex)
            {
                LogError(moduleName, $"Error logging tensor {tensorName}", ex);
            }
        }

        public void LogMessage(string moduleName, string message)
        {
            if (!_enabled || !ShouldTrackModule(moduleName))
                return;

            try
            {
                var timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
                var formattedMessage = $"[{timestamp}] [{moduleName}] {message}";

                lock (_lock)
                {
                    // Add to in-memory buffer
                    _logBuffer.AppendLine(formattedMessage);

                    // Write to file if configured
                    _logWriter?.WriteLine(formattedMessage);
                    _logWriter?.Flush();

                    // Also write to debug output
                    Debug.WriteLine(formattedMessage);
                }
            }
            catch (Exception ex)
            {
                // If logging fails, try to write to debug output at least
                Debug.WriteLine($"Error logging message: {ex.Message}");
            }
        }

        public void LogError(string moduleName, string message, Exception ex = null)
        {
            if (!_enabled || !ShouldTrackModule(moduleName))
                return;

            try
            {
                var timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
                var sb = new StringBuilder();
                sb.AppendLine($"[{timestamp}] [{moduleName}] ERROR: {message}");

                if (ex != null)
                {
                    sb.AppendLine("Exception Details:");
                    sb.AppendLine($"  Type: {ex.GetType().FullName}");
                    sb.AppendLine($"  Message: {ex.Message}");
                    sb.AppendLine($"  Stack Trace:");
                    sb.AppendLine(ex.StackTrace);

                    if (ex.InnerException != null)
                    {
                        sb.AppendLine("Inner Exception:");
                        sb.AppendLine($"  Type: {ex.InnerException.GetType().FullName}");
                        sb.AppendLine($"  Message: {ex.InnerException.Message}");
                        sb.AppendLine($"  Stack Trace:");
                        sb.AppendLine(ex.InnerException.StackTrace);
                    }
                }

                var errorMessage = sb.ToString();

                lock (_lock)
                {
                    // Add to stats
                    if (!_stats.ContainsKey(moduleName))
                        _stats[moduleName] = new ModuleStats();

                    _stats[moduleName].Errors.Add(new ErrorRecord
                    {
                        Message = message,
                        Exception = ex,
                        Timestamp = DateTime.Now,
                        Step = _step
                    });

                    // Add to buffer
                    _logBuffer.AppendLine(errorMessage);

                    // Write to file
                    _logWriter?.WriteLine(errorMessage);
                    _logWriter?.Flush();

                    // Write to debug output
                    Debug.WriteLine(errorMessage);
                }
            }
            catch (Exception loggingEx)
            {
                // Last resort - write to debug output
                Debug.WriteLine($"Error logging error: {loggingEx.Message}");
                Debug.WriteLine($"Original error: {message}");
                if (ex != null)
                    Debug.WriteLine($"Original exception: {ex}");
            }
        }


        public void LogModuleOutput(string moduleName, string label, Tensor output)
        {
            if (!_enabled || !ShouldTrackModule(moduleName)) return;
            _tensorLogger?.LogTensor(moduleName, label, output);
        }

        public void LogModuleOutput(string moduleName, string label, IEnumerable<Tensor> outputs)
        {
            if (!_enabled || !ShouldTrackModule(moduleName)) return;
            _tensorLogger?.LogTensors(moduleName, label, outputs);
        }

        public void GenerateComparisonScript(string path)
        {
            _tensorLogger?.GenerateComparisonScript(path);
        }

        public void LogModuleExecution(
            string moduleName,
            long executionTimeMs,
            long memoryBytes,
            long[] inputShape = null,
            long[] outputShape = null)
        {
            if (!_enabled || !ShouldTrackModule(moduleName)) return;

            var stats = _stats.GetOrAdd(moduleName, _ => new ModuleStats());

            lock (_lock)
            {
                stats.CallCount++;
                stats.TotalExecutionTime += executionTimeMs;
                stats.MaxMemoryBytes = Math.Max(stats.MaxMemoryBytes, memoryBytes);
                stats.AverageMemoryBytes =
                    ((stats.AverageMemoryBytes * (stats.CallCount - 1)) + memoryBytes) / stats.CallCount;

                if (inputShape != null && outputShape != null)
                {
                    stats.ShapeHistory.Add((inputShape, outputShape));
                }
            }

            CodecEventSource.Log.ModuleExecution(moduleName, executionTimeMs, memoryBytes);

            LogMessage($"Module {moduleName} execution: {executionTimeMs}ms, Memory: {memoryBytes / 1024.0:F2}KB");
        }

        public void LogTensorStats(string moduleName, string tensorName, Tensor tensor)
        {
            if (!_enabled || !ShouldTrackModule(moduleName) || tensor.IsInvalid) return;

            using var noGrad = torch.no_grad();

            var min = tensor.min().ToDouble();
            var max = tensor.max().ToDouble();
            var shape = string.Join("x", tensor.shape);

            lock (_lock)
            {
                var stats = _stats.GetOrAdd(moduleName, _ => new ModuleStats());
                stats.TensorStats.Add(new TensorStat
                {
                    Name = tensorName,
                    Min = min,
                    Max = max
                });
            }

            CodecEventSource.Log.TensorStats(moduleName, tensorName, min, max, shape);

            if (!string.IsNullOrEmpty(_tensorboardDir))
            {
                SaveTensorVisualization(moduleName, tensorName, tensor);
            }
        }

        public void LogGradients(string moduleName, string parameterName, Tensor gradient)
        {
            if (!_enabled || !ShouldTrackModule(moduleName) || gradient.IsInvalid) return;

            using var noGrad = torch.no_grad();
            var gradNorm = gradient.norm().ToDouble();

            lock (_lock)
            {
                var stats = _stats.GetOrAdd(moduleName, _ => new ModuleStats());
                if (!stats.GradientHistory.ContainsKey(parameterName))
                {
                    stats.GradientHistory[parameterName] = new List<double>();
                }
                stats.GradientHistory[parameterName].Add(gradNorm);
            }
        }

        public void DetectAnomalies(string moduleName, Tensor tensor, double threshold = 5.0)
        {
            if (!_enabled || !ShouldTrackModule(moduleName) || tensor.IsInvalid) return;

            using var noGrad = torch.no_grad();

            var stats = tensor.std().ToDouble();
            if (double.IsNaN(stats) || double.IsInfinity(stats) || Math.Abs(stats) > threshold)
            {
                var message = $"Anomaly detected: std={stats:F3}";
                CodecEventSource.Log.AnomalyDetected(moduleName, message);
                LogMessage($"WARNING: {moduleName} - {message}");
            }
        }

        private void SaveTensorVisualization(string moduleName, string tensorName, Tensor tensor)
        {
            if (_step % 100 != 0) return; // Only save periodically

            try
            {
                using var noGrad = torch.no_grad();
                var data = tensor.cpu().to(ScalarType.Float32);

                // Save raw data for visualization
                var fileName = $"{_step}_{moduleName}_{tensorName}.bin";
                var path = Path.Combine(_tensorboardDir, fileName);

                using var fs = File.Create(path);
                using var writer = new BinaryWriter(fs);

                // Write shape info
                writer.Write(tensor.shape.Length);
                foreach (var dim in tensor.shape)
                {
                    writer.Write(dim);
                }

                // Write flattened tensor data
                var flat = data.flatten();
                for (var i = 0; i < flat.shape[0]; i++)
                {
                    writer.Write(flat[i].ToSingle());
                }
            }
            catch (Exception ex)
            {
                LogMessage($"Error saving tensor visualization: {ex.Message}");
            }
        }

        private bool ShouldTrackModule(string moduleName)
        {
            return !_modulesToTrack.Any() || _modulesToTrack.Contains(moduleName);
        }

        private void LogMessage(string message)
        {
            var timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
            var logMessage = $"[{timestamp}] {message}";
            // if writer is disposed, it will throw an exception
            // so we need to check if it is disposed or not
            if (_logWriter != null)
            {
                try
                {
                    _logWriter.WriteLine(logMessage);
                }
                catch (ObjectDisposedException)
                {
                    // Handle the case where the writer is disposed
                }
            }
            Debug.WriteLine(logMessage);
        }

        public void PrintSummary()
        {
            if (!_enabled) return;

            var summary = new System.Text.StringBuilder();
            summary.AppendLine("\nNeural Codec Diagnostics Summary");
            summary.AppendLine("==============================");

            foreach (var kvp in _stats.OrderByDescending(x => x.Value.TotalExecutionTime))
            {
                var stats = kvp.Value;
                if (stats.CallCount == 0) continue;

                summary.AppendLine($"\nModule: {kvp.Key}");
                summary.AppendLine($"  Calls: {stats.CallCount}");
                summary.AppendLine($"  Total Time: {stats.TotalExecutionTime}ms");
                summary.AppendLine($"  Avg Time: {stats.TotalExecutionTime / (double)stats.CallCount:F2}ms");
                summary.AppendLine($"  Max Memory: {stats.MaxMemoryBytes / 1024.0:F2}KB");

                if (stats.GradientHistory.Any())
                {
                    summary.AppendLine("  Gradient Norms:");
                    foreach (var (param, norms) in stats.GradientHistory)
                    {
                        var avg = norms.Average();
                        var max = norms.Max();
                        summary.AppendLine($"    {param}: avg={avg:F3}, max={max:F3}");
                    }
                }
            }

            LogMessage(summary.ToString());
        }
        public void Dispose()
        {
            try
            {
                _logWriter?.Dispose();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error disposing diagnostics context: {ex.Message}");
            }
        }
        public void IncrementStep() => Interlocked.Increment(ref _step);

        public IDisposable TrackScope(string moduleName, Tensor input)
        {
            return new ExecutionScope(this, moduleName, input);
        }

        private class ExecutionScope : IDisposable
        {
            private readonly DiagnosticsContext _context;
            private readonly string _moduleName;
            private readonly Tensor _input;
            private readonly Stopwatch _sw;
            private readonly long _initialMemory;

            public ExecutionScope(DiagnosticsContext context, string moduleName, Tensor input)
            {
                _context = context;
                _moduleName = moduleName;
                _input = input;
                _sw = Stopwatch.StartNew();
                _initialMemory = GC.GetTotalMemory(false);
            }

            public void Dispose()
            {
                var finalMemory = GC.GetTotalMemory(false);
                _context.LogModuleExecution(
                    _moduleName,
                    _sw.ElapsedMilliseconds,
                    finalMemory - _initialMemory,
                    _input.shape);
                _sw.Stop();
            }
        }



    }
}
