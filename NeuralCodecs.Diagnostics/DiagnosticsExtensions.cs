// NeuralCodecs.Diagnostics/CodecDiagnostics.cs
using static TorchSharp.torch;

namespace NeuralCodecs.Diagnostics
{
    /// <summary>
    /// Extension methods for tensor diagnostics
    /// </summary>
    public static class DiagnosticsExtensions
    {
        public static void LogTensor(
            this DiagnosticsContext context,
            string moduleName,
            string tensorName,
            Tensor tensor,
            bool checkAnomalies = true)
        {
            context.LogTensorStats(moduleName, tensorName, tensor);
            if (checkAnomalies)
            {
                context.DetectAnomalies(moduleName, tensor);
            }
        }
    }
}