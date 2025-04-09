using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TorchSharp;

namespace NeuralCodecs.Diagnostics
{
    /// <summary>
    /// Provides functionality for comparing tensors between different implementations
    /// </summary>
    public class TensorComparison
    {
        private readonly string _csharpLogDir;
        private readonly string _pythonLogDir;
        private readonly string _outputDir;

        public TensorComparison(
            string csharpLogDir = "tensor_logs", 
            string pythonLogDir = "python_logs", 
            string outputDir = "comparison_results")
        {
            _csharpLogDir = csharpLogDir;
            _pythonLogDir = pythonLogDir;
            _outputDir = outputDir;
            
            Directory.CreateDirectory(_outputDir);
        }

        /// <summary>
        /// Compare two tensors and return statistics about their differences
        /// </summary>
        public static (float meanError, float maxError, float correlation) CompareTensors(
            torch.Tensor csharpTensor, 
            torch.Tensor pythonTensor)
        {
            if (csharpTensor.shape.Length != pythonTensor.shape.Length)
            {
                throw new ArgumentException($"Tensor shapes don't match: {string.Join(",", csharpTensor.shape)} vs {string.Join(",", pythonTensor.shape)}");
            }

            for (int i = 0; i < csharpTensor.shape.Length; i++)
            {
                if (csharpTensor.shape[i] != pythonTensor.shape[i])
                {
                    throw new ArgumentException($"Tensor dimension {i} doesn't match: {csharpTensor.shape[i]} vs {pythonTensor.shape[i]}");
                }
            }

            using var scope = torch.NewDisposeScope();
            
            // Calculate absolute difference
            var diff = (csharpTensor - pythonTensor).abs();
            
            // Calculate statistics
            float meanError = diff.mean().item<float>();
            float maxError = diff.max().item<float>();
            
            // Calculate correlation
            var csharpFlat = csharpTensor.reshape(-1);
            var pythonFlat = pythonTensor.reshape(-1);
            
            var csharpMean = csharpFlat.mean().item<float>();
            var pythonMean = pythonFlat.mean().item<float>();
            
            var csharpCentered = csharpFlat - csharpMean;
            var pythonCentered = pythonFlat - pythonMean;
            
            var numerator = (csharpCentered * pythonCentered).sum().item<float>();
            var csharpDenominator = (csharpCentered * csharpCentered).sum().item<float>();
            var pythonDenominator = (pythonCentered * pythonCentered).sum().item<float>();
            
            float correlation = numerator / (float)Math.Sqrt(csharpDenominator * pythonDenominator);
            
            return (meanError, maxError, correlation);
        }

        /// <summary>
        /// Compare tensors from files in C# and Python log directories
        /// </summary>
        public void CompareFromFiles()
        {
            var report = new StringBuilder();
            report.AppendLine("# Tensor Comparison Report");
            report.AppendLine($"Generated at: {DateTime.Now:yyyy-MM-dd HH:mm:ss}\n");
            
            var csharpFiles = Directory.GetFiles(_csharpLogDir, "*_csharp.txt")
                                      .Select(f => Path.GetFileNameWithoutExtension(f).Replace("_csharp", ""))
                                      .ToList();
            
            var pythonFiles = Directory.GetFiles(_pythonLogDir, "*_python.txt")
                                      .Select(f => Path.GetFileNameWithoutExtension(f).Replace("_python", ""))
                                      .ToList();
            
            var commonNames = csharpFiles.Intersect(pythonFiles).ToList();
            
            foreach (var name in commonNames)
            {
                report.AppendLine($"## Layer: {name}");
                
                var csharpFile = Path.Combine(_csharpLogDir, $"{name}_csharp.txt");
                var pythonFile = Path.Combine(_pythonLogDir, $"{name}_python.txt");
                
                try
                {
                    var csharpTensor = LoadTensorFromFile(csharpFile);
                    var pythonTensor = LoadTensorFromFile(pythonFile);
                    
                    var (meanError, maxError, correlation) = CompareTensors(csharpTensor, pythonTensor);
                    
                    report.AppendLine($"- Mean Absolute Error: {meanError:E4}");
                    report.AppendLine($"- Max Absolute Error: {maxError:E4}");
                    report.AppendLine($"- Correlation: {correlation:F4}\n");
                }
                catch (Exception ex)
                {
                    report.AppendLine($"- Error comparing tensors: {ex.Message}\n");
                }
            }
            
            File.WriteAllText(Path.Combine(_outputDir, "comparison_report.md"), report.ToString());
        }

        /// <summary>
        /// Load a tensor from a file created by TensorLogger
        /// </summary>
        private static torch.Tensor LoadTensorFromFile(string filePath)
        {
            string content = File.ReadAllText(filePath);
            
            int startIndex = content.IndexOf('[');
            if (startIndex == -1) startIndex = 0;
            
            int endIndex = content.IndexOf(']');
            if (endIndex == -1) endIndex = content.Length;
            
            string valuesText = content.Substring(startIndex, endIndex - startIndex).Trim();
            
            string[] valueStrings = valuesText.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
            
            float[] values = valueStrings.Select(s => float.Parse(s.Trim())).ToArray();
            
            return torch.tensor(values);
        }
    }
}
