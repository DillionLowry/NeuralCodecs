using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Diagnostics
{
    /// <summary>
    /// Provides logging functionality for tensors to help with model debugging and comparison
    /// </summary>
    public class TensorLogger
    {
        private Dictionary<string, List<torch.Tensor>> _loggedTensors = new();
        private readonly string _outputDirectory;
        private bool _enabled = true;

        public TensorLogger(string outputDirectory = "tensor_logs")
        {
            _outputDirectory = outputDirectory;
            Directory.CreateDirectory(_outputDirectory);
        }

        /// <summary>
        /// Enable or disable logging
        /// </summary>
        public bool Enabled
        {
            get => _enabled;
            set => _enabled = value;
        }

        /// <summary>
        /// Log a tensor at a specific layer/point in the model
        /// </summary>
        /// <param name="tensor">The tensor to log</param>
        /// <param name="layerName">Name or identifier of the layer</param>
        /// <param name="writeToFile">Whether to immediately write to file</param>
        public void LogTensor(torch.Tensor tensor, string layerName, bool writeToFile = true)
        {
            if (!_enabled) return;

            if (tensor.IsInvalid)
            {
                Console.WriteLine($"Warning: Invalid tensor provided for layer {layerName}");
                return;
            }

            using var scope = torch.NewDisposeScope();
            var detached = tensor.detach().cpu();

            if (!_loggedTensors.ContainsKey(layerName))
            {
                _loggedTensors[layerName] = new List<torch.Tensor>();
            }

            _loggedTensors[layerName].Add(detached.clone().MoveToOuterDisposeScope());

            if (writeToFile)
            {
                SaveTensorToFile(detached, layerName);
            }
        }

        // TODO: add item count, front/back or just front support
        /// <summary>
        /// Write a tensor to file with the specified name
        /// </summary>
        /// <param name="tensor">The tensor to write</param>
        /// <param name="name">Name for the file (without extension)</param>
        /// <param name="precision">Number of decimal places</param>
        public void WriteTensorToFile(torch.Tensor tensor, string name, int precision = 30)
        {
            if (!_enabled) return;

            var filePath = Path.Combine(_outputDirectory, $"{name.Replace('.', '_')}_csharp.txt");
            
            try
            {
                // Convert the tensor to an array
                var tensorArray = tensor.clone().cpu().detach().reshape(-1).to(torch.float32).data<float>().ToArray();

                // Create a format string for the specified precision
                string format = $"F{precision}";

                // Write the tensor values to the file
                using var writer = new StreamWriter(filePath, false);
                
                writer.WriteLine($"Tensor values ({tensorArray.Length} elements):");
                if (tensorArray.Length > 400)
                {
                    // If the tensor is large, show the first and last 200 elements
                    writer.Write($"{string.Join(", ", tensorArray.Take(200).Select(x => x.ToString(format)))}");
                    writer.WriteLine($", {string.Join(", ", tensorArray.TakeLast(200).Select(x => x.ToString(format)))}");
                }
                else
                {
                    // Otherwise show all elements
                    writer.WriteLine($"{string.Join(", ", tensorArray.Select(x => x.ToString(format)))}");
                }
                writer.WriteLine();

                Console.WriteLine($"Tensor logged to {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error writing tensor to file: {ex.Message}");
            }
        }
        private void SaveTensorToFile(Tensor tensor, string path)
        {
            using var fs = File.Create(path);
            using var writer = new BinaryWriter(fs);

            // Write header - format:
            // - Magic number (0x1E2D3C4B)
            // - Number of dimensions
            // - Shape dimensions
            // - Data type code
            // - Data

            writer.Write(0x1E2D3C4B); // Magic number
            writer.Write(tensor.shape.Length);
            foreach (var dim in tensor.shape)
            {
                writer.Write(dim);
            }

            // Data type code (matching numpy dtypes)
            var dtype = GetDTypeCode(tensor.dtype);
            writer.Write((byte)dtype);

            // Convert to float32 and write data
            var flatTensor = tensor.to(ScalarType.Float32).flatten();
            var buffer = new byte[flatTensor.numel() * 4];
            Buffer.BlockCopy(flatTensor.data<float>().ToArray(), 0, buffer, 0, buffer.Length);
            writer.Write(buffer);
        }

        private byte GetDTypeCode(ScalarType dtype)
        {
            return dtype switch
            {
                ScalarType.Float32 => 0x23,  // np.float32
                ScalarType.Float64 => 0x24,  // np.float64
                ScalarType.Int32 => 0x21,    // np.int32
                ScalarType.Int64 => 0x22,    // np.int64
                _ => 0x23 // Default to float32
            };
        }
        /// <summary>
        /// Compute and log statistics about a tensor
        /// </summary>
        /// <param name="tensor">The tensor to analyze</param>
        /// <param name="name">Name for identifying the tensor</param>
        public void LogTensorStatistics(torch.Tensor tensor, string name)
        {
            if (!_enabled) return;

            try
            {
                var min = tensor.min().item<float>();
                var max = tensor.max().item<float>();
                var mean = tensor.mean().item<float>();
                var std = tensor.std().item<float>();

                var filePath = Path.Combine(_outputDirectory, $"{name.Replace('.', '_')}_stats_csharp.txt");
                using var writer = new StreamWriter(filePath, false);
                
                writer.WriteLine($"Tensor: {name}");
                writer.WriteLine($"Shape: [{string.Join(", ", tensor.shape)}]");
                writer.WriteLine($"Min: {min:F6}");
                writer.WriteLine($"Max: {max:F6}");
                writer.WriteLine($"Mean: {mean:F6}");
                writer.WriteLine($"Std: {std:F6}");
                
                Console.WriteLine($"Statistics logged to {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error logging tensor statistics: {ex.Message}");
            }
        }

        /// <summary>
        /// Clears all logged tensors
        /// </summary>
        public void Clear()
        {
            foreach (var tensors in _loggedTensors.Values)
            {
                foreach (var tensor in tensors)
                {
                    tensor.Dispose();
                }
                tensors.Clear();
            }
            _loggedTensors.Clear();
        }

        /// <summary>
        /// Dispose all resources
        /// </summary>
        public void Dispose()
        {
            Clear();
        }

        /// <summary>
        /// Helper class for generating Python code to load and compare tensors
        /// </summary>
        public void GenerateComparisonScript(string pythonScriptPath)
        {
            var sb = new StringBuilder();
            sb.AppendLine("import numpy as np");
            sb.AppendLine("import torch");
            sb.AppendLine("import matplotlib.pyplot as plt");
            sb.AppendLine("");
            sb.AppendLine("def load_tensor(path):");
            sb.AppendLine("    with open(path, 'rb') as f:");
            sb.AppendLine("        magic = np.fromfile(f, dtype=np.int32, count=1)[0]");
            sb.AppendLine("        if magic != 0x1E2D3C4B:");
            sb.AppendLine("            raise ValueError('Invalid file format')");
            sb.AppendLine("        ndims = np.fromfile(f, dtype=np.int32, count=1)[0]");
            sb.AppendLine("        shape = np.fromfile(f, dtype=np.int64, count=ndims)");
            sb.AppendLine("        dtype = np.fromfile(f, dtype=np.uint8, count=1)[0]");
            sb.AppendLine("        dtype_map = {");
            sb.AppendLine("            0x23: np.float32,");
            sb.AppendLine("            0x24: np.float64,");
            sb.AppendLine("            0x21: np.int32,");
            sb.AppendLine("            0x22: np.int64");
            sb.AppendLine("        }");
            sb.AppendLine("        data = np.fromfile(f, dtype=dtype_map[dtype])");
            sb.AppendLine("        return torch.from_numpy(data.reshape(shape))");
            sb.AppendLine("");
            sb.AppendLine("def compare_tensors(csharp_path, python_path):");
            sb.AppendLine("    csharp_tensor = load_tensor(csharp_path)");
            sb.AppendLine("    python_tensor = torch.load(python_path)");
            sb.AppendLine("    ");
            sb.AppendLine("    # Print stats");
            sb.AppendLine("    print(f'Shape match: {csharp_tensor.shape == python_tensor.shape}')");
            sb.AppendLine("    print(f'C# tensor stats: min={csharp_tensor.min():.6f}, max={csharp_tensor.max():.6f}, mean={csharp_tensor.mean():.6f}')");
            sb.AppendLine("    print(f'Python tensor stats: min={python_tensor.min():.6f}, max={python_tensor.max():.6f}, mean={python_tensor.mean():.6f}')");
            sb.AppendLine("    ");
            sb.AppendLine("    # Calculate difference");
            sb.AppendLine("    diff = (csharp_tensor - python_tensor).abs()");
            sb.AppendLine("    max_diff = diff.max().item()");
            sb.AppendLine("    mean_diff = diff.mean().item()");
            sb.AppendLine("    print(f'Max difference: {max_diff:.6f}')");
            sb.AppendLine("    print(f'Mean difference: {mean_diff:.6f}')");
            sb.AppendLine("    ");
            sb.AppendLine("    # Visualize if 2D or 3D");
            sb.AppendLine("    if len(csharp_tensor.shape) in [2, 3]:");
            sb.AppendLine("        plt.figure(figsize=(15, 5))");
            sb.AppendLine("        ");
            sb.AppendLine("        plt.subplot(131)");
            sb.AppendLine("        plt.title('C# Output')");
            sb.AppendLine("        plt.imshow(csharp_tensor.squeeze().detach().numpy())");
            sb.AppendLine("        plt.colorbar()");
            sb.AppendLine("        ");
            sb.AppendLine("        plt.subplot(132)");
            sb.AppendLine("        plt.title('Python Output')");
            sb.AppendLine("        plt.imshow(python_tensor.squeeze().detach().numpy())");
            sb.AppendLine("        plt.colorbar()");
            sb.AppendLine("        ");
            sb.AppendLine("        plt.subplot(133)");
            sb.AppendLine("        plt.title('Absolute Difference')");
            sb.AppendLine("        plt.imshow(diff.squeeze().numpy())");
            sb.AppendLine("        plt.colorbar()");
            sb.AppendLine("        ");
            sb.AppendLine("        plt.tight_layout()");
            sb.AppendLine("        plt.show()");
            sb.AppendLine("");
            sb.AppendLine("# Example usage:");
            sb.AppendLine("# compare_tensors('csharp_tensor.npz', 'python_tensor.pt')");

            File.WriteAllText(pythonScriptPath, sb.ToString());
        }
    }
}