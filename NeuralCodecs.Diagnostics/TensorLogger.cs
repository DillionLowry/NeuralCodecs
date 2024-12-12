// NeuralCodecs.Diagnostics/CodecDiagnostics.cs
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

public class TensorLogger
{
    private readonly string _outputDir;
    private readonly Dictionary<string, int> _callCounts = new();
    private readonly object _lock = new object();
    private bool _enabled;

    public TensorLogger(string outputDir)
    {
        _outputDir = outputDir;
        _enabled = true;
        Directory.CreateDirectory(_outputDir);
    }

    public void Enable() => _enabled = true;

    public void Disable() => _enabled = false;

    /// <summary>
    /// Logs tensor data to a file in a numpy-compatible format
    /// </summary>
    public void LogTensor(string moduleName, string label, Tensor tensor)
    {
        if (!_enabled || tensor.IsInvalid) return;

        try
        {
            lock (_lock)
            {
                if (!_callCounts.ContainsKey(moduleName))
                    _callCounts[moduleName] = 0;

                var callNum = _callCounts[moduleName];
                var fileName = $"{moduleName}_{callNum:D4}_{label}.npz";
                var path = Path.Combine(_outputDir, fileName);

                using var noGrad = torch.no_grad();
                var tensorCpu = tensor.cpu().detach();
                SaveTensorToFile(tensorCpu, path);

                _callCounts[moduleName]++;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error logging tensor: {ex.Message}");
        }
    }

    /// <summary>
    /// Logs tuple or list of tensors
    /// </summary>
    public void LogTensors(string moduleName, string label, IEnumerable<Tensor> tensors)
    {
        if (!_enabled) return;

        var i = 0;
        foreach (var tensor in tensors)
        {
            LogTensor(moduleName, $"{label}_{i}", tensor);
            i++;
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