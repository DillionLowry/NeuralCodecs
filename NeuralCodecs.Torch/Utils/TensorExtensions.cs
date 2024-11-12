using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Utils;

public static class TensorExtensions
{
    public static (int B, int C, int T) GetDimensions(this Tensor tensor)
    {
        return ((int)tensor.size(0), (int)tensor.size(1), (int)tensor.size(2));
    }

    public static Tensor L2Normalize(this Tensor input, double p = 2.0, int dim = 1, bool keepDim = true, double eps = 1e-12)
    {
        var norm = input.pow(2)
           .sum(dim, keepdim: keepDim)
           .sqrt()
           .add(1e-7); // Add small epsilon for stability
        return input.div(norm);
    }

    public static Tensor Detach(this Tensor tensor)
    {
        using var scope = NewDisposeScope();
        return tensor.detach().MoveToOuterDisposeScope();
    }

    public static void PrintVals(this Tensor tensor, int count = 20, string name = "")
    {
        // Convert the tensor to an array
        var tensorArray = tensor.clone().reshape(-1).to(float32).data<float>().ToArray();
        Console.WriteLine(name ?? "Tensor values:");

        Console.Write($"{string.Join(", ", tensorArray.Take(count / 2).Select(x => x.ToString("F30")))},");
        Console.WriteLine($" , {string.Join(", ", tensorArray.TakeLast(count / 2).Select(x => x.ToString("F30")))}");
    }

    public static void WriteTensorToFile(this Tensor tensor, string filePath, int precision = 30, bool append = true, int? count = 200)
    {
        // Convert the tensor to an array
        var tensorArray = tensor.clone().cpu().detach().reshape(-1).to(float32).data<float>().ToArray();

        // Create a format string for the specified precision
        string format = $"F{precision}";

        // Write the tensor values to the file
        using (var writer = new StreamWriter(filePath, append))
        {
            Console.WriteLine($"Tensor values ({tensorArray.Length} elements), printing {Math.Min(tensorArray.Length, count.Value)} elements:");
            if (count is not null && count < tensorArray.Length)
            {
                writer.Write($"{string.Join(", ", tensorArray.Take(Math.Min(tensorArray.Length, count.Value) / 2).Select(x => x.ToString(format)))}");
                writer.WriteLine($", {string.Join(", ", tensorArray.TakeLast(Math.Min(tensorArray.Length, count.Value) / 2).Select(x => x.ToString(format)))}");
                writer.WriteLine();
            }
            else
            {
                writer.WriteLine($"{string.Join(", ", tensorArray.Select(x => x.ToString(format)))}");
                writer.WriteLine();
            }
        }
    }


}