using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Utils;

public static class TensorExtensions
{
    /// <summary>
    /// Performs L2 normalization of the input tensor along the specified dimension
    /// </summary>
    /// <param name="input">The input tensor</param>
    /// <param name="p">The power for the norm calculation (default is 2.0)</param>
    /// <param name="dim">The dimension along which to normalize (default is 1)</param>
    /// <param name="keepDim">Whether to keep the dimensions (default is true)</param>
    /// <param name="eps">A small epsilon value for numerical stability (default is 1e-12)</param>
    /// <returns>The L2 normalized tensor</returns>
    public static Tensor L2Normalize(this Tensor input, double p = 2.0, int dim = 1, bool keepDim = true, double eps = 1e-12)
    {
        return input.div(input.pow(2)
                           .sum(dim, keepdim: keepDim)
                           .sqrt()
                           .add(eps));
    }

    /// <summary>
    /// Implements in-place rolling (Torch roll_) for the last dimension
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="shift"></param>
    /// <param name="dim"></param>
    public static void RollInPlace(this Tensor tensor, int shift, int dim = -1)
    {
        if (shift == 0)
        {
            return;
        }

        var n = tensor.size(dim);
        shift = (int)((shift % n + n) % n); // Handle negative shifts

        // Store the wrapped portion
        using var temp = tensor.narrow(dim, n - shift, shift).clone();

        // Shift the main portion
        tensor.narrow(dim, shift, n - shift).copy_(tensor.narrow(dim, 0, n - shift));

        // Copy wrapped portion to start
        tensor.narrow(dim, 0, shift).copy_(temp);
    }

    /// <summary>
    /// Converts the tensor to a float array.
    /// </summary>
    /// <param name="tensor">The input tensor</param>
    /// <returns>A float array containing the tensor data</returns>
    public static float[] ToFloatArray(this Tensor tensor)
    {
        return tensor.cpu().detach().to(float32).data<float>().ToArray();
    }

    public static void WriteTensorToFile(this Tensor tensor, string filePath, int precision = 30, bool append = false, int? count = 200)
    {
        // Convert the tensor to an array
        var tensorArray = tensor.clone().cpu().detach().reshape(-1).to(float32).data<float>().ToArray();

        // Create a format string for the specified precision
        string format = $"F{precision}";

        // Write the tensor values to the file
        using var writer = new StreamWriter(filePath, append);
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