using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Utils;

public static class TensorExtensions
{
    public static float[] ToFloatArray(this Tensor tensor)
    {
        return tensor.cpu().detach().to(torch.float32).data<float>().ToArray();
    }

    /// <summary>
    /// Gets the dimensions of the tensor as a tuple (B, C, T)
    /// </summary>
    /// <param name="tensor">The input tensor</param>
    /// <returns>A tuple containing the dimensions (B, C, T)</returns>
    public static (int B, int C, int T) GetDimensions(this Tensor tensor)
    {
        if (tensor.dim() != 3)
        {
            throw new ArgumentException("Tensor must have 3 dimensions (B,C,T)");
        }
        return ((int)tensor.size(0), (int)tensor.size(1), (int)tensor.size(2));
    }

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
    /// Detaches the tensor from the computation graph
    /// </summary>
    /// <param name="tensor">The input tensor</param>
    /// <returns>The detached tensor</returns>
    public static Tensor Detach(this Tensor tensor)
    {
        if (tensor.IsInvalid) return tensor;
        using var scope = torch.NewDisposeScope();
        return tensor.detach().MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Implements in-place rolling (Torch roll_) for the last dimension
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="shift"></param>
    /// <param name="dim"></param>
    public static void RollInPlace(this torch.Tensor tensor, int shift, int dim = -1)
    {
        if (shift == 0) return;

        var n = tensor.size(dim);
        shift = (int)(((shift % n) + n) % n); // Handle negative shifts

        // Store the wrapped portion
        using var temp = tensor.narrow(dim, n - shift, shift).clone();

        // Shift the main portion
        tensor.narrow(dim, shift, n - shift).copy_(tensor.narrow(dim, 0, n - shift));

        // Copy wrapped portion to start
        tensor.narrow(dim, 0, shift).copy_(temp);
    }

    /// <summary>
    /// Converts frequency in Hertz to Mel scale.
    /// </summary>
    /// <param name="hz">The frequency in Hertz</param>
    /// <returns>The frequency in Mel scale</returns>
    public static Tensor HertzToMel(this Tensor hz)
    {
        return 2595.0f * torch.log10(1.0f + hz / 700.0f);
    }

    /// <summary>
    /// Converts frequency in Mel scale to Hertz.
    /// </summary>
    /// <param name="mel">The frequency in Mel scale</param>
    /// <returns>The frequency in Hertz</returns>
    public static Tensor MelToHertz(this Tensor mel)
    {
        return 700.0f * (torch.pow(10.0f, mel / 2595.0f) - 1.0f);
    }

    public static void WriteTensorToFile(this Tensor tensor, string filePath, int precision = 30, bool append = false, int? count = 200)
    {
        // Convert the tensor to an array
        var tensorArray = tensor.clone().cpu().detach().reshape(-1).to(torch.float32).data<float>().ToArray();

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