using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Utils;

public static class TensorExtensions
{
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
}