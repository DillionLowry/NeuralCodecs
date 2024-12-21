using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Implements the Snake activation function for 1D inputs.
/// Snake is a learned periodic activation function that generalizes both ReLU and Sine.
/// </summary>
/// <remarks>
/// The transform applies the function:
/// x + sin²(αx)/(α + ε)
/// where α is a learnable parameter and ε is a small constant for numerical stability
/// </remarks>
public class Snake1d : Module<Tensor, Tensor>
{
    /// <summary>
    /// Learnable parameter that controls the frequency of the periodic component
    /// </summary>
    private readonly Parameter alpha;

    /// <summary>
    /// Small constant to prevent numerical instability
    /// </summary>
    private const float EPSILON = 1e-9f;

    /// <summary>
    /// Flag indicating GPU availability for optimized computation paths
    /// </summary>
    private static readonly bool UseGPU = cuda.is_available();

    /// <summary>
    /// Initializes Snake activation with learnable parameters
    /// </summary>
    /// <param name="channels">Number of input channels</param>
    public Snake1d(long channels) : base("Snake")
    {
        if (channels <= 0)
        {
            throw new ArgumentException("Channels must be positive", nameof(channels));
        }

        alpha = Parameter(ones(1, channels, 1, dtype: float32));
        RegisterComponents();
    }

    /// <summary>
    /// Performs an optimized sine calculation with precision and synchronization controls
    /// based on whether GPU or CPU computation is being used.
    /// </summary>
    /// <param name="x">Input tensor to compute sine of</param>
    /// <returns>
    /// Sine of input tensor with controlled precision and synchronization
    /// </returns>
    private static Tensor OptimizedSin(Tensor x)
    {
        if (UseGPU)
        {
            cuda.synchronize();
            return sin(x).to(float32, non_blocking: false);
        }
        else
        {
            var result = sin(x).to(float32);
            cuda.synchronize();
            return result;
        }
    }
    private static Tensor OptimizedSin(Tensor x)
    {
        GC.Collect();
        // For non-CUDA tensors just do simple conversion
        if (!x.is_cuda)
        {
            return torch.sin_(x);
        }

        // For CUDA tensors, synchronize after in-place operation
        torch.sin_(x);
        cuda.synchronize();
        return x;
    }
    /// <summary>
    /// Performs forward pass of Snake activation: x + (1/α) * sin²(αx)
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, channels, time)</param>
    /// <returns>Activated tensor of same shape</returns>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();

        var shape = x.shape;
        var reshaped = x.reshape(shape[0], shape[1], -1);

        // Follow exact torch graph operation order
        var alpha_mul = alpha.mul(reshaped);
        var sin_result = OptimizedSin(alpha_mul);
        var powered = sin_result.pow(2);

        var alpha_eps = alpha.add(EPSILON);
        var reciprocal = alpha_eps.reciprocal();
        var mul_result = reciprocal.mul(powered);

        var added = reshaped.add(mul_result, alpha: 1.0f);

        return added.reshape(shape).MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            alpha?.Dispose();
        }
        base.Dispose(disposing);
    }
}