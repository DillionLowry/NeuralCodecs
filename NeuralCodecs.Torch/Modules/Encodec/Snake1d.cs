using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Snake activation function for 1D inputs.
/// A periodic activation function with learnable frequency.
/// Ref: https://arxiv.org/abs/2006.08195
/// </summary>
public class Snake1d : Module<Tensor, Tensor>
{
    private const float Epsilon = 1e-9f;
    private readonly Parameter _alpha;
    private readonly long _channels;

    /// <summary>
    /// Initialize Snake activation function
    /// </summary>
    /// <param name="channels">Number of input channels</param>
    public Snake1d(long channels) : base("Snake")
    {
        ValidateParameters(channels);
        _channels = channels;

        // Initialize alpha parameter with ones
        _alpha = Parameter(ones(1, channels, 1, dtype: float32));
        RegisterComponents();
    }

    /// <summary>
    /// Forward pass applying Snake activation
    /// </summary>
    /// <param name="x">Input tensor [B, C, T]</param>
    /// <returns>Activated tensor</returns>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        ValidateInputShape(x);

        var shape = x.shape;
        var reshaped = x.reshape(shape[0], shape[1], -1);

        // Compute Snake activation:
        // y = x + (1/a) * sin^2(ax)
        // where a is the learnable frequency

        // Compute sin^2(ax)
        using var alphaMul = _alpha.mul(reshaped);
        using var sinResult = OptimizedSin(alphaMul);
        using var powered = sinResult.pow(2);

        // Compute 1/a term with epsilon for stability
        using var alphaEps = _alpha.add(Epsilon);
        using var reciprocal = alphaEps.reciprocal();

        // Combine terms
        using var mulResult = reciprocal.mul(powered);
        var output = reshaped.add(mulResult, alpha: 1.0f);

        // Restore original shape
        return output.reshape(shape).MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _alpha?.Dispose();
        }
        base.Dispose(disposing);
    }

    /// <summary>
    /// Optimized sin operation handling both CPU and CUDA tensors
    /// </summary>
    private static Tensor OptimizedSin(Tensor x)
    {
        if (!x.is_cuda)
        {
            return sin_(x); // In-place operation for CPU
        }

        using var scope = NewDisposeScope();
        var result = sin_(x.clone()); // Clone for CUDA to avoid modifying input
        cuda.synchronize(); // Ensure CUDA operations complete
        return result.MoveToOuterDisposeScope();
    }

    private static void ValidateParameters(long channels)
    {
        if (channels <= 0)
        {
            throw new ArgumentException(
                $"Number of channels must be positive, got {channels}",
                nameof(channels));
        }
    }

    private void ValidateInputShape(Tensor x)
    {
        if (x.dim() != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input tensor [B, C, T], got shape [{string.Join(", ", x.shape)}]");
        }

        var channels = x.size(1);
        if (channels != _channels)
        {
            throw new ArgumentException(
                $"Expected {_channels} channels, got {channels}");
        }
    }
}