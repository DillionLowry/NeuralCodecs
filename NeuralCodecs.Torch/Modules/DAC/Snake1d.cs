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
    /// Initializes Snake activation with learnable parameters
    /// </summary>
    /// <param name="channels">Number of input channels</param>
    public Snake1d(long channels) : base($"Snake_{channels}")
    {
        if (channels <= 0)
        {
            throw new ArgumentException("Channels must be positive", nameof(channels));
        }

        alpha = Parameter(ones(1, channels, 1, dtype: float32));
        RegisterComponents();
    }

    /// <summary>
    /// Performs forward pass of Snake activation: x + (1/α) * sin²(αx)
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, channels, time)</param>
    /// <returns>Activated tensor of same shape</returns>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var output = torch.where(alpha == 0, x, addcdiv(x, sin(alpha * x).pow_(2), alpha, 1));
        if (cuda_is_available())
        {
            cuda.synchronize();
        }
        return output.MoveToOuterDisposeScope();
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