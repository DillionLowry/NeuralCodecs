using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Implements a residual unit block commonly used in neural networks for audio processing.
/// This unit consists of dilated convolutions with weight normalization and Snake activations.
/// </summary>
public class ResidualUnit : Module<Tensor, Tensor>
{
    /// <summary>
    /// Sequential container for the residual block operations
    /// </summary>
    private readonly Sequential block;

    /// <summary>
    /// Initializes a new instance of the ResidualUnit class
    /// </summary>
    /// <param name="dim">Number of input/output channels</param>
    /// <param name="dilation">Dilation factor for the main convolution</param>
    public ResidualUnit(long dim = 16, long dilation = 1)
        : base($"ResUnit_{dim}_{dilation}")
    {
        var pad = (7 - 1) * dilation / 2;

        block = Sequential(
            new Snake1d(dim),
            new WNConv1d(dim, dim, kernelSize: 7, padding: pad, dilation: dilation),
            new Snake1d(dim),
            new WNConv1d(dim, dim, kernelSize: 1)
        );
        RegisterComponents();
    }

    /// <summary>
    /// Performs the forward pass of the residual unit
    /// </summary>
    /// <param name="x">Input tensor</param>
    /// <returns>
    /// Output tensor with residual connection added:
    /// output = input + block(input)
    /// </returns>
    /// <remarks>
    /// Handles padding adjustment to ensure input and output shapes match
    /// for the residual connection
    /// </remarks>
    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var y = block.forward(x);
        var pad = (int)(x.shape[^1] - y.shape[^1]) / 2;
        if (pad > 0)
        {
            x = x[.., .., pad..^pad];
        }
        return x.add(y).MoveToOuterDisposeScope();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            block?.Dispose();
        }
        base.Dispose(disposing);
    }
}