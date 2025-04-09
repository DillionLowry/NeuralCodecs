using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.SNAC;

/// <summary>
/// Implements an encoder block for audio processing that progressively increases the receptive field
/// through dilated convolutions and downsamples the signal through strided convolution.
/// </summary>
/// <remarks> SNAC adds an optional input dimension parameter and groups to the original DAC implementation
/// </remarks>
public class EncoderBlock : Module<Tensor, Tensor>
{
    /// <summary>
    /// Sequential container for the encoder block operations
    /// </summary>
    private readonly Sequential block;

    /// <summary>
    /// Initializes a new instance of the EncoderBlock class
    /// </summary>
    /// <param name="outputDim">Output channel dimension</param>
    /// <param name="inputDim">Input channel dimension (defaults to outputDim/2)</param>
    /// <param name="stride">Stride factor for downsampling</param>
    /// <param name="groups">Number of groups for grouped convolutions</param>
    public EncoderBlock(int outputDim = 16, int? inputDim = null, int stride = 1, int groups = 1)
        : base($"EncoderBlock_{outputDim}")
    {
        inputDim = inputDim == null || inputDim == 0 ? outputDim / 2 : inputDim;

        block = Sequential(
            new ResidualUnit(inputDim.Value, dilation: 1, groups: groups),
            new ResidualUnit(inputDim.Value, dilation: 3, groups: groups),
            new ResidualUnit(inputDim.Value, dilation: 9, groups: groups),
            new Snake1d(inputDim.Value),
            new WNConv1d(
                inputDim.Value,
                outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: (int)Math.Ceiling(stride / 2.0)
            )
        );
        RegisterComponents();
    }

    /// <summary>
    /// Performs forward pass of the encoder block
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, channels, time)</param>
    /// <returns>
    /// Processed and downsampled tensor
    /// </returns>
    public override Tensor forward(Tensor x) => block.forward(x);

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            block?.Dispose();
        }
        base.Dispose(disposing);
    }
}