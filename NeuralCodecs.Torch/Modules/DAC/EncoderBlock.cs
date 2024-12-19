using NeuralCodecs.Torch.Modules;
using NeuralCodecs.Torch.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Represents an encoder block in the DAC architecture that processes audio signals through
/// a sequence of residual units, snake activation, and convolution.
/// </summary>
public class EncoderBlock : Module<Tensor, Tensor>, IDisposable
{
    private readonly Sequential block;

    /// <summary>
    /// Initializes a new instance of the EncoderBlock class.
    /// </summary>
    /// <param name="dim">The dimensionality of the input features. Default is 16.</param>
    /// <param name="stride">The stride value for the convolution operation. Default is 1.</param>
    public EncoderBlock(int dim = 16, int stride = 1) : base($"EncoderBlock_{dim}_{stride}")
    {
        block = Sequential(
            new ResidualUnit(dim / 2, dilation: 1),
            new ResidualUnit(dim / 2, dilation: 3),
            new ResidualUnit(dim / 2, dilation: 9),
            new Snake1d(dim / 2),
            new WNConv1d(
                dim / 2,
                dim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: (int)Math.Ceiling(stride / 2.0)
            )
        );
        RegisterComponents();
    }

    /// <summary>
    /// Performs the forward pass of the encoder block.
    /// </summary>
    /// <param name="x">The input tensor to process.</param>
    /// <returns>The processed output tensor.</returns>
    public override Tensor forward(Tensor x)
    {
        var output = block.forward(x);
        TensorUtils.LogTensor(tensor: output, name: $"{name}_output");
        return output;

    }

    /// <summary>
    /// Disposes the resources used by the encoder block.
    /// </summary>
    public void Dispose()
    {
        block?.Dispose();
        GC.SuppressFinalize(this);
    }
}