using NeuralCodecs.Torch.Modules;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Represents a decoder block in the DAC (Dilated Audio Codec) architecture.
/// </summary>
public class DecoderBlock : Module<Tensor, Tensor>, IDisposable
{
    private readonly Sequential block;
    private bool disposedValue;

    /// <summary>
    /// Initializes a new instance of the DecoderBlock class.
    /// </summary>
    /// <param name="inputDim">The input dimension size. Default is 16.</param>
    /// <param name="outputDim">The output dimension size. Default is 8.</param>
    /// <param name="stride">The stride value for transposed convolution. Default is 1.</param>
    public DecoderBlock(int inputDim = 16, int outputDim = 8, int stride = 1)
        : base($"DecoderBlock_{inputDim}_{outputDim}")
    {
        block = Sequential(
            new Snake1d(inputDim),
            new WNConvTranspose1d(
                inputDim,
                outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: (int)Math.Ceiling(stride / 2.0)
            ),
            new ResidualUnit(outputDim, dilation: 1),
            new ResidualUnit(outputDim, dilation: 3),
            new ResidualUnit(outputDim, dilation: 9)
        );
        RegisterComponents();
    }

    /// <summary>
    /// Performs the forward pass of the decoder block.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The processed output tensor.</returns>
    public override Tensor forward(Tensor x) => block.forward(x);

    /// <summary>
    /// Disposes the resources used by this decoder block.
    /// </summary>
    /// <param name="disposing">True to dispose managed resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                block.Dispose();
            }
            disposedValue = true;
        }
    }

    /// <summary>
    /// Disposes the resources used by this decoder block.
    /// </summary>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}