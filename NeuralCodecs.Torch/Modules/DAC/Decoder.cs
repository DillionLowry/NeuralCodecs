using NeuralCodecs.Torch.Modules;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Decoder module that transforms encoded latent representations back into audio waveforms
/// through a series of upsampling and residual blocks.
/// </summary>
public class Decoder : Module<Tensor, Tensor>, IDisposable
{
    private readonly Sequential model;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the Decoder class.
    /// </summary>
    /// <param name="inputChannel">Number of input channels</param>
    /// <param name="channels">Number of internal channels</param>
    /// <param name="rates">Array of upsampling rates for each decoder block</param>
    /// <param name="dOut">Number of output channels (default: 1)</param>
    public Decoder(
        int inputChannel,
        int channels,
        int[] rates,
        int dOut = 1) : base("Decoder")
    {
        var layers = new List<Module<Tensor, Tensor>>
        {
            new WNConv1d(inputChannel, channels, kernelSize: 7, padding: 3)
        };

        // Add upsampling + residual blocks
        int outputDim = 1;
        for (int i = 0; i < rates.Length; i++)
        {
            int inputDim = channels / (1 << i);
            outputDim = channels / (1 << (i + 1));
            layers.Add(new DecoderBlock(inputDim, outputDim, rates[i]));
        }

        layers.AddRange(new Module<Tensor, Tensor>[]
        {
            new Snake1d(outputDim),
            new WNConv1d(outputDim, dOut, kernelSize: 7, padding: 3),
            Tanh()
        });

        model = Sequential(layers);
        RegisterComponents();
    }

    /// <summary>
    /// Performs the forward pass of the decoder.
    /// </summary>
    /// <param name="x">Input tensor to decode</param>
    /// <returns>Decoded audio waveform tensor</returns>
    public override Tensor forward(Tensor x) => model.forward(x);

    /// <summary>
    /// Disposes the decoder and its resources.
    /// </summary>
    public new void Dispose()
    {
        if (!_disposed)
        {
            model?.Dispose();
            base.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}