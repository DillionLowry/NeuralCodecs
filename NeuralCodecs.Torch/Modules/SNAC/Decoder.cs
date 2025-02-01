using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.SNAC;
/// <summary>
/// Implements a hierarchical decoder for audio signal processing that progressively
/// upsamples and reconstructs the signal through multiple decoder blocks.
/// </summary>
/// <remarks>
/// SNAC adds noise injection, depthwise separable convolutions, and optional attention
/// window to the original DAC implementation.
/// </remarks>
public class Decoder : Module<Tensor, Tensor>
{
    /// <summary>
    /// Sequential container for all decoder operations
    /// </summary>
    private readonly Sequential model;

    /// <summary>
    /// Initializes a new instance of the Decoder
    /// </summary>
    /// <param name="inputChannel">Number of input channels</param>
    /// <param name="channels">Number of input channels</param>
    /// <param name="rates">Array of upsampling rates for each decoder block</param>
    /// <param name="noise">Whether to include noise injection in decoder blocks</param>
    /// <param name="depthwise">Whether to use depthwise separable convolutions</param>
    /// <param name="attnWindowSize">Size of local attention window (optional)</param>
    /// <param name="dOut">Output dimension (defaults to 1)</param>
    public Decoder(
        int inputChannel,
        int channels,
        int[] rates,
        bool noise = false,
        bool depthwise = false,
        int? attnWindowSize = 32,
        int dOut = 1) : base("Decoder")
    {
        var layers = new List<Module<Tensor, Tensor>>();

        if (depthwise)
        {
            layers.AddRange(new Module<Tensor, Tensor>[] {
                    new WNConv1d(inputChannel, inputChannel, kernelSize: 7, padding: 3, groups: inputChannel),
                    new WNConv1d(inputChannel, channels, kernelSize: 1)
                });
        }
        else
        {
            layers.Add(new WNConv1d(inputChannel, channels, kernelSize: 7, padding: 3));
        }

        if (attnWindowSize.HasValue)
        {
            layers.Add(new LocalMHA(dim: channels, windowSize: attnWindowSize.Value));
        }

        int outputDim = 1;
        for (int i = 0; i < rates.Length; i++)
        {
            int inputDim = channels / (1 << i);
            outputDim = channels / (1 << i + 1);
            int groups = depthwise ? outputDim : 1;

            layers.Add(new DecoderBlock(inputDim, outputDim, rates[i], noise, groups: groups));
        }

        layers.AddRange(new Module<Tensor, Tensor>[] {
            new Snake1d(outputDim),
            new WNConv1d(outputDim, dOut, kernelSize: 7, padding: 3),
            Tanh()
        });

        model = Sequential(layers);
        RegisterComponents();
    }

    /// <summary>
    /// Performs forward pass through the decoder
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, channels, time)</param>
    /// <returns>
    /// Reconstructed audio signal
    /// </returns>
    public override Tensor forward(Tensor x) => model.forward(x);

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            model?.Dispose();
        }
        base.Dispose(disposing);
    }
}
