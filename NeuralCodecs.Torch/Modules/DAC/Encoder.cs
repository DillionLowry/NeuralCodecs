using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Encoder module that transforms input audio signals into latent representations.
/// </summary>
public class Encoder : Module<Tensor, Tensor>, IDisposable
{
    private readonly int _encoderDim;
    private readonly Sequential block;

    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the Encoder class.
    /// </summary>
    /// <param name="dModel">The initial dimension of the model (default: 64)</param>
    /// <param name="strides">Array of stride values for each encoder block (default: [2,4,8,8])</param>
    /// <param name="dLatent">The dimension of the latent space (default: 64)</param>
    public Encoder(
        int dModel = 64,
        int[] strides = null,
        int dLatent = 64) : base("Encoder")
    {
        strides ??= [2, 4, 8, 8];

        var layers = new List<Module<Tensor, Tensor>>
        {
            // First convolution layer
            new WNConv1d(1, dModel, kernelSize: 7, padding: 3)
        };

        // Add encoder blocks that double channels and downsample
        foreach (var stride in strides)
        {
            dModel *= 2;
            layers.Add(new EncoderBlock(dModel, stride: stride));
        }

        // Add final layers
        layers.AddRange(new Module<Tensor, Tensor>[]
        {
            new Snake1d(dModel),
            new WNConv1d(dModel, dLatent, kernelSize: 3, padding: 1)
        });

        block = Sequential(layers);
        _encoderDim = dModel;
        RegisterComponents();
    }

    /// <summary>
    /// Performs the forward pass of the encoder.
    /// </summary>
    /// <param name="x">Input tensor</param>
    /// <returns>Encoded representation of the input</returns>
    public override Tensor forward(Tensor x) => block.forward(x);

    public new void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                block?.Dispose();
            }
            base.Dispose(disposing);
            _disposed = true;
        }
    }
}