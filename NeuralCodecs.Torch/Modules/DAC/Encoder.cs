using NeuralCodecs.Diagnostics;
using System.Text;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Encoder module that transforms input audio signals into latent representations.
/// </summary>
[DiagnosticsEnabled("Encoder")]
public class Encoder : Module<Tensor, Tensor>, IDisposable
{
    private readonly int _encoderDim;
    private readonly Sequential block;
    private IDiagnosticsContext _diagnostics;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the Encoder class.
    /// </summary>
    /// <param name="dModel">The initial dimension of the model (default: 64)</param>
    /// <param name="strides">Array of stride values for each encoder block (default: [2,4,8,8])</param>
    /// <param name="dLatent">The dimension of the latent space (default: 64)</param>
    /// <param name="diagnostics">Optional diagnostics context</param>
    public Encoder(
        int dModel = 64,
        int[] strides = null,
        int dLatent = 64,
        IDiagnosticsContext diagnostics = null) : base("Encoder")
    {
        _diagnostics = diagnostics ?? DiagnosticsFactory.Create(enabled: false);
        strides ??= [2, 4, 8, 8];

        _diagnostics.LogMessage("Encoder",
            $"Initializing with dModel={dModel}, strides=[{string.Join(",", strides)}]");

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
            _diagnostics.LogMessage("Encoder",
                   $"Added EncoderBlock: outputDim={dModel}, stride={stride}");
        }

        _diagnostics.LogMessage("Encoder", $"Encoder - Creating Snake1d with input_dim: {dModel}");
        _diagnostics.LogMessage("Encoder", $"Encoder - Creating WNConv1d with input_dim: {dModel}, outputDim: {dLatent}, kernel_size: 3, padding: 1");

        // Add final layers
        layers.AddRange(new Module<Tensor, Tensor>[]
        {
            new Snake1d(dModel),
            new WNConv1d(dModel, dLatent, kernelSize: 3, padding: 1)
        });

        block = Sequential(layers);
        _encoderDim = dModel;
        RegisterComponents();

        if (_diagnostics.IsEnabled)
        {
            var structure = new StringBuilder();
            structure.AppendLine("Encoder Structure:");
            foreach (var layer in layers)
            {
                structure.AppendLine($"  - {layer.GetType().Name}");
            }
            _diagnostics.LogMessage("Encoder", structure.ToString());
        }
    }

    /// <summary>
    /// Performs the forward pass of the encoder.
    /// </summary>
    /// <param name="x">Input tensor</param>
    /// <returns>Encoded representation of the input</returns>
    public override Tensor forward(Tensor x)
    {
        using var scope = _diagnostics.TrackScope("Encoder", x);

        _diagnostics.LogTensor("Encoder", "input", x);
        var output = block.forward(x);

        _diagnostics.LogTensor("Encoder", "output", output);
        _diagnostics.DetectAnomalies("Encoder", output);
        return output;
    }

    /// <summary>
    /// Adds a diagnostics context to the encoder.
    /// </summary>
    /// <param name="diagnosticsContext">The diagnostics context to add</param>
    public void AddDiagnostics(IDiagnosticsContext diagnosticsContext)
    {
        _diagnostics = diagnosticsContext;
    }

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
                _diagnostics?.Dispose();
            }
            base.Dispose(disposing);
            _disposed = true;
        }
    }
}