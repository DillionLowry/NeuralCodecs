using NeuralCodecs.Torch.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// Streaming Transformer Encoder with support for causal attention.
/// </summary>
public class StreamingTransformerEncoder : Module<Tensor, (Tensor output, List<Tensor> states, int offset)>
{
    private readonly float _maxPeriod;
    private readonly int _pastContext;
    private readonly ModuleList<StreamingTransformerEncoderLayer> layers;
    private readonly Module<Tensor, Tensor> norm_in;

    /// <summary>
    /// Initialize a streaming transformer encoder
    /// </summary>
    /// <param name="dimension">Model dimension</param>
    /// <param name="hiddenScale">Scale factor for feed-forward hidden dimension</param>
    /// <param name="numHeads">Number of attention heads</param>
    /// <param name="numLayers">Number of transformer layers</param>
    /// <param name="maxPeriod">Maximum period for positional encoding</param>
    /// <param name="pastContext">Maximum context size to consider</param>
    /// <param name="gelu">Whether to use GELU activation (otherwise ReLU)</param>
    /// <param name="normIn">Whether to normalize input</param>
    /// <param name="dropout">Dropout probability</param>
    public StreamingTransformerEncoder(
        int dimension,
        float hiddenScale = 4.0f,
        int numHeads = 8,
        int numLayers = 5,
        float maxPeriod = 10000.0f,
        int pastContext = 1000,
        bool gelu = true,
        bool normIn = true,
        float dropout = 0.0f) : base("StreamingTransformerEncoder")
    {
        ValidateParameters(dimension, numHeads, hiddenScale, numLayers, pastContext, dropout);

        _maxPeriod = maxPeriod;
        _pastContext = pastContext;

        var hiddenDim = (int)(dimension * hiddenScale);

        norm_in = normIn ? LayerNorm(dimension) : torch.nn.Identity() as Module<Tensor, Tensor>;

        var llist = new List<StreamingTransformerEncoderLayer>();
        for (int i = 0; i < numLayers; i++)
        {
            llist.Add(new StreamingTransformerEncoderLayer(
                dimension, numHeads, hiddenDim,
                activation: gelu ? Activations.GELU : Activations.ReLU,
                dropout: dropout));
        }
        layers = ModuleList(llist.ToArray());

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass with no previous state
    /// </summary>
    /// <param name="x">Input tensor [batch, time, dimension]</param>
    /// <returns>Tuple of (output, states, new_offset)</returns>
    public override (Tensor output, List<Tensor> states, int offset) forward(Tensor x)
    {
        return forward(x, null, 0);
    }

    /// <summary>
    /// Forward pass with streaming support
    /// </summary>
    /// <param name="x">Input tensor [batch, time, dimension]</param>
    /// <param name="states">Previous states from each layer, or null for initial state</param>
    /// <param name="offset">Current time offset</param>
    /// <returns>Tuple of (output, new_states, new_offset)</returns>
    public (Tensor output, List<Tensor> states, int offset) forward(
        Tensor x, List<Tensor>? states, int offset = 0)
    {
        using var scope = NewDisposeScope();
        var (_, timeSteps, channels) = x.GetBCTDimensions();

        // Initialize states if not provided
        if (states is null || states.Count == 0 || states.Any(x => x.IsInvalid))
        {
            states = new List<Tensor>();
            for (int i = 0; i < layers.Count; i++)
            {
                states.Add(zeros_like(x.slice(1, 0, 1, 1)));
            }
        }

        if (states.Count < layers.Count)
        {
            throw new ArgumentException($"Expected at least {layers.Count} states, got {states.Count}");
        }
        // Create positional encoding
        var positions = arange(timeSteps, device: x.device)
            .view(1, -1, 1)
            .add(offset);

        var posEmb = CreateSinEmbedding(positions, channels);

        // Apply input normalization and positional encoding
        var output = norm_in.forward(x);
        output = output.add(posEmb);

        var newStates = new List<Tensor>();

        // Process through transformer layers
        for (int i = 0; i < layers.Count; i++)
        {
            (output, var newLayerState) = layers[i].forward(x, states[i], _pastContext);

            using var combinedState = cat(new[] { states[i], newLayerState }, dim: 1);
            var trimmedState = combinedState
                .slice(1, Math.Max(0, combinedState.size(1) - _pastContext), combinedState.size(1), 1)
                .clone();

            newStates.Add(trimmedState.MoveToOuterDisposeScope());
        }

        return (output.MoveToOuterDisposeScope(), newStates, offset + timeSteps);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            norm_in?.Dispose();
            layers?.Dispose();
        }
        base.Dispose(disposing);
    }

    private static void ValidateParameters(
           int dimension, int numHeads, float hiddenScale, int numLayers, int pastContext, float dropout)
    {
        if (dimension <= 0)
        {
            throw new ArgumentException($"Dimension must be positive, got {dimension}");
        }

        if (dimension % numHeads != 0)
        {
            throw new ArgumentException(
                $"Dimension ({dimension}) must be divisible by number of heads ({numHeads})");
        }

        if (numHeads <= 0)
        {
            throw new ArgumentException($"Number of heads must be positive, got {numHeads}");
        }

        if (hiddenScale <= 0)
        {
            throw new ArgumentException($"Hidden scale must be positive, got {hiddenScale}");
        }

        if (numLayers <= 0)
        {
            throw new ArgumentException($"Number of layers must be positive, got {numLayers}");
        }

        if (pastContext <= 0)
        {
            throw new ArgumentException($"Past context must be positive, got {pastContext}");
        }

        if (dropout is < 0 or > 1)
        {
            throw new ArgumentException($"Dropout must be between 0 and 1, got {dropout}");
        }
    }

    private Tensor CreateSinEmbedding(Tensor positions, int dimension)
    {
        using var scope = NewDisposeScope();

        if (dimension % 2 != 0)
        {
            throw new ArgumentException($"Dimension must be even, got {dimension}");
        }

        // Create time embedding with shape [1, T, dimension]
        var halfDim = dimension / 2;
        var arange = torch.arange(halfDim, device: positions.device).view(1, 1, -1);
        var phase = positions.div(torch.pow(_maxPeriod, arange.div(halfDim - 1.0f)));

        return cat(new[] { cos(phase), sin(phase) }, dim: -1).MoveToOuterDisposeScope();
    }
}