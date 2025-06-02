using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// Single layer of the streaming transformer encoder with self-attention and feed-forward networks
/// </summary>
public class StreamingTransformerEncoderLayer : Module<Tensor, (Tensor output, Tensor state)>
{
    private readonly float _dropout;
    private readonly Sequential _feedForward;
    private readonly bool _normFirst;
    private readonly Linear linear1;
    private readonly Linear linear2;
    private readonly LayerNorm norm1;
    private readonly LayerNorm norm2;
    private readonly MultiheadAttention self_attn;

    /// <summary>
    /// Initialize a streaming transformer encoder layer
    /// </summary>
    /// <param name="dimension">Hidden dimension size</param>
    /// <param name="numHeads">Number of attention heads</param>
    /// <param name="feedForwardDim">Feed-forward layer dimension</param>
    /// <param name="activation">Activation function (GELU or ReLU)</param>
    /// <param name="dropout">Dropout probability</param>
    /// <param name="normFirst">Whether to apply normalization before or after sublayers</param>
    public StreamingTransformerEncoderLayer(
        int dimension,
        int numHeads,
        int feedForwardDim,
        Activations? activation = null,
        float dropout = 0.0f,
        bool normFirst = false) : base("StreamingTransformerEncoderLayer")
    {
        ValidateParameters(dimension, numHeads, feedForwardDim, dropout);

        activation ??= Activations.GELU;
        _dropout = dropout;
        _normFirst = normFirst;

        norm1 = LayerNorm(dimension);
        norm2 = LayerNorm(dimension);
        self_attn = MultiheadAttention(dimension, numHeads, dropout: dropout);
        linear1 = Linear(dimension, feedForwardDim);
        linear2 = Linear(feedForwardDim, dimension);
        RegisterComponents();
        _feedForward = Sequential(
            linear1,
            activation == Activations.GELU ? GELU() : ReLU(),
            Dropout(dropout),
            linear2,
            Dropout(dropout));
    }

    /// <summary>
    /// Forward pass of the transformer encoder layer
    /// </summary>
    /// <param name="x">Input tensor [batch, time, dimension]</param>
    /// <returns>Output tensor and state for the next layer</returns>
    public override (Tensor output, Tensor state) forward(Tensor x)
    {
        if (x.dim() < 3)
        {
            // [time, dimension] -> [1, time, dimension]
            x = x.unsqueeze(0);
        }

        // Create a default pastState with proper dimensions
        var pastState = zeros(1, x.size(1), x.size(2), device: x.device);
        return forward(x, pastState, 1000);
    }

    /// <summary>
    /// Forward pass with streaming support
    /// </summary>
    /// <param name="x">Input tensor [batch, time, dimension]</param>
    /// <param name="pastState">Past state from previous calls [past_len, batch, dimension]</param>
    /// <param name="pastContext">Maximum context size to consider</param>
    /// <returns>Output tensor and new state for future calls</returns>
    public (Tensor output, Tensor state) forward(
        Tensor x, Tensor pastState, int pastContext)
    {
        using var scope = NewDisposeScope();
        if (pastState is null || pastState.IsInvalid)
        {
            pastState = zeros(1, x.size(1), x.size(2), device: x.device);
        }

        Tensor inputForState;
        Tensor output = x;

        if (_normFirst)
        {
            // norm -> attn -> + -> norm -> ff -> +
            var saInput = norm1.forward(x);

            output = output.add(ApplySelfAttention(saInput, pastState, pastContext));
            inputForState = saInput;

            output = output.add(ApplyFeedForward(norm2.forward(output)));
        }
        else
        {
            // attn -> norm -> + -> ff -> norm -> +
            inputForState = x;
            output = norm1.forward(output.add(ApplySelfAttention(inputForState, pastState, pastContext)));
            output = norm2.forward(output.add(ApplyFeedForward(output)));
        }
        return (output.MoveToOuterDisposeScope(), inputForState.MoveToOuterDisposeScope());
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            norm1?.Dispose();
            norm2?.Dispose();
            linear1?.Dispose();
            linear2?.Dispose();
            self_attn?.Dispose();
            _feedForward?.Dispose();
        }
        base.Dispose(disposing);
    }

    private static void ValidateParameters(
            int dimension, int numHeads, int feedForwardDim, float dropout)
    {
        if (dimension <= 0)
        {
            throw new ArgumentException($"Dimension must be positive, got {dimension}");
        }

        if (numHeads <= 0)
        {
            throw new ArgumentException($"Number of heads must be positive, got {numHeads}");
        }

        if (dimension % numHeads != 0)
        {
            throw new ArgumentException(
                $"Dimension ({dimension}) must be divisible by number of heads ({numHeads})");
        }

        if (feedForwardDim <= 0)
        {
            throw new ArgumentException($"Feed-forward dimension must be positive, got {feedForwardDim}");
        }

        if (dropout is < 0 or > 1)
        {
            throw new ArgumentException($"Dropout must be between 0 and 1, got {dropout}");
        }
    }

    private Tensor ApplyDropout(Tensor input, float dropoutRate, bool training)
    {
        if (training && dropoutRate > 0)
        {
            return nn.functional.dropout(input, dropoutRate, training);
        }
        return input;
    }

    private Tensor ApplyFeedForward(Tensor x)
    {
        return _feedForward.forward(x);
    }

    /// <summary>
    /// Apply self-attention with causal masking
    /// </summary>
    /// <param name="x">Input tensor</param>
    /// <param name="pastState">Past state for caching</param>
    /// <param name="pastContext">Maximum context length</param>
    /// <returns>Self-attention output</returns>
    private Tensor ApplySelfAttention(Tensor x, Tensor pastState, int pastContext)
    {
        using var scope = NewDisposeScope();

        // Prepare query, key, value tensors
        var queries = x;
        var keys = cat(new[] { pastState, x }, dim: 1);
        var values = keys;

        var T = x.size(1);  // Current sequence length
        var H = pastState.size(0);  // Past sequence length

        // Create causal attention mask:
        // - Each position in current sequence can attend to itself and past positions
        // - Up to pastContext steps back from current position

        // Create position indices
        var queriesPos = arange(H, T + H, device: x.device).reshape(-1, 1);
        var keysPos = arange(T + H, device: x.device).reshape(1, -1);

        // Compute position differences
        var delta = queriesPos.sub(keysPos);

        // Valid attention when delta E [0, pastContext]
        var validAccess = logical_and(
            delta.greater_equal(0),
            delta.less_equal(pastContext));

        // Create attention mask (True values are masked positions)
        var attnMask = logical_not(validAccess);

        // Apply self-attention with causal mask
        var (attnOutput, _) = self_attn.forward(
            queries,
            keys,
            values,
            key_padding_mask: null,
            need_weights: false,
            attn_mask: attnMask);

        return ApplyDropout(attnOutput, _dropout, training: training).MoveToOuterDisposeScope();
    }
}