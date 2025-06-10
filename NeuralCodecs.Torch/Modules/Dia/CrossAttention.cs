using NeuralCodecs.Torch.Config.Dia;
using NeuralCodecs.Torch.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Cross-attention module for decoder layers.
/// </summary>
public class CrossAttention : Module<Tensor, Tensor, Tensor, Tensor?, Tensor>
{
    private readonly int _numQueryHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _outputDim;
    private readonly int _projectedQueryDim;
    private readonly int _numGqaGroups;
    private readonly ScalarType _computeDtype;

    /// <summary>
    /// Query projection layer.
    /// </summary>
    public readonly DenseGeneral q_proj;

    /// <summary>
    /// Key projection layer.
    /// </summary>
    public readonly DenseGeneral k_proj;

    /// <summary>
    /// Value projection layer.
    /// </summary>
    public readonly DenseGeneral v_proj;

    /// <summary>
    /// Output projection layer.
    /// </summary>
    public readonly DenseGeneral o_proj;

    /// <summary>
    /// Rotary positional embedding layer.
    /// </summary>
    public readonly RotaryEmbedding _rotaryEmb;

    /// <summary>
    /// Creates a new CrossAttention module.
    /// </summary>
    /// <param name="config">Model configuration</param>
    /// <param name="qEmbedDim">Query embedding dimension</param>
    /// <param name="kvEmbedDim">Key/value embedding dimension</param>
    /// <param name="numQueryHeads">Number of query heads</param>
    /// <param name="numKvHeads">Number of key/value heads</param>
    /// <param name="headDim">Dimension per head</param>
    /// <param name="computeDtype">Computation data type</param>
    /// <param name="outEmbedDim">Output embedding dimension (if different from query)</param>
    public CrossAttention(
        DiaConfig config,
        int qEmbedDim,
        int kvEmbedDim,
        int numQueryHeads,
        int numKvHeads,
        int headDim,
        ScalarType computeDtype,
        int? outEmbedDim = null)
        : base(nameof(CrossAttention))
    {
        _numQueryHeads = numQueryHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _outputDim = outEmbedDim ?? qEmbedDim;
        _projectedQueryDim = numQueryHeads * headDim;
        _computeDtype = computeDtype;

        if (numQueryHeads % numKvHeads != 0)
        {
            throw new ArgumentException(
                $"num_query_heads ({numQueryHeads}) must be divisible by num_kv_heads ({numKvHeads})");
        }

        _numGqaGroups = numQueryHeads / numKvHeads;

        // Projection layers
        q_proj = new DenseGeneral(
            inShapes: new[] { qEmbedDim },
            outFeatures: new[] { numQueryHeads, headDim },
            axis: new[] { -1 },
            weightDtype: computeDtype
        );

        k_proj = new DenseGeneral(
            inShapes: new[] { kvEmbedDim },
            outFeatures: new[] { numKvHeads, headDim },
            axis: new[] { -1 },
            weightDtype: computeDtype
        );

        v_proj = new DenseGeneral(
            inShapes: new[] { kvEmbedDim },
            outFeatures: new[] { numKvHeads, headDim },
            axis: new[] { -1 },
            weightDtype: computeDtype
        );

        o_proj = new DenseGeneral(
            inShapes: new[] { numQueryHeads, headDim },
            outFeatures: new[] { _outputDim },
            axis: new[] { -2, -1 },
            weightDtype: computeDtype
        );

        RegisterComponents();

        // Create Rotary embeddings
        _rotaryEmb = new RotaryEmbedding(
            embeddingDims: headDim,
            minTimescale: config.Model.RopeMinTimescale,
            maxTimescale: config.Model.RopeMaxTimescale,
            computeDtype: computeDtype
        );
    }

    /// <summary>
    /// Forward pass for cross-attention.
    /// </summary>
    /// <param name="Xq">Query tensor [B, T, D]</param>
    /// <param name="qPositions">Query positions [B, T]</param>
    /// <param name="kvPositions">Key/value positions [B, S]</param>
    /// <param name="attnMask">Attention mask (optional)</param>
    /// <returns>Cross-attention output</returns>
    public override Tensor forward(Tensor Xq, Tensor qPositions, Tensor kvPositions, Tensor? attnMask)
        => forward(Xq, qPositions, kvPositions, attnMask, null);

    /// <summary>
    /// Performs cross-attention computation.
    /// </summary>
    /// <param name="Xq">Query tensor [B, T, D]</param>
    /// <param name="qPositions">Query positions [B, T]</param>
    /// <param name="kvPositions">Key/value positions [B, S]</param>
    /// <param name="attnMask">Attention mask (optional)</param>
    /// <param name="cache">KV cache containing precomputed keys and values</param>
    /// <param name="isCausal">Whether to apply causal masking.</param>
    /// <returns>Cross-attention output tensor</returns>
    public Tensor forward(
        Tensor Xq,
        Tensor qPositions,
        Tensor kvPositions,
        Tensor? attnMask = null,
        KVCache? cache = null,
        bool isCausal = false)
    {
        using var scope = NewDisposeScope();

        if (cache == null)
        {
            throw new ArgumentException(
                "Cache must be provided for cross-attention. Use PrecomputeCrossAttnCache() to create it.");
        }

        kvPositions ??= qPositions; // keeping for parity with python
        var originalDtype = Xq.dtype;

        // Project queries and apply RoPE
        var XqCompute = Xq.to(_computeDtype);
        var XqBxTxNxH = q_proj.forward(XqCompute);
        var XqBxNxTxH = _rotaryEmb.forward(XqBxTxNxH, position: qPositions).transpose(1, 2);

        // Get keys and values from cache
        var attnK = cache.K;
        var attnV = cache.V;

        Tensor attnOutput = AttentionUtils.ScaledDotProductAttention(
            XqBxNxTxH,
            attnK,
            attnV,
            attentionMask: isCausal ? null : attnMask,
            scale: 1.0f,
            isCausal: isCausal,
            enableGqa: _numGqaGroups > 1,
            gqaGroups: _numGqaGroups
        );
        var attnOutputTransposed = attnOutput.transpose(1, 2).contiguous();
        var output = o_proj.forward(attnOutputTransposed);

        return output.to(originalDtype).MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            q_proj?.Dispose();
            k_proj?.Dispose();
            v_proj?.Dispose();
            o_proj?.Dispose();
            _rotaryEmb?.Dispose();
        }

        base.Dispose(disposing);
    }
}