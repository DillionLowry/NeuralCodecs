using NeuralCodecs.Torch.Config.Dia;
using NeuralCodecs.Torch.Utils;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Self-attention module with support for Grouped Query Attention (GQA) and fused QKV.
/// </summary>
public class SelfAttention : Module<Tensor, Tensor, int, Tensor>
{
    private readonly int _numQueryHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _outputDim;
    private readonly int _projectedQueryDim;
    private readonly int _numGqaGroups;
    private readonly int _kvEmbedDim;
    private readonly int _qEmbedDim;
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
    /// Fused QKV layer (optional, used when patched).
    /// </summary>
    public FusedQKV? _qkv;

    /// <summary>
    /// Whether fused QKV is enabled.
    /// </summary>
    public bool IsFusedQkv { get; private set; }

    /// <summary>
    /// Creates a new SelfAttention module.
    /// </summary>
    /// <param name="config">Model configuration</param>
    /// <param name="qEmbedDim">Query embedding dimension</param>
    /// <param name="kvEmbedDim">Key/value embedding dimension</param>
    /// <param name="numQueryHeads">Number of query heads</param>
    /// <param name="numKvHeads">Number of key/value heads</param>
    /// <param name="headDim">Dimension per head</param>
    /// <param name="computeDtype">Computation data type</param>
    /// <param name="outEmbedDim">Output embedding dimension (if different from query)</param>
    public SelfAttention(
        DiaConfig config,
        int qEmbedDim,
        int kvEmbedDim,
        int numQueryHeads,
        int numKvHeads,
        int headDim,
        ScalarType computeDtype,
        int? outEmbedDim = null)
        : base(nameof(SelfAttention))
    {
        _numQueryHeads = numQueryHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _outputDim = outEmbedDim ?? qEmbedDim;
        _projectedQueryDim = numQueryHeads * headDim;
        _computeDtype = computeDtype;
        _kvEmbedDim = kvEmbedDim;
        _qEmbedDim = qEmbedDim;

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

        // Create Rotary embeddings after registering components
        _rotaryEmb = new RotaryEmbedding(
            embeddingDims: headDim,
            minTimescale: config.Model.RopeMinTimescale,
            maxTimescale: config.Model.RopeMaxTimescale,
            computeDtype: computeDtype
        );

        IsFusedQkv = false;
    }

    /// <summary>
    /// Gets the linear weight from a DenseGeneral layer for fused QKV conversion.
    /// </summary>
    private Tensor GetLinearWeight(DenseGeneral dense)
    {
        using var scope = NewDisposeScope();

        var wDg = dense.weight;

        long outFeatures = 1;
        long inputFeatures = 1;

        foreach (var dim in dense.OutFeatures)
        {
            outFeatures *= dim;
        }

        foreach (var dim in dense.InShapes)
        {
            inputFeatures *= dim;
        }

        var wDgReshapedForLinearT = wDg.reshape(inputFeatures, outFeatures);
        var linearWeight = wDgReshapedForLinearT.transpose(0, 1).contiguous();

        return linearWeight.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Patches the module to use fused QKV for better performance.
    /// </summary>
    public void PatchFusedQkv()
    {
        using var scope = NewDisposeScope();

        var qProjWeight = GetLinearWeight(q_proj);
        var kProjWeight = GetLinearWeight(k_proj);
        var vProjWeight = GetLinearWeight(v_proj);

        _qkv = new FusedQKV(
            _kvEmbedDim,
            (_numQueryHeads * _headDim) + (2 * _numKvHeads * _headDim),
            bias: false,
            numQHeads: _numQueryHeads,
            qHeadDim: _headDim,
            numKvHeads: _numKvHeads,
            kvHeadDim: _headDim
        );

        using (torch.no_grad())
        {
            var concatenatedWeight = cat(new[] { qProjWeight, kProjWeight, vProjWeight }, dim: 0);
            _qkv.Linear.weight.copy_(concatenatedWeight.DetachFromDisposeScope());
        }
        IsFusedQkv = true;
    }

    /// <summary>
    /// Repeats the key-value tensor along the head dimension by the specified number of repetitions.
    /// </summary>
    public static Tensor RepeatKV(Tensor x, int nRep)
    {
        if (nRep == 1)
        {
            return x;
        }

        var batchSize = x.shape[0];
        var nKVHeads = x.shape[1];
        var seqLen = x.shape[2];
        var headDim = x.shape[3];

        return WrappedTensorDisposeScope(() =>
        {
            var expanded = x.unsqueeze(2)
                           .expand(batchSize, nKVHeads, nRep, seqLen, headDim);
            return expanded.reshape(batchSize, nKVHeads * nRep, seqLen, headDim);
        });
    }

    /// <summary>
    /// Performs forward pass for self-attention.
    /// </summary>
    /// <param name="x">Input tensor [B, T, D]</param>
    /// <param name="qPositions">Query positions [B, T]</param>
    /// <param name="currentIndex">Current decoding step</param>
    /// <returns>Attention output</returns>
    public override Tensor forward(Tensor x, Tensor qPositions, int currentIndex)
        => forward(x, qPositions, null, null, null, false, false, currentIndex);

    /// <summary>
    /// Performs self-attention with optional KV caching and rotary positional embeddings.
    /// </summary>
    /// <param name="x">Input tensor [batch_size, sequence_length, embedding_dim].</param>
    /// <param name="qPositions">Query positions tensor.</param>
    /// <param name="kvPositions">Key-value positions tensor. Defaults to <paramref name="qPositions"/> if null.</param>
    /// <param name="attnMask">Optional attention mask tensor.</param>
    /// <param name="cache">Optional KV cache for autoregressive decoding.</param>
    /// <param name="prefill">Whether to prefill the KV cache.</param>
    /// <param name="isCausal">Whether to apply causal masking.</param>
    /// <param name="currentIdx">Current sequence index for cache updates (required when cache is used without prefill).</param>
    /// <returns>Attention output tensor with the same dtype as input.</returns>
    public Tensor forward(
            Tensor x,
            Tensor qPositions,
            Tensor? kvPositions = null,
            Tensor? attnMask = null,
            KVCache? cache = null,
            bool prefill = false,
            bool isCausal = false,
            int? currentIdx = null)
    {
        kvPositions ??= qPositions; // keeping for parity with python
        var originalDtype = x.dtype;
        using var scope = NewDisposeScope();

        Tensor xqBxTxNxH, xkBxSxKxH, xvBxSxKxH;

        if (IsFusedQkv)
        {
            var (qf, kf, vf) = _qkv!.forward(x);
            xqBxTxNxH = qf;
            xkBxSxKxH = kf;
            xvBxSxKxH = vf;
        }
        else
        {
            xqBxTxNxH = q_proj.forward(x);
            xkBxSxKxH = k_proj.forward(x);
            xvBxSxKxH = v_proj.forward(x);
        }

        var position = qPositions.unsqueeze(-1).unsqueeze(-1);
        var sinusoidInp = position / _rotaryEmb.Timescale.to(position.device);
        var sin = torch.sin(sinusoidInp);
        var cos = torch.cos(sinusoidInp);

        xqBxTxNxH = _rotaryEmb.ApplyRope(xqBxTxNxH, sin, cos);
        xkBxSxKxH = _rotaryEmb.ApplyRope(xkBxSxKxH, sin, cos);

        var xqBxNxTxH = xqBxTxNxH.transpose(1, 2);
        var xkBxKxSxH = xkBxSxKxH.transpose(1, 2);
        var xvBxKxSxH = xvBxSxKxH.transpose(1, 2);

        Tensor attnK, attnV;

        if (cache is null)
        {
            attnK = xkBxKxSxH;
            attnV = xvBxKxSxH;
        }
        else if (prefill)
        {
            attnK = xkBxKxSxH;
            attnV = xvBxKxSxH;
            cache.Prefill(attnK, attnV);
        }
        else
        {
            (attnK, attnV) = cache.Update(xkBxKxSxH, xvBxKxSxH, currentIdx);
        }

        Tensor attnOutput = AttentionUtils.ScaledDotProductAttention(
            xqBxNxTxH,
            attnK,
            attnV,
            attentionMask: isCausal ? null : attnMask,
            isCausal: isCausal,
            scale: 1.0f,
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
            _qkv?.Dispose();
        }

        base.Dispose(disposing);
    }
}