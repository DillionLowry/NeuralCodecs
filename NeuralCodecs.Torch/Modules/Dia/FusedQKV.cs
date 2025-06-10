using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Fused QKV projection layer for efficient attention computation.
/// </summary>
public class FusedQKV : Module<Tensor, (Tensor q, Tensor k, Tensor v)>
{
    private readonly int _numQHeads;
    private readonly int _qHeadDim;
    private readonly int _numKvHeads;
    private readonly int _kvHeadDim;
    private readonly int _qOutputDim;
    private readonly int _kvOutputDim;
    private readonly Linear _linear;

    /// <summary>
    /// Creates a new FusedQKV layer.
    /// </summary>
    /// <param name="inFeatures">Input feature dimension</param>
    /// <param name="outFeatures">Total output feature dimension (q_dim + k_dim + v_dim)</param>
    /// <param name="bias">Whether to use bias</param>
    /// <param name="numQHeads">Number of query heads</param>
    /// <param name="qHeadDim">Query head dimension</param>
    /// <param name="numKvHeads">Number of key/value heads</param>
    /// <param name="kvHeadDim">Key/value head dimension</param>
    public FusedQKV(
        int inFeatures,
        int outFeatures,
        bool bias = false,
        int numQHeads = 1,
        int qHeadDim = 1,
        int numKvHeads = 1,
        int kvHeadDim = 1)
        : base(nameof(FusedQKV))
    {
        _numQHeads = numQHeads;
        _qHeadDim = qHeadDim;
        _numKvHeads = numKvHeads;
        _kvHeadDim = kvHeadDim;
        _qOutputDim = numQHeads * qHeadDim;
        _kvOutputDim = numKvHeads * kvHeadDim;

        _linear = Linear(inFeatures, outFeatures, hasBias: bias);
        RegisterComponents();
    }

    /// <summary>
    /// Forward pass that splits output into Q, K, V tensors.
    /// </summary>
    /// <param name="inputs">Input tensor [B, T, D]</param>
    /// <returns>Tuple of (Q, K, V) tensors</returns>
    public override (Tensor q, Tensor k, Tensor v) forward(Tensor inputs)
    {
        using var scope = NewDisposeScope();

        var x = _linear.forward(inputs);

        // Split into Q, K, V
        var splits = x.split(new long[] { _qOutputDim, _kvOutputDim, _kvOutputDim }, dim: -1);
        var q = splits[0];
        var k = splits[1];
        var v = splits[2];

        // Reshape to include head dimensions
        var qShape = q.shape[..^1].Append(_numQHeads).Append(_qHeadDim).ToArray();
        var kShape = k.shape[..^1].Append(_numKvHeads).Append(_kvHeadDim).ToArray();
        var vShape = v.shape[..^1].Append(_numKvHeads).Append(_kvHeadDim).ToArray();

        var qr = q.reshape(qShape);
        var kr = k.reshape(kShape);
        var vr = v.reshape(vShape);

        return (qr.MoveToOuterDisposeScope(), kr.MoveToOuterDisposeScope(), vr.MoveToOuterDisposeScope());
    }

    /// <summary>
    /// Gets the underlying linear layer for weight access.
    /// </summary>
    public Linear Linear => _linear;

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _linear?.Dispose();
        }

        base.Dispose(disposing);
    }
}