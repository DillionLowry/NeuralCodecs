using NeuralCodecs.Torch.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.SNAC;

/// <summary>
/// Implements Local Multi-Head Attention with windowed attention patterns.
/// Processes sequences by dividing them into fixed-size windows and applying
/// attention mechanisms within each window independently.
/// </summary>
public class LocalMHA : Module<Tensor, Tensor>
{
    private readonly int _numHeads;
    private readonly int _windowSize;

    /// <summary>
    /// Layer normalization applied before attention computation
    /// </summary>
    private readonly LayerNorm norm;

    /// <summary>
    /// Linear projection for Query, Key, and Value generation
    /// </summary>
    private readonly Linear to_qkv;

    /// <summary>
    /// Linear projection for output transformation
    /// </summary>
    private readonly Linear to_out;

    /// <summary>
    /// Relative positional embeddings using sinusoidal patterns
    /// </summary>
    private readonly SinusoidalEmbedding rel_pos;

    /// <summary>
    /// Initializes a new instance of Local Multi-Head Attention
    /// </summary>
    /// <param name="dim">Model dimension (must be divisible by number of heads)</param>
    /// <param name="windowSize">Size of local attention window</param>
    /// <param name="dimHead">Dimension of each attention head</param>
    /// <param name="useRotaryPosEmb">Whether to use rotary positional embeddings</param>
    /// <exception cref="ArgumentException">Thrown when dim is not divisible by number of heads</exception>
    public LocalMHA(
        int dim = 1024,
        int windowSize = 32,
        int dimHead = 64,
        bool useRotaryPosEmb = true) : base("LocalMHA")
    {
        if (dim % (dim / dimHead) != 0)
        {
            throw new ArgumentException($"dim {dim} must be divisible by num_heads {dim / dimHead}");
        }
        norm = LayerNorm(dim);
        _numHeads = dim / dimHead;
        _windowSize = windowSize;

        to_qkv = Linear(dim, dim * 3, hasBias: false, dtype: float32);
        to_out = Linear(dim, dim, hasBias: false, dtype: float32);

        if (useRotaryPosEmb)
        {
            rel_pos = new SinusoidalEmbedding(dimHead, scaleBase: windowSize / 2);
        }

        RegisterComponents();
    }

    /// <summary>
    /// Performs forward pass of local multi-head attention
    /// </summary>
    /// <param name="x">Input tensor of shape (batch, channels, time)</param>
    /// <returns>
    /// Output tensor of same shape with locally attended features
    /// </returns>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();

        var (_, _, timeSteps) = x.GetBCTDimensions();
        var residual = x;

        // Layer norm expects [..., channels]
        x = norm.forward(x.transpose(1, 2));
        var windows = timeSteps / _windowSize;

        // Generate Query, Key, and Value tensors
        var qkv = to_qkv.forward(x).chunk(3, dim: -1);
        var (q, k, v) = (qkv[0], qkv[1], qkv[2]);

        // Reshape to attention format (batch, heads, window, seq, dim)
        q = RearrangeQKV(q, windows);
        k = RearrangeQKV(k, windows);
        v = RearrangeQKV(v, windows);

        // Apply rotary embeddings if enabled
        if (rel_pos != null)
        {
            var (posEmb, scale) = rel_pos.forward(k);
            (q, k) = RotaryEmbedding.ApplyRotaryPosEmb(q, k, posEmb, scale);
        }

        var attn = functional.scaled_dot_product_attention(q, k, v);

        // (batch, heads, window, seq, dim) => (batch, window*seq, heads*dim)
        var output = RearrangeOutput(attn);

        // Project to output dimension
        output = to_out.forward(output);

        // Restore original shape and add residual
        return output.transpose(1, 2).add_(residual).MoveToOuterDisposeScope();
    }

    private Tensor RearrangeQKV(Tensor x, long windows)
    {
        // Reshape the tensor to (batch, windows, seqPerWindow, heads, dimPerHead)
        var output = x.reshape(x.shape[0], windows, x.shape[1] / windows, _numHeads, x.shape[2] / _numHeads);

        // Permute the dimensions to (batch, heads, windows, seqPerWindow, dimPerHead)
        return output.permute(0, 3, 1, 2, 4).MoveToOuterDisposeScope();
    }

    private Tensor RearrangeOutput(Tensor x)
    {
        var shape = x.shape;

        // Permute dimensions to (batch, window, seq, heads, dim)
        x = x.permute(0, 2, 3, 1, 4);

        // Reshape to (batch, window*seq, heads*dim)
        return x.reshape(shape[0], shape[2] * shape[3], shape[1] * shape[4]).MoveToOuterDisposeScope();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            norm?.Dispose();
            to_qkv?.Dispose();
            to_out?.Dispose();
            rel_pos?.Dispose();
        }
        base.Dispose(disposing);
    }
}