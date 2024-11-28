using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Modules.SNAC;

/// <summary>
/// Implements Rotary Position Embedding (RoPE) operations for transformer models.
/// </summary>
public static class RotaryEmbedding
{
    /// <summary>
    /// Performs rotation operation on input tensor by splitting it into two halves
    /// and creating a new tensor with specific rotation pattern.
    /// </summary>
    /// <param name="x">Input tensor to rotate</param>
    /// <returns>Rotated tensor where second half is negated and swapped with first half</returns>
    public static Tensor RotateHalf(Tensor x)
    {
        using var scope = NewDisposeScope();
        var lastDim = x.size(-1);

        var firstHalf = x.slice(-1, 0, lastDim / 2, 1);
        var secondHalf = x.slice(-1, lastDim / 2, lastDim, 1);

        // Concatenate -secondHalf and firstHalf along the last dimension
        return cat(new[] { secondHalf.neg(), firstHalf }, dim: -1).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Applies rotary positional embeddings to query and key tensors used in attention mechanisms.
    /// This implementation includes scaling factors for better numerical stability.
    /// </summary>
    /// <param name="q">Query tensor from attention mechanism</param>
    /// <param name="k">Key tensor from attention mechanism</param>
    /// <param name="freqs">Frequency tensor for positional encoding</param>
    /// <param name="scale">Scaling tensor for numerical stability</param>
    /// <returns>
    /// Tuple containing:
    /// - Modified query tensor with rotary embeddings applied
    /// - Modified key tensor with rotary embeddings applied
    /// </returns>
    /// <remarks>
    /// The method applies the RoPE formula:
    /// q = (q * cos(freq) * scale) + (rotate_half(q) * sin(freq) * scale)
    /// k = (k * cos(freq) * inv_scale) + (rotate_half(k) * sin(freq) * inv_scale)
    /// </remarks>
    public static (Tensor q, Tensor k) ApplyRotaryPosEmb(
        Tensor q, Tensor k, Tensor freqs, Tensor scale)
    {
        using var scope = NewDisposeScope();

        var qLength = q.size(-2);
        var qFreqs = freqs.slice(dim: -2, start: -qLength, finish: freqs.size(-2), step: 1);

        var inverseScale = scale.reciprocal();

        if (scale.dim() == 2)
        {
            scale = scale.slice(dim: 0, start: -qLength, finish: scale.size(0), step: 1);
        }

        q = q.mul(qFreqs.cos()).mul(scale)
            .add(RotateHalf(q).mul(qFreqs.sin()).mul(scale));

        k = k.mul(freqs.cos()).mul(inverseScale)
            .add(RotateHalf(k).mul(freqs.sin()).mul(inverseScale));

        return (q.MoveToOuterDisposeScope(), k.MoveToOuterDisposeScope());
    }
}