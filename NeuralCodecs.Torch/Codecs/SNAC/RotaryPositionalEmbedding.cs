using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Codecs.SNAC;

public partial class SNAC
{
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
            var newShape = x.shape.ToList();

            newShape[^1] = 2;
            newShape.Add(lastDim / 2);

            // Get the two halves
            var x1 = x.select(dim: -2, index: 0);
            var x2 = x.select(dim: -2, index: 1);

            // Concatenate -x2 and x1 along the last dimension
            return cat(new[] { x2.neg(), x1 }, dim: -1).MoveToOuterDisposeScope();
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

            var qLen = q.size(-2);
            var qFreqs = freqs.slice(dim: -2, start: -qLen, finish: freqs.size(-2), step: 1);
            var invScale = scale.reciprocal();

            if (scale.dim() == 2)
            {
                scale = scale.slice(dim: 0, start: -qLen, finish: scale.size(0), step: 1);
            }

            q = q.mul(qFreqs.cos()).mul(scale)
                .add(RotateHalf(q).mul(qFreqs.sin()).mul(scale));

            k = k.mul(freqs.cos()).mul(invScale)
                .add(RotateHalf(k).mul(freqs.sin()).mul(invScale));

            return (q.MoveToOuterDisposeScope(), k.MoveToOuterDisposeScope());
        }
    }
}