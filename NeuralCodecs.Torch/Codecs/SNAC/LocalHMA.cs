using NeuralCodecs.Torch.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Codecs.SNAC;

public partial class SNAC
{
    /// <summary>
    /// Implements Local Multi-Head Attention with windowed attention patterns.
    /// Processes sequences by dividing them into fixed-size windows and applying
    /// attention mechanisms within each window independently.
    /// </summary>
    public class LocalMHA : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Layer normalization applied before attention computation
        /// </summary>
        private readonly LayerNorm norm;

        /// <summary>
        /// Number of attention heads
        /// </summary>
        private readonly int heads;

        /// <summary>
        /// Size of local attention window
        /// </summary>
        private readonly int windowSize;

        /// <summary>
        /// Linear projection for Query, Key, and Value generation
        /// </summary>
        private readonly Linear toQkv;

        /// <summary>
        /// Linear projection for output transformation
        /// </summary>
        private readonly Linear toOut;

        /// <summary>
        /// Relative positional embeddings using sinusoidal patterns
        /// </summary>
        private readonly SinusoidalEmbedding relPos;

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
            norm = LayerNorm(dim);
            heads = dim / dimHead;
            this.windowSize = windowSize;

            // Ensure dim is divisible by heads
            if (dim % heads != 0)
                throw new ArgumentException($"dim {dim} must be divisible by num_heads {heads}");

            toQkv = Linear(dim, dim * 3, hasBias: false, dtype: float32);
            toOut = Linear(dim, dim, hasBias: false, dtype: float32);

            if (useRotaryPosEmb)
            {
                relPos = new SinusoidalEmbedding(dimHead, scaleBase: windowSize / 2);
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

            var (batchSize, channels, timeSteps) = x.GetDimensions();
            var residual = x;

            // Layer norm expects [..., channels]
            x = norm.forward(x.transpose(1, 2));

            var windows = timeSteps / windowSize;

            // Generate Q, K, V
            var qkv = toQkv.forward(x).chunk(3, dim: -1);
            var (q, k, v) = (qkv[0], qkv[1], qkv[2]);

            // Reshape to attention format
            // Python: q, k, v = map(lambda t: rearrange(t, "b (w n) (h d) -> b h w n d", w=windows, h=self.heads), (q, k, v))
            q = RearrangeQKV(q, windows);
            k = RearrangeQKV(k, windows);
            v = RearrangeQKV(v, windows);

            // Apply rotary embeddings if enabled
            if (relPos != null)
            {
                var (posEmb, scale) = relPos.forward(k);
                (q, k) = RotaryEmbedding.ApplyRotaryPosEmb(q, k, posEmb, scale);
            }

            // Scaled dot-product attention
            var attn = functional.scaled_dot_product_attention(q, k, v);

            // Reshape back
            // Python: out = rearrange(out, "b h w n d -> b (w n) (h d)")
            var output = RearrangeOutput(attn);

            // Project to output dimension
            output = toOut.forward(output);

            // Restore original shape and add residual
            return output.transpose(1, 2) + residual;
        }

        private Tensor RearrangeQKV(Tensor x, long windows)
        {
            var shape = x.shape;
            return x.reshape(shape[0], windows, -1, heads, -1)
                    .transpose(2, 3);
        }

        private Tensor RearrangeOutput(Tensor x)
        {
            var shape = x.shape;
            return x.transpose(2, 3)
                    .reshape(shape[0], -1, shape[1] * shape[-1]);
        }
    }
}