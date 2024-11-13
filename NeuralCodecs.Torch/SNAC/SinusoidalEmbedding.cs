using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public partial class SNAC
{
    /// <summary>
    /// Implements sinusoidal positional embeddings with optional XPos scaling.
    /// This class generates position-dependent frequency patterns used in transformer models
    /// to encode sequence position information.
    /// </summary>
    public class SinusoidalEmbedding : Module<Tensor, (Tensor freqs, Tensor scale)>
    {
        /// <summary>
        /// Inverse frequency tensor calculated using logarithmic spacing
        /// </summary>
        private readonly Tensor invFreq;

        /// <summary>
        /// Optional base value for XPos scaling calculations
        /// </summary>
        private readonly int? scaleBase;

        /// <summary>
        /// Flag to enable XPos scaling mechanism
        /// </summary>
        private readonly bool useXpos;

        /// <summary>
        /// Scale tensor used for XPos calculations
        /// </summary>
        private readonly Tensor scale;

        /// <summary>
        /// Initializes a new instance of sinusoidal embeddings
        /// </summary>
        /// <param name="dim">Dimensionality of the embeddings</param>
        /// <param name="scaleBase">Base value for XPos scaling (required if useXpos is true)</param>
        /// <param name="useXpos">Whether to use XPos scaling mechanism</param>
        /// <exception cref="ArgumentException">Thrown when useXpos is true but scaleBase is not provided</exception>
        public SinusoidalEmbedding(int dim, int? scaleBase = null, bool useXpos = false)
            : base("SinusoidalEmbedding")
        {
            this.useXpos = useXpos;
            this.scaleBase = scaleBase;

            invFreq = arange(0, dim, 2).to(float32)
                .div(dim)
                .neg()
                .exp()
                .mul(Math.Log(10000));
            ConditionallyRegisterBuffer("inv_freq", invFreq);

            if (useXpos && !scaleBase.HasValue)
                throw new ArgumentException("scale_base must be defined if using xpos");

            scale = arange(0, dim, 2)
                .add(0.4f * dim)
                .div(1.4f * dim);
            ConditionallyRegisterBuffer("scale", scale, persistent: false);

            RegisterComponents();
        }

        /// <summary>
        /// Generates sinusoidal position embeddings for input sequences
        /// </summary>
        /// <param name="x">Input tensor to generate embeddings for</param>
        /// <returns>
        /// A tuple containing:
        /// - freqs: Tensor of sinusoidal frequencies
        /// - scale: Scaling factors (ones if useXpos is false, computed scales if true)
        /// </returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when useXpos is true but scaleBase is not set
        /// </exception>
        public override (Tensor freqs, Tensor scale) forward(Tensor x)
        {
            using var scope = NewDisposeScope();

            var seqLen = x.size(-2);
            var t = arange(seqLen, device: x.device).to(invFreq.dtype);

            var freqs = einsum("i,j->ij", t, invFreq);

            freqs = cat(new[] { freqs, freqs }, dim: -1);

            if (!useXpos)
                return (freqs.MoveToOuterDisposeScope(), ones(1, device: x.device));

            if (!scaleBase.HasValue)
            {
                throw new InvalidOperationException("scaleBase must have a value when useXpos is true.");
            }
            var power = t.sub(seqLen / 2).div(scaleBase.Value);

            var scaleValues = scale.pow(power.reshape(-1, 1));

            scaleValues = cat([scaleValues, scaleValues], dim: -1);

            return (freqs.MoveToOuterDisposeScope(), scaleValues.MoveToOuterDisposeScope());
        }
    }
}