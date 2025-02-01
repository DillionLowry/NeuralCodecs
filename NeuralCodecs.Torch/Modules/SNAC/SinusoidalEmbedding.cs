using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.SNAC;

/// <summary>
/// Implements sinusoidal positional embeddings with optional XPos scaling.
/// This class generates position-dependent frequency patterns used in transformer models
/// to encode sequence position information.
/// </summary>
public class SinusoidalEmbedding : Module<Tensor, (Tensor freqs, Tensor scale)>
{
    private readonly int? _scaleBase;
    private readonly bool _useXpos;

    /// <summary>
    /// Scale tensor used for XPos calculations
    /// </summary>
    private readonly Tensor scale;

    /// <summary>
    /// Inverse frequency tensor calculated using logarithmic spacing
    /// </summary>
    private readonly Tensor inv_freq;

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
        if (useXpos && !scaleBase.HasValue)
            throw new ArgumentException("scale_base must be defined if using xpos");

        _useXpos = useXpos;
        _scaleBase = scaleBase;

        // Calculate inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        var indices = arange(0, dim, 2).to(float32);
        var power = indices.div(dim);
        inv_freq = ones_like(power).div(pow(10000, power));
        ConditionallyRegisterBuffer("inv_freq", inv_freq);

        // Calculate scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        scale = arange(0, dim, 2).add(0.4f * dim).div(1.4f * dim);
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
        var t = arange(seqLen, device: x.device).to(inv_freq.dtype);

        // Calculate position frequencies - shape will be [seqLen, dim/2]
        var freqs = einsum("i,j->ij", t, inv_freq);

        // Duplicate frequencies - shape will be [seqLen, dim]
        freqs = cat(new[] { freqs, freqs }, dim: -1);

        if (!_useXpos)
        {
            return (freqs.MoveToOuterDisposeScope(), ones(1, device: x.device).MoveToOuterDisposeScope());
        }

        if (!_scaleBase.HasValue)
        {
            throw new InvalidOperationException("scaleBase must have a value when useXpos is true.");
        }

        // Calculate power - shape [seq_len]
        var power = t.sub(seqLen / 2).div(_scaleBase.Value);

        // Reshape power to [seq_len, 1]
        power = power.reshape(-1, 1);

        // Take self.scale [dim/2] => [1, dim/2] and expand to [seq_len, dim/2]
        var expandedScale = scale.unsqueeze(0).expand(seqLen, -1);

        // Apply power to expanded scale => [seq_len, dim/2]
        var scaleValues = expandedScale.pow(power);

        // Duplicate scale values => [seq_len, dim]
        scaleValues = cat(new[] { scaleValues, scaleValues }, dim: -1);

        return (freqs.MoveToOuterDisposeScope(), scaleValues.MoveToOuterDisposeScope());
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            inv_freq?.Dispose();
            scale?.Dispose();
        }
        base.Dispose(disposing);
    }
}