using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Rotary Position Embedding (RoPE) implementation.
/// </summary>
public class RotaryEmbedding : Module<Tensor, Tensor, Tensor>
{
    private readonly int _embeddingDims;
    private readonly int _minTimescale;
    private readonly int _maxTimescale;
    private readonly ScalarType _computeDtype;
    private readonly Tensor _timescale;

    /// <summary>
    /// Creates a new RotaryEmbedding layer.
    /// </summary>
    /// <param name="embeddingDims">Embedding dimension (must be even)</param>
    /// <param name="minTimescale">Minimum timescale</param>
    /// <param name="maxTimescale">Maximum timescale</param>
    /// <param name="dtype">Data type</param>
    public RotaryEmbedding(
        int embeddingDims,
        int minTimescale = 1,
        int maxTimescale = 10000,
        ScalarType computeDtype = ScalarType.Float32,
        Device? device = null)
        : base(nameof(RotaryEmbedding))
    {
        if (embeddingDims % 2 != 0)
        {
            throw new ArgumentException("Embedding dim must be even for RoPE.");
        }

        _embeddingDims = embeddingDims;
        _minTimescale = minTimescale;
        _maxTimescale = maxTimescale;
        _computeDtype = computeDtype;

        var halfEmbeddingDim = embeddingDims / 2;
        var fraction = 2.0f * arange(0, halfEmbeddingDim) / embeddingDims;

        // Calculate timescales
        _timescale = (_minTimescale * pow(
            _maxTimescale / _minTimescale,
            fraction
        )).to(float32);

        register_buffer("timescale", _timescale, persistent: false);
    }

    /// <summary>
    /// Applies rotary position embedding to the input tensor.
    /// </summary>
    /// <param name="inputs">Input tensor to apply RoPE to (shape: [B, T, H] or [B, T, N, H])</param>
    /// <param name="position">Position tensor (shape: [B, T])</param>
    /// <returns>Tensor with rotary position embedding applied (same shape as input)</returns>
    public override Tensor forward(Tensor inputs, Tensor position)
    {
        using var scope = NewDisposeScope();
        var positionExpanded = position.unsqueeze(-1).unsqueeze(-1);
        var sinusoidInp = positionExpanded.div(_timescale.to(position.device));

        var sin = sinusoidInp.sin();
        var cos = sinusoidInp.cos();

        var floatInputs = inputs.to(float32);
        var chunks = chunk(floatInputs, 2, dim: -1);
        var firstHalf = chunks[0];
        var secondHalf = chunks[1];

        var firstPart = firstHalf.mul(cos).sub(secondHalf.mul(sin));
        var secondPart = secondHalf.mul(cos).add(firstHalf.mul(sin));

        // Concatenate rotated parts back together
        var result = cat(
            new[] { firstPart.to(_computeDtype), secondPart.to(_computeDtype) },
            dim: -1
        );

        return result.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Applies rotary position embedding using precomputed sin and cos tensors.
    /// This is an optimized version that avoids recomputing sin/cos for each call.
    /// </summary>
    /// <param name="inputs">Input tensor to apply RoPE to</param>
    /// <param name="sin">Precomputed sin tensor</param>
    /// <param name="cos">Precomputed cos tensor</param>
    /// <returns>Tensor with rotary position embedding applied</returns>
    public Tensor ApplyRope(Tensor inputs, Tensor sin, Tensor cos)
    {
        using var scope = NewDisposeScope();
        var floatInputs = inputs.to(float32);
        var chunks = chunk(floatInputs, 2, dim: -1);
        var firstHalf = chunks[0];
        var secondHalf = chunks[1];

        var firstPart = (firstHalf * cos) - (secondHalf * sin);
        var secondPart = (secondHalf * cos) + (firstHalf * sin);

        var result = cat(
            new[] { firstPart.to(_computeDtype), secondPart.to(_computeDtype) },
            dim: -1
        );

        return result.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Gets the timescale tensor for external sin/cos computation.
    /// </summary>
    public Tensor Timescale => _timescale;

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _timescale?.Dispose();
        }

        base.Dispose(disposing);
    }
}