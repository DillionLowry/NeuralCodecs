using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

public class RMSNorm : Module<Tensor, Tensor>
{
    public readonly Tensor weight;
    private readonly float _epsilon;
    private readonly long[] _normalizedShape;
    private readonly bool _elementwiseAffine;

    public RMSNorm(long normalizedShape, float eps = 1e-6f, bool elementwiseAffine = true, ScalarType? dtype = null, Device? device = null)
        : this(new long[] { normalizedShape }, eps, elementwiseAffine, dtype, device)
    {
    }

    /// <summary>
    /// Initializes a new instance of the RMSNorm class.
    /// </summary>
    /// <param name="normalizedShape">Feature dimension to normalize.</param>
    /// <param name="eps">Small constant for numerical stability.</param>
    /// <param name="elementwiseAffine">Whether to include a weight term (gamma).</param>
    /// <param name="dtype">Data type for parameters.</param>
    /// <param name="device">Device for the parameters.</param>
    public RMSNorm(
        long[] normalizedShape,
        float eps = 1e-6f,
        bool elementwiseAffine = true,
        ScalarType? dtype = null,
        Device? device = null) : base("RMSNorm")
    {
        _normalizedShape = normalizedShape;
        _epsilon = eps;
        _elementwiseAffine = elementwiseAffine;

        if (elementwiseAffine)
        {
            weight = Parameter(empty(normalizedShape, device: device, dtype: dtype));
        }

        ResetParameters();
    }

    private void ResetParameters()
    {
        if (_elementwiseAffine)
        {
            init.ones_(weight);
        }
    }

    /// <summary>
    /// Normalizes the input tensor along the last dimension using Root Mean Square (RMS) normalization.
    /// </summary>
    /// <remarks>RMS normalization is performed by dividing the input tensor by the square root of the mean of its
    /// squared values  along the last dimension, with a small epsilon added for numerical stability. If element-wise affine
    /// transformation  is enabled, the normalized values are scaled by the <c>weight</c> parameter.</remarks>
    /// <param name="input">The input tensor to be normalized. Must be a non-null tensor.</param>
    /// <returns>A tensor with the same shape as the input, where the values are normalized along the last dimension. If element-wise
    /// affine transformation is enabled, the result is scaled by the <c>weight</c> parameter.</returns>
    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();
        var inputSqr = input.pow(2);
        var variance = inputSqr.mean([-1L], keepdim: true);
        var rms = rsqrt(variance + _epsilon);
        var normed = rms * input;

        if (_elementwiseAffine)
        {
            normed *= weight;
        }

        return normed.MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            weight?.Dispose();
        }
        base.Dispose(disposing);
    }
}