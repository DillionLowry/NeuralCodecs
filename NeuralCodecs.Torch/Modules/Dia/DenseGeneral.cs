using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// PyTorch equivalent of flax.linen.DenseGeneral with shapes defined at init.
/// Enables arbitrary tensor contractions, not just matrix multiplication.
/// </summary>
public class DenseGeneral : Module<Tensor, Tensor>
{
    private readonly int[] _axis;
    private readonly int[] _inShapes;
    private readonly int[] _outFeatures;
    public readonly Parameter weight;
    private readonly Parameter _bias;

    public int[] InShapes => _inShapes;
    public int[] OutFeatures => _outFeatures;

    /// <summary>
    /// Creates a new DenseGeneral layer.
    /// </summary>
    /// <param name="inShapes">Sizes of the input dimensions specified by axis</param>
    /// <param name="outFeatures">Shape of the output features (non-contracted dims)</param>
    /// <param name="axis">Input axis or axes to contract (default: -1)</param>
    /// <param name="weightDtype">Data type for weights</param>
    /// <param name="device">Device for the layer</param>
    public DenseGeneral(
        int[] inShapes,
        int[] outFeatures,
        int[] axis = null,
        ScalarType? weightDtype = null,
        Device device = null)
        : base(nameof(DenseGeneral))
    {
        _inShapes = inShapes ?? throw new ArgumentNullException(nameof(inShapes));
        _outFeatures = outFeatures ?? throw new ArgumentNullException(nameof(outFeatures));
        _axis = axis ?? [-1];
        device ??= CPU;
        weightDtype ??= ScalarType.Float32;

        // Calculate kernel shape (inShapes + outFeatures)
        var kernelShape = new long[_inShapes.Length + _outFeatures.Length];
        for (int i = 0; i < _inShapes.Length; i++)
        {
            kernelShape[i] = _inShapes[i];
        }
        for (int i = 0; i < _outFeatures.Length; i++)
        {
            kernelShape[_inShapes.Length + i] = _outFeatures[i];
        }

        weight = Parameter(empty(kernelShape, dtype: weightDtype, device: device));
        RegisterComponents();
    }

    /// <summary>
    /// Normalizes the specified axes to ensure all values are non-negative and within the range of dimensions.
    /// </summary>
    /// <param name="axes">An array of integers representing the axes to normalize. Negative values are interpreted as offsets from the
    /// end of the dimensions.</param>
    /// <param name="ndim">The total number of dimensions. Must be a positive value.</param>
    /// <returns>An array of long integers where each value corresponds to a normalized axis, guaranteed to be in the range
    /// [0, <paramref name="ndim"/> - 1].</returns>
    public static long[] NormalizeAxes(int[] axes, long ndim)
    {
        var result = new long[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            result[i] = axes[i] >= 0 ? axes[i] : ndim + axes[i];
        }
        return result;
    }

    /// <summary>
    /// Applies a tensor contraction operation between the input tensor and the weight tensor along specified axes.
    /// </summary>
    /// <remarks>This method performs a generalized tensor contraction (similar to a matrix
    /// multiplication) between the input tensor and the weight tensor.  The axes for contraction are determined by
    /// the <c>_axis</c> field for the input tensor and the corresponding axes of the weight tensor.</remarks>
    /// <param name="inputs">The input tensor to be processed. Must have dimensions compatible with the weight tensor along the specified
    /// axes.</param>
    /// <returns>A tensor resulting from the contraction of the input tensor with the weight tensor. The resulting tensor has
    /// the same data type as the input tensor.</returns>
    public override Tensor forward(Tensor inputs)
    {
        var normAxis = NormalizeAxes(_axis, inputs.dim());

        var kernelContractAxes = new long[normAxis.Length];
        for (int i = 0; i < normAxis.Length; i++)
        {
            kernelContractAxes[i] = i;
        }
        var output = tensordot(
            inputs.to(weight.dtype),
            weight,
            dims1: normAxis,
            dims2: kernelContractAxes
        );

        return output.to(inputs.dtype);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            weight?.Dispose();
            _bias?.Dispose();
        }
        base.Dispose(disposing);
    }
}