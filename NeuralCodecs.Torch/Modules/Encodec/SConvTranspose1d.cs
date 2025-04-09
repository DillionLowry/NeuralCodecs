using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Transposed convolution with integrated padding and causality handling.
/// Provides support for asymmetric padding and causal convolution with configurable right trimming.
/// </summary>
public class SConvTranspose1d : Module<Tensor, Tensor>
{
    /// <summary>
    /// Flag indicating if the convolution is causal.
    /// </summary>
    private readonly bool _causal;

    /// <summary>
    /// The underlying normalized transposed convolution module.
    /// </summary>
    private readonly NormConvTranspose1d _normConv;

    /// <summary>
    /// Total padding to apply (kernel_size - stride).
    /// </summary>
    private readonly long _paddingTotal;

    /// <summary>
    /// Convolution stride value.
    /// </summary>
    private readonly long _stride;

    /// <summary>
    /// Ratio for trimming padding on the right side (between 0 and 1).
    /// </summary>
    private readonly float _trimRightRatio;

    /// <summary>
    /// Initializes a new instance of the <see cref="SConvTranspose1d"/> class.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="kernelSize">Size of the convolving kernel.</param>
    /// <param name="stride">Stride of the convolution (default: 1).</param>
    /// <param name="padding">Zero-padding added to both sides of the input (default: 0).</param>
    /// <param name="outputPadding">Additional size added to output shape (default: 0).</param>
    /// <param name="causal">If true, uses causal convolution (default: false).</param>
    /// <param name="normType">Type of normalization to apply (default: "weight_norm").</param>
    /// <param name="normParams">Optional parameters for normalization (default: null).</param>
    /// <param name="trimRightRatio">Ratio for trimming right padding, between 0 and 1 (default: 1.0).</param>
    /// <exception cref="ArgumentException">Thrown when trimRightRatio is not between 0 and 1.</exception>
    public SConvTranspose1d(
        int inChannels,
        int outChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        int outputPadding = 0,
        bool causal = false,
        string normType = "weight_norm",
        Dictionary<string, object> normParams = null,
        float trimRightRatio = 1.0f) : base("SConvTranspose1d")
    {
        if (trimRightRatio is < 0 or > 1)
        {
            throw new ArgumentException($"Trim right ratio must be between 0 and 1, got {trimRightRatio}");
        }

        _stride = stride;
        _causal = causal;
        _trimRightRatio = trimRightRatio;
        _paddingTotal = kernelSize - stride;

        _normConv = new NormConvTranspose1d(
        inChannels, outChannels, kernelSize,
            stride: stride,
            padding: padding,
            outputPadding: outputPadding,
            causal: causal,
            normType: normType,
            normParams: normParams ?? new Dictionary<string, object>());

        if (normType == "weight_norm")
        {
            if (_normConv.conv is not WNConvTranspose1d wnConv)
            {
                throw new ArgumentException("Weight normalization not found in convolution module");
            }

            register_parameter("conv.weight_v", wnConv.weight_v);
            register_parameter("conv.weight_g", wnConv.weight_g);
            if (wnConv.bias is not null)
            {
                register_parameter("conv.bias", wnConv.bias);
            }
        }
        else
        {
            var conv1d = _normConv.conv as ConvTranspose1d;
            register_parameter("conv.weight", conv1d.weight);
            if (conv1d.bias is not null)
            {
                register_parameter("conv.bias", conv1d.bias);
            }
        }
        register_module("norm", _normConv.norm);
        RegisterComponents();
    }

    /// <summary>
    /// Performs the forward pass of the transposed convolution with appropriate padding handling.
    /// </summary>
    /// <param name="x">Input tensor of shape [B, C, T] where B is batch size, C is channels, and T is time steps.</param>
    /// <returns>Output tensor after transposed convolution and padding adjustment.</returns>
    /// <exception cref="ArgumentException">Thrown when input tensor does not have the expected shape.</exception>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        ValidateInputShape(x);

        var y = _normConv.forward(x);

        if (_causal)
        {
            // Trim the padding on the right according to the specified ratio
            var paddingRight = (long)Math.Ceiling(_paddingTotal * _trimRightRatio);
            var paddingLeft = _paddingTotal - paddingRight;
            y = Unpad1d(y, paddingLeft, paddingRight);
        }
        else
        {
            // Asymmetric padding required for odd strides
            var paddingRight = _paddingTotal / 2;
            var paddingLeft = _paddingTotal - paddingRight;
            y = Unpad1d(y, paddingLeft, paddingRight);
        }

        return y.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Disposes the managed resources used by the module.
    /// </summary>
    /// <param name="disposing">True to dispose managed resources, false otherwise.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _normConv?.Dispose();
        }
        base.Dispose(disposing);
    }

    /// <summary>
    /// Removes padding from both ends of the output tensor.
    /// </summary>
    /// <param name="x">Input tensor to unpad.</param>
    /// <param name="paddingLeft">Amount of padding to remove from the left.</param>
    /// <param name="paddingRight">Amount of padding to remove from the right.</param>
    /// <returns>Unpadded tensor.</returns>
    /// <exception cref="ArgumentException">Thrown when total padding exceeds tensor length.</exception>
    private static Tensor Unpad1d(Tensor x, long paddingLeft, long paddingRight)
    {
        var length = x.size(-1);
        if (paddingLeft + paddingRight >= length)
        {
            throw new ArgumentException(
                "Total padding exceeds tensor length, " +
                $"got padding ({paddingLeft}, {paddingRight}) for length {length}");
        }

        var end = length - paddingRight;
        return x.slice(-1, paddingLeft, end, 1);
    }

    /// <summary>
    /// Validates that the input tensor has the correct shape [B, C, T].
    /// </summary>
    /// <param name="x">Input tensor to validate.</param>
    /// <exception cref="ArgumentException">Thrown when tensor shape is invalid.</exception>
    private void ValidateInputShape(Tensor x)
    {
        if (x.dim() != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input tensor [B, C, T], got shape [{string.Join(", ", x.shape)}]");
        }
    }
}