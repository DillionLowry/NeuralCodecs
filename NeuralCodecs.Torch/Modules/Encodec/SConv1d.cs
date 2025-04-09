using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Conv1d module with integrated handling of asymmetric/causal padding and normalization.
/// Provides proper padding and dilation support matching the Encodec implementation.
/// </summary>
public class SConv1d : Module<Tensor, Tensor>
{
    /// <summary>
    /// Set of valid padding modes supported by this module.
    /// </summary>
    private static readonly HashSet<string> ValidPadModes = new()
    {
        "reflect", "replicate", "circular", "zero"
    };

    /// <summary>
    /// Flag indicating if the convolution is causal.
    /// </summary>
    private readonly bool _causal;

    /// <summary>
    /// Spacing between kernel elements.
    /// </summary>
    private readonly long _dilation;

    /// <summary>
    /// Size of the convolving kernel.
    /// </summary>
    private readonly int _kernelSize;

    /// <summary>
    /// The underlying normalized convolution module.
    /// </summary>
    private readonly NormConv1d _normConv;

    /// <summary>
    /// The padding mode to use ('reflect', 'replicate', 'circular', or 'zero').
    /// </summary>
    private readonly string _padMode;

    /// <summary>
    /// Convolution stride value.
    /// </summary>
    private readonly int _stride;

    /// <summary>
    /// Initializes a new instance of the <see cref="SConv1d"/> class.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="kernelSize">Size of the convolving kernel.</param>
    /// <param name="stride">Stride of the convolution (default: 1).</param>
    /// <param name="dilation">Spacing between kernel elements (default: 1).</param>
    /// <param name="groups">Number of blocked connections from input to output channels (default: 1).</param>
    /// <param name="bias">If true, adds a learnable bias to the output (default: true).</param>
    /// <param name="causal">If true, uses causal convolution (default: false).</param>
    /// <param name="normType">Type of normalization to apply (default: "weight_norm").</param>
    /// <param name="normParams">Optional parameters for normalization (default: null).</param>
    /// <param name="padMode">Padding mode to use (default: "reflect").</param>
    public SConv1d(
        int inChannels,
        int outChannels,
        int kernelSize,
        int stride = 1,
        long dilation = 1,
        int groups = 1,
        bool bias = true,
        bool causal = false,
        string normType = "weight_norm",
        Dictionary<string, object> normParams = null,
        string padMode = "reflect") : base($"SConv1d")
    {
        ValidateParameters(inChannels, outChannels, kernelSize, stride,
            dilation, groups, padMode);

        if (stride > 1 && dilation > 1)
        {
            Console.WriteLine(
                "Warning: SConv1d has been initialized with stride > 1 and dilation > 1 " +
                $"(kernel_size={kernelSize}, stride={stride}, dilation={dilation}).");
        }
        _causal = causal;
        _padMode = padMode.ToLowerInvariant();
        _stride = stride;
        _kernelSize = kernelSize;
        _dilation = dilation;

        _normConv = new NormConv1d(
                inChannels, outChannels, kernelSize,
                stride: stride,
                padding: 0, // We handle padding manually
                dilation: dilation,
                groups: groups,
                bias: bias,
                causal: causal,
                norm: normType,
                normParams: normParams ?? new Dictionary<string, object>());

        // Register inner components directly
        if (normType == "weight_norm")
        {
            if (_normConv.conv is not WNConv1d wnConv)
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
            var conv1d = _normConv.conv as Conv1d;
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
    /// Gets the normalization module of the underlying convolution.
    /// </summary>
    public Module<Tensor, Tensor> Norm => _normConv.norm;

    /// <summary>
    /// Performs the forward pass of the module with appropriate padding handling.
    /// </summary>
    /// <param name="x">Input tensor of shape [B, C, T] where B is batch size, C is channels, and T is time steps.</param>
    /// <returns>Output tensor after convolution with padding adjustments.</returns>
    /// <exception cref="ArgumentException">Thrown when input tensor does not have the expected shape.</exception>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        ValidateInputShape(x);

        var length = x.size(2);

        // Calculate effective kernel size with dilation
        var effectiveKernelSize = ((_kernelSize - 1) * _dilation) + 1;
        var paddingTotal = effectiveKernelSize - _stride;

        // Get any extra padding needed for stride alignment
        var extraPadding = GetExtraPaddingForConv1d(length, effectiveKernelSize, _stride, paddingTotal);

        Tensor padded;
        if (_causal)
        {
            // Left padding for causal convolution
            padded = Pad1d(x, (paddingTotal, extraPadding));
        }
        else
        {
            // Asymmetric padding for odd strides
            var paddingRight = paddingTotal / 2;
            var paddingLeft = paddingTotal - paddingRight;
            padded = Pad1d(x, (paddingLeft, paddingRight + extraPadding));
        }

        return _normConv.forward(padded).MoveToOuterDisposeScope();
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
    /// Validates all input parameters for the convolution.
    /// </summary>
    /// <param name="inChannels">Number of input channels to validate.</param>
    /// <param name="outChannels">Number of output channels to validate.</param>
    /// <param name="kernelSize">Kernel size to validate.</param>
    /// <param name="stride">Stride value to validate.</param>
    /// <param name="dilation">Dilation value to validate.</param>
    /// <param name="groups">Number of groups to validate.</param>
    /// <param name="padMode">Padding mode to validate.</param>
    /// <exception cref="ArgumentException">Thrown when any parameter is invalid.</exception>
    private static void ValidateParameters(
        int inChannels, int outChannels, int kernelSize,
        int stride, long dilation, int groups, string padMode)
    {
        if (inChannels <= 0)
        {
            throw new ArgumentException($"Input channels must be positive, got {inChannels}");
        }

        if (outChannels <= 0)
        {
            throw new ArgumentException($"Output channels must be positive, got {outChannels}");
        }

        if (kernelSize <= 0)
        {
            throw new ArgumentException($"Kernel size must be positive, got {kernelSize}");
        }

        if (stride <= 0)
        {
            throw new ArgumentException($"Stride must be positive, got {stride}");
        }

        if (dilation <= 0)
        {
            throw new ArgumentException($"Dilation must be positive, got {dilation}");
        }

        if (groups <= 0)
        {
            throw new ArgumentException($"Groups must be positive, got {groups}");
        }

        if (!ValidPadModes.Contains(padMode.ToLowerInvariant()))
        {
            throw new ArgumentException(
                $"Invalid padding mode '{padMode}'. Valid options are: {string.Join(", ", ValidPadModes)}");
        }
    }

    /// <summary>
    /// Calculates extra padding needed to maintain proper alignment with stride.
    /// </summary>
    /// <param name="length">Input length.</param>
    /// <param name="kernelSize">Effective kernel size.</param>
    /// <param name="stride">Convolution stride.</param>
    /// <param name="paddingTotal">Total padding being applied.</param>
    /// <returns>Extra padding needed for proper stride alignment.</returns>
    private long GetExtraPaddingForConv1d(long length, long kernelSize, int stride, long paddingTotal)
    {
        var nFrames = ((length - kernelSize + paddingTotal) / (float)stride) + 1;
        var idealLength = (((long)Math.Ceiling(nFrames) - 1) * stride) + (kernelSize - paddingTotal);
        return idealLength - length;
    }

    /// <summary>
    /// Applies padding to the input tensor using the specified mode.
    /// </summary>
    /// <param name="x">Input tensor to pad.</param>
    /// <param name="padding">Tuple of (left, right) padding amounts.</param>
    /// <returns>Padded tensor.</returns>
    private Tensor Pad1d(Tensor x, (long left, long right) padding)
    {
        if (_padMode == "reflect" && x.size(-1) <= Math.Max(padding.left, padding.right))
        {
            // Handle small input case for reflect padding
            var maxPad = Math.Max(padding.left, padding.right);
            var extraPad = maxPad - x.size(-1) + 1;

            // Add zero padding first
            var zeroPadded = nn.functional.pad(x, [0, extraPad], PaddingModes.Constant, 0);

            // Then apply reflection padding
            return nn.functional.pad(zeroPadded, [padding.left, padding.right], PaddingModes.Reflect);
        }

        return nn.functional.pad(x, [padding.left, padding.right], PaddingModes.Reflect);
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