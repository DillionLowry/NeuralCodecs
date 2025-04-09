using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Weight-normalized 1D convolution layer.
/// Weight normalization is a reparameterization that decouples the magnitude
/// of a weight tensor from its direction.
/// </summary>
public class WNConv1d : Module<Tensor, Tensor>
{
    /// <summary>
    /// The learnable bias parameter for the convolution layer.
    /// Only available when useBias is set to true during initialization.
    /// </summary>
    public readonly Parameter bias;

    /// <summary>
    /// The gain parameter (g) used in weight normalization.
    /// Represents the magnitude/scale of the normalized weight vectors.
    /// </summary>
    public readonly Parameter weight_g;

    /// <summary>
    /// The directional weight parameter (v) used in weight normalization.
    /// Represents the direction of the weight vectors before normalization.
    /// </summary>
    public readonly Parameter weight_v;

    private const float Epsilon = 1e-7f;
    private readonly long _dilation;
    private readonly long _groups;
    private readonly long _inChannels;
    private readonly long _kernelSize;
    private readonly long _outChannels;
    private readonly long _padding;
    private readonly long _stride;

    /// <summary>
    /// Initialize weight-normalized 1D convolution
    /// </summary>
    /// <param name="inChannels">Number of input channels</param>
    /// <param name="outChannels">Number of output channels</param>
    /// <param name="kernelSize">Size of the convolving kernel</param>
    /// <param name="stride">Stride of the convolution</param>
    /// <param name="padding">Zero-padding added to both sides</param>
    /// <param name="dilation">Spacing between kernel elements</param>
    /// <param name="groups">Number of blocked connections</param>
    /// <param name="useBias">If True, adds a learnable bias</param>
    /// <param name="device">Device to use for computation</param>
    public WNConv1d(
        long inChannels,
        long outChannels,
        long kernelSize,
        long stride = 1,
        long padding = 0,
        long dilation = 1,
        long groups = 1,
        bool useBias = true,
        Device? device = null) : base("WNConv1d")
    {
        ValidateParameters(inChannels, outChannels, kernelSize,
            stride, padding, dilation, groups);

        device ??= CPU;
        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _dilation = dilation;
        _groups = groups;

        if (stride > 1 && dilation > 1)
        {
            Console.WriteLine(
                $"Warning: WNConv1d initialized with stride > 1 and dilation > 1 " +
                $"(kernel_size={kernelSize}, stride={stride}, dilation={dilation})");
        }

        weight_v = Parameter(
            empty([outChannels, inChannels / groups, kernelSize],
                  dtype: float32,
                  device: device));

        weight_g = Parameter(
            ones([outChannels, 1, 1],
                 dtype: float32,
                 device: device));

        if (useBias)
        {
            bias = Parameter(empty(outChannels, dtype: float32, device: device));
        }

        RegisterComponents();
        ResetParameters(useBias);
    }

    /// <summary>
    /// The computed normalized weight tensor.
    /// Calculated as (v * g) / ||v|| during the forward pass.
    /// </summary>
    public Tensor weight { get; set; }

    /// <summary>
    /// Forward pass of the weight-normalized 1D convolution layer.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();
        ValidateInputShape(input);

        var v_norm = weight_v.contiguous().pow(2)
                           .sum([1, 2], keepdim: true, ScalarType.Float32)
                           .sqrt();

        weight = mul(weight_v.div(v_norm), weight_g.sub(Epsilon)).contiguous();

        return functional.conv1d(input, weight, bias, _stride,
                                    _padding, _dilation, _groups)
                                    .MoveToOuterDisposeScope();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            weight_v?.Dispose();
            weight_g?.Dispose();
            bias?.Dispose();
        }
        base.Dispose(disposing);
    }

    private static void ValidateParameters(
       long inChannels, long outChannels, long kernelSize,
       long stride, long padding, long dilation, long groups)
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

        if (padding < 0)
        {
            throw new ArgumentException($"Padding must be non-negative, got {padding}");
        }

        if (dilation <= 0)
        {
            throw new ArgumentException($"Dilation must be positive, got {dilation}");
        }

        if (groups <= 0)
        {
            throw new ArgumentException($"Groups must be positive, got {groups}");
        }

        if (inChannels % groups != 0)
        {
            throw new ArgumentException(
                $"Input channels ({inChannels}) must be divisible by groups ({groups})");
        }

        if (outChannels % groups != 0)
        {
            throw new ArgumentException(
                $"Output channels ({outChannels}) must be divisible by groups ({groups})");
        }
    }

    private void ResetParameters(bool useBias)
    {
        using (no_grad())
        {
            weight = empty_like(weight_v);
            init.trunc_normal_(weight, std: 0.02f);

            // Compute norm along dims [1,2] (in_channels and kernel_size) with keepdim
            var norm = weight_v.contiguous().pow(2)
               .sum([1, 2], keepdim: true, ScalarType.Float32)
               .sqrt();

            weight_g.set_(norm);

            weight_v.set_(weight.div(norm.sub(Epsilon)));

            if (useBias)
            {
                var fan_in = weight_v.size(1) * weight_g.size(2);
                var bound = 1.0 / Math.Sqrt(fan_in);

                init.uniform_(bias, -bound, bound);
            }
        }
    }

    private void ValidateInputShape(Tensor input)
    {
        if (input.dim() != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input tensor [B, C, T], got shape [{string.Join(", ", input.shape)}]");
        }

        var channels = input.size(1);
        if (channels != _inChannels)
        {
            throw new ArgumentException(
                $"Expected {_inChannels} input channels, got {channels}");
        }
    }
}