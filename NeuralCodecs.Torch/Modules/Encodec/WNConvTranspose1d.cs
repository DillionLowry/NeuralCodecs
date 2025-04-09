using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Implements a Weight Normalized 1D Transposed Convolution layer.
/// Weight Normalization is a reparameterization of the weights that decouples the magnitude
/// of those weights from their direction, simulating PyTorch's torch.nn.utils.parametrizations.weight_norm().
/// </summary>
public class WNConvTranspose1d : Module<Tensor, Tensor>
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

    /// <summary>
    /// Dilation factor for the convolution
    /// </summary>
    private readonly long _dilation;

    /// <summary>
    /// Number of groups for grouped convolution
    /// </summary>
    private readonly long _groups;

    private readonly long _outputPadding;

    /// <summary>
    /// Padding applied to input before convolution
    /// </summary>
    private readonly long _padding;

    /// <summary>
    /// Stride for the convolution operation
    /// </summary>
    private readonly long _stride;

    /// <summary>
    /// Initialize weight-normalized transposed 1D convolution
    /// </summary>
    /// <param name="inChannels">Number of channels in the input tensor</param>
    /// <param name="outChannels">Number of channels produced by the convolution</param>
    /// <param name="kernelSize">Size of the convolving kernel</param>
    /// <param name="stride">Stride of the convolution (default: 1)</param>
    /// <param name="padding">Padding added to input (default: 0)</param>
    /// <param name="outputPadding">Additional size added to output shape (default: 0)</param>
    /// <param name="dilation">Spacing between kernel elements (default: 1)</param>
    /// <param name="groups">Number of blocked connections from input to output channels (default: 1)</param>
    /// <param name="useBias">If true, adds a learnable bias to the output (default: true)</param>
    /// <param name="device">Device to use for computation</param>
    public WNConvTranspose1d(
           long inChannels,
            long outChannels,
            long kernelSize,
            long stride = 1,
            long padding = 0,
            long outputPadding = 0,
            long dilation = 1,
            long groups = 1,
            bool useBias = true,
            Device? device = null)
        : base($"conv_t")
    {
        ValidateParameters(inChannels, outChannels, kernelSize, stride, padding, outputPadding, dilation, groups);

        _stride = stride;
        _padding = padding;
        _dilation = dilation;
        _groups = groups;
        _outputPadding = outputPadding;
        device ??= CPU;

        weight_v = Parameter(
            empty([inChannels, outChannels / groups, kernelSize],
                  dtype: float32,
                  device: device));

        weight_g = Parameter(
            ones([1, outChannels / groups, 1],
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
    /// Performs forward pass applying weight normalized transposed convolution
    /// </summary>
    /// <param name="input">Input tensor of shape (batch, in_channels, length)</param>
    /// <returns>
    /// Upsampled tensor with shape (batch, out_channels, new_length)
    /// where new_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    /// </returns>
    public override Tensor forward(Tensor input)
    {
        try
        {
            using var scope = NewDisposeScope();

            if (input.dim() != 3)
            {
                throw new ArgumentException(
                    $"Expected 3D input tensor [B, C, T], got shape [{string.Join(", ", input.shape)}]");
            }
            var vNorm = weight_v.contiguous().pow(2)
                .sum(new long[] { 1, 2 }, keepdim: true, ScalarType.Float32)
                .sqrt();
            weight = mul(weight_v.div(vNorm), weight_g.sub(Epsilon)).contiguous();

            return functional.conv_transpose1d(
                input,
                weight,
                bias,
                stride: _stride,
                padding: _padding,
                output_padding: _outputPadding,
                groups: _groups,
                dilation: _dilation)
                .MoveToOuterDisposeScope();
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            throw;
        }
    }

    /// <inheritdoc/>
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
               long stride, long padding, long outputPadding, long dilation, long groups)
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

        if (outputPadding < 0)
        {
            throw new ArgumentException($"Output padding must be non-negative, got {outputPadding}");
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
            throw new ArgumentException($"Input channels ({inChannels}) must be divisible by groups ({groups})");
        }

        if (outChannels % groups != 0)
        {
            throw new ArgumentException($"Output channels ({outChannels}) must be divisible by groups ({groups})");
        }
    }

    /// <summary>
    /// Resets the parameters of the layer, initializing the weights using Kaiming uniform initialization and setting the bias.
    /// </summary>
    /// <param name="useBias">If true, initializes the bias parameter.</param>
    private void ResetParameters(bool useBias)
    {
        using (no_grad())
        {
            weight = empty_like(weight_v);
            init.trunc_normal_(weight, std: 0.02f);

            var norm = weight_v.contiguous().pow(2)
                       .sum([1, 2], keepdim: true, ScalarType.Float32)
                       .sqrt();

            weight_g.set_(norm);

            weight_v.set_(weight.div(norm.sub(Epsilon)));

            if (useBias)
            {
                init.constant_(bias, 0f);
                var fan_in = weight_v.size(1) * weight_g.size(2);
                var bound = 1.0 / Math.Sqrt(fan_in);

                init.uniform_(bias, -bound, bound);
            }
        }
    }
}