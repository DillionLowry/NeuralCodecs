using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Implements a Weight Normalized 1D Convolution layer.
/// Weight Normalization is a reparameterization of the weights that decouples the magnitude
/// of those weights from their direction, simulating PyTorch's deprecated torch.nn.utils.weight_norm().
/// </summary>
public class WNConv1d : Module<Tensor, Tensor>
{
    /// <summary>
    /// Stride for the convolution operation
    /// </summary>
    private readonly long _stride;

    /// <summary>
    /// Padding applied to input before convolution
    /// </summary>
    private readonly long _padding;

    /// <summary>
    /// Dilation factor for the convolution
    /// </summary>
    private readonly long _dilation;

    /// <summary>
    /// Number of groups for grouped convolution
    /// </summary>
    private readonly long _groups;

    /// <summary>
    /// Initializes a new instance of the WNConv1d class.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="kernelSize">Size of the convolution kernel.</param>
    /// <param name="stride">Stride of the convolution. Default is 1.</param>
    /// <param name="padding">Zero-padding added to both sides of the input. Default is 0.</param>
    /// <param name="dilation">Spacing between kernel elements. Default is 1.</param>
    /// <param name="groups">Number of blocked connections from input channels to output channels. Default is 1.</param>
    /// <param name="useBias">If true, adds a learnable bias to the output. Default is true.</param>
    public WNConv1d(long inChannels, long outChannels, long kernelSize,
        long stride = 1, long padding = 0, long dilation = 1, long groups = 1, bool useBias = true, Device device = null)
        : base($"WNConv1d_{inChannels}_{outChannels}")
    {
        _stride = stride;
        _padding = padding;
        _dilation = dilation;
        _groups = groups;
        device ??= torch.CPU;

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
            bias = Parameter(empty(outChannels, dtype: float32));
        }
        RegisterComponents();
        ResetParameters(useBias);
    }

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

    /// <summary>
    /// The computed normalized weight tensor.
    /// Calculated as (v * g) / ||v|| during the forward pass.
    /// </summary>
    public Tensor weight { get; set; }

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

            weight_v.set_(weight.div(norm.sub(1e-7f)));

            if (useBias)
            {
                init.constant_(bias, 0f);
                var fan_in = weight_v.size(1) * weight_g.size(2);
                var bound = 1.0 / Math.Sqrt(fan_in);

                init.uniform_(bias, -bound, bound);
            }
        }
    }

    /// <summary>
    /// Performs forward pass applying weight normalized convolution
    /// </summary>
    /// <remarks>
    /// The computed L2 norm of v is almost exactly the same as PyTorch's value, but there is a potential floating
    /// point difference in the least significant digit causing a compounding error of +/-(2^31 - 1)
    /// </remarks>
    /// <param name="input">Input tensor of shape (batch, in_channels, length)</param>
    /// <returns>
    /// Convolved tensor with shape (batch, out_channels, new_length)
    /// </returns>
    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();

        // Compute norm per output channel
        var v_norm = weight_v.contiguous().pow(2)
                           .sum([1, 2], keepdim: true, ScalarType.Float32)
                           .sqrt();

        weight = mul(weight_v.div(v_norm), weight_g.sub(1e-7f)).contiguous();

        return functional.conv1d(input, weight, bias, _stride,
                                    _padding, _dilation, _groups)
                                    .MoveToOuterDisposeScope();
    }

    // Used in the DAC Discriminator
    public static Sequential WithLeakyReLU(
        long inChannels,
        long outChannels,
        long kernelSize,
        long stride = 1,
        long padding = 0,
        long dilation = 1,
        long groups = 1,
        bool useBias = true,
        double negativeSlope = 0.1)
    {
        var conv = new WNConv1d(inChannels, outChannels, kernelSize,
            stride: stride, padding: padding, dilation: dilation, groups: groups, useBias);

        return Sequential(conv, LeakyReLU(negativeSlope));
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
}