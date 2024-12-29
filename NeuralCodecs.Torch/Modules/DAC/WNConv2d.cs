using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

public class WNConv2d : Module<Tensor, Tensor>
{
    /// <summary>
    /// Stride for the convolution operation
    /// </summary>
    private readonly (long, long) _stride;

    /// <summary>
    /// Padding applied to input before convolution
    /// </summary>
    private readonly (long, long) _padding;

    /// <summary>
    /// Dilation factor for the convolution
    /// </summary>
    private readonly long _dilation;

    /// <summary>
    /// Number of groups for grouped convolution
    /// </summary>
    private readonly long _groups;

    public WNConv2d(
        long inChannels,
        long outChannels,
        (long, long) kernelSize,
        (long, long) stride = default,
        (long, long) padding = default,
        long dilation = 1,
        long groups = 1,
        bool useBias = true,
        Device device = null) : base($"WNConv2d_{inChannels}_{outChannels}")
    {
        if (stride == default) stride = (1, 1);
        if (padding == default) padding = (0, 0);
        device ??= torch.CPU;

        _stride = stride;
        _padding = padding;
        _dilation = dilation;
        _groups = groups;

        weight_g = Parameter(
            ones([1, outChannels / groups, 1, 1],
                 dtype: float32,
                 device: device));

        weight_v = Parameter(
            empty([outChannels, inChannels / groups, kernelSize.Item1, kernelSize.Item2],
                  dtype: float32,
                  device: device));

        if (useBias)
        {
            bias = new Parameter(empty(outChannels, dtype: float32));
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

    private void ResetParameters(bool useBias)
    {
        using (no_grad())
        {
            weight = empty_like(weight_v);
            init.kaiming_uniform_(weight, Math.Sqrt(5));

            var norm = weight.pow(2).sum([1, 2, 3], keepdim: true).sqrt();

            weight_g.set_(norm);
            weight_v.set_(weight.div(norm.add(1e-7)));

            if (useBias)
            {
                var fanIn = weight_v.size(1) * weight_v.size(2) * weight_v.size(3);
                var bound = fanIn > 0 ? 1.0 / Math.Sqrt(fanIn) : 0.0;
                init.uniform_(bias, -bound, bound);
            }
        }
    }

    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();

        var v_norm = weight_v.contiguous().pow(2)
                            .sum([1, 2, 3], keepdim: true, ScalarType.Float32)
                            .sqrt();

        weight = mul(weight_v.div(v_norm), weight_g.sub(1e-7f)).contiguous();

        return functional.conv2d(
            input,
            weight,
            bias,
            strides: [_stride.Item1, _stride.Item2],
            padding: [_padding.Item1, _padding.Item2],
            dilation: [_dilation, _dilation],
            groups: _groups).MoveToOuterDisposeScope();
    }

    // Used in the DAC Discriminator
    public static Sequential WithLeakyReLU(
        long inChannels,
        long outChannels,
        (long, long) kernelSize,
        (long, long) stride = default,
        (long, long) padding = default,
        long dilation = 1,
        long groups = 1,
        bool useBias = true,
        double negativeSlope = 0.1)
    {
        using var scope = torch.NewDisposeScope();
        var conv = new WNConv2d(inChannels, outChannels, kernelSize,
            stride: stride, padding: padding, dilation: dilation, groups: groups, useBias);

        return Sequential(conv, LeakyReLU(negativeSlope));
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
}