using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

public class WNConv2d : Module<Tensor, Tensor>
{
    private readonly Parameter _bias;
    private readonly ParameterDict _parametrizations = new();

    private readonly (long, long) _stride;
    private readonly (long, long) _padding;
    private readonly long _dilation;
    private readonly long _groups;

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _parametrizations["weight.original0"]?.Dispose();
            _parametrizations["weight.original1"]?.Dispose();
            _bias?.Dispose();
        }
        base.Dispose(disposing);
    }

    public WNConv2d(
        long inChannels,
        long outChannels,
        (long, long) kernelSize,
        (long, long) stride = default,
        (long, long) padding = default,
        long dilation = 1,
        long groups = 1,
        bool useBias = true) : base($"WNConv2d")
    {
        if (stride == default) stride = (1, 1);
        if (padding == default) padding = (0, 0);

        _stride = stride;
        _padding = padding;
        _dilation = dilation;
        _groups = groups;

        // g parameter shape: [1, outChannels/groups, 1, 1]
        _parametrizations.Add("weight.original0", Parameter(
            empty([1, outChannels / groups, 1, 1])));

        // v parameter shape: [outChannels, inChannels/groups, kernelSize0, kernelSize1]
        _parametrizations.Add("weight.original1", Parameter(
            empty([outChannels, inChannels / groups, kernelSize.Item1, kernelSize.Item2])));

        if (useBias)
        {
            _bias = new Parameter(empty(outChannels, dtype: float32));
        }

        RegisterComponents();
        ResetParameters();
    }

    public void ResetParameters()
    {
        using (no_grad())
        {
            // Initialize weight_v using Kaiming initialization
            var weight = empty_like(_parametrizations["weight.original1"]);
            init.kaiming_uniform_(weight, Math.Sqrt(5));

            // Compute norm along dims [1,2,3] with keepdim
            var norm = sqrt(weight.pow(2).sum([1, 2, 3], keepdim: true));

            // Set weight_g and weight_v
            _parametrizations["weight.original0"].set_(norm);
            _parametrizations["weight.original1"].set_(weight.div(norm.add(1e-7)));

            if (_bias is not null)
            {
                var fanIn = _parametrizations["weight.original1"].size(1) *
                           _parametrizations["weight.original1"].size(2) *
                           _parametrizations["weight.original1"].size(3);
                var bound = fanIn > 0 ?
                            1.0 / Math.Sqrt(fanIn) :
                            0.0;
                init.uniform_(_bias, -bound, bound);
            }
        }
    }

    public override Tensor forward(Tensor input)
    {
        try
        {
            using var scope = NewDisposeScope();
            var weight_v = _parametrizations["weight.original1"];
            var weight_g = _parametrizations["weight.original0"];
            var v_norm = weight_v.contiguous().pow(2).sum([1, 2, 3], keepdim: true, ScalarType.Float32).sqrt();
            var weight = mul(weight_v.div(v_norm), weight_g.sub(1e-7f)).contiguous();
            return functional.conv2d(
                input,
                weight,
                _bias,
                strides: [_stride.Item1, _stride.Item2],
                padding: [_padding.Item1, _padding.Item2],
                dilation: [_dilation, _dilation], // TODO: Check this, dilation was a single value
                groups: _groups);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in forward pass: {ex.Message}");
            throw;
        }
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
        var conv = new WNConv2d(inChannels, outChannels, kernelSize,
            stride: stride, padding: padding, dilation: dilation, groups: groups, useBias);

        return Sequential(conv, LeakyReLU(negativeSlope));
    }
}