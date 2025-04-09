using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Normalized transposed convolution implementation.
/// </summary>
public class NormConvTranspose1d : Module<Tensor, Tensor>
{
    private static readonly HashSet<string> ValidNorms = new()
    {
        "none", "layer_norm", "weight_norm", "spectral_norm", "group_norm", "time_group_norm"
    };

    //private readonly ConvTranspose1d _convtr;
    //private readonly Module<Tensor, Tensor> _norm;
    //private readonly string _normType;

    public NormConvTranspose1d(
        int inChannels,
        int outChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        int outputPadding = 0,
        int dilation = 1,
        int groups = 1,
        bool causal = false,
        string normType = "weight_norm",
        Dictionary<string, object> normParams = null) : base("NormConvTranspose1d")
    {
        ValidateParameters(inChannels, outChannels, kernelSize, stride,
            padding, outputPadding, dilation, groups, normType);

        conv = normType.Equals("weight_norm", StringComparison.InvariantCultureIgnoreCase) ?
            new WNConvTranspose1d(inChannels, outChannels, kernelSize,
                    stride: stride,
                    padding: padding,
                    outputPadding: outputPadding,
                    dilation: dilation,
                    groups: groups) :
            ConvTranspose1d(inChannels, outChannels, kernelSize,
                    stride: stride,
                    padding: padding,
                    output_padding: outputPadding,
                    dilation: dilation,
                    groups: groups);

        if (normType.Equals("group_norm", StringComparison.InvariantCultureIgnoreCase) || normType.Equals("time_group_norm", StringComparison.InvariantCultureIgnoreCase))
        {
            norm = GetNormModule(conv as ConvTranspose1d, causal, normType, normParams);
        }

        //RegisterComponents();
    }

    public Module<Tensor, Tensor> conv { get; set; }
    public Module<Tensor, Tensor> norm { get; set; }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        ValidateInputShape(x);

        // Apply transposed convolution and normalization
        var convOut = conv.forward(x);
        if (norm is not null)
        {
            var normOut = norm.forward(convOut);
            return normOut.MoveToOuterDisposeScope();
        }

        return convOut.MoveToOuterDisposeScope();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            conv?.Dispose();
            norm?.Dispose();
        }
        base.Dispose(disposing);
    }

    private static Module<Tensor, Tensor> GetNormModule(
        ConvTranspose1d conv,
        bool causal,
        string norm,
        Dictionary<string, object> normParams)
    {
        normParams ??= new Dictionary<string, object>();
        var eps = normParams.TryGetValue("eps", out var epsObj) ?
    Convert.ToSingle(epsObj) : 1e-5f;

        switch (norm.ToLowerInvariant())
        {
            case "layer_norm":
                return new ConvLayerNorm((int)conv.out_channels, eps);

            //return normParams.TryGetValue("eps", out object? leps) ? LayerNorm(new long[] { conv.out_channels }, (double)leps)
            //    : LayerNorm(new long[] { conv.out_channels });

            case "group_norm":
            case "time_group_norm":
                if (causal)
                {
                    throw new ArgumentException(
                        "GroupNorm doesn't support causal evaluation");
                }

                if (conv == null)
                {
                    throw new ArgumentException("Cannot determine output channels for group norm");
                }

                //return new TimeGroupNorm(conv.out_channels, eps);
                //case "group_norm":
                //    if (causal)
                //    {
                //        throw new ArgumentException(
                //            "GroupNorm doesn't support causal evaluation");
                //    }
                return GroupNorm(1, conv.out_channels, eps, affine: true);

            case "weight_norm":
            case "spectral_norm":
            case "none":
                return Identity();

            default:
                throw new ArgumentException($"Unsupported normalization: {norm}");
        }
    }

    private static void ValidateParameters(
                        int inChannels, int outChannels, int kernelSize,
            int stride, int padding, int outputPadding,
            int dilation, int groups, string norm)
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

        if (!ValidNorms.Contains(norm.ToLowerInvariant()))
        {
            throw new ArgumentException(
                $"Invalid normalization '{norm}'. Valid options are: {string.Join(", ", ValidNorms)}");
        }
    }

    private void ValidateInputShape(Tensor x)
    {
        if (x.dim() != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input tensor [B, C, T], got shape [{string.Join(", ", x.shape)}]");
        }
    }
}