using System.Diagnostics;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Wrapper around Conv1d with integrated normalization support,
/// including weight, layer, and group normalization.
/// </summary>
public class NormConv1d : Module<Tensor, Tensor>
{
    /// <summary>
    /// Set of valid normalization types supported by this module.
    /// </summary>
    private static readonly HashSet<string> ValidNorms = new()
    {
        "none", "layer_norm", "weight_norm", "group_norm", "time_group_norm"
    };

    /// <summary>
    /// Initializes a new instance of the <see cref="NormConv1d"/> class.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="kernelSize">Size of the convolving kernel.</param>
    /// <param name="stride">Stride of the convolution (default: 1).</param>
    /// <param name="padding">Zero-padding added to both sides of the input (default: 0).</param>
    /// <param name="dilation">Spacing between kernel elements (default: 1).</param>
    /// <param name="groups">Number of blocked connections from input to output channels (default: 1).</param>
    /// <param name="bias">If true, adds a learnable bias to the output (default: true).</param>
    /// <param name="causal">If true, uses causal convolution (default: false).</param>
    /// <param name="norm">Type of normalization to apply (default: "weight_norm").</param>
    /// <param name="normParams">Optional parameters for normalization (default: null).</param>
    public NormConv1d(
        int inChannels,
        int outChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        long dilation = 1,
        int groups = 1,
        bool bias = true,
        bool causal = false,
        string norm = "weight_norm",
        Dictionary<string, object>? normParams = null) : base($"NormConv1d")
    {
        ValidateParameters(inChannels, outChannels, kernelSize,
            stride, padding, dilation, groups, norm, bias, causal);

        // Create appropriate convolution based on normalization type
        conv = norm.Equals("weight_norm", StringComparison.InvariantCultureIgnoreCase) ?
            new WNConv1d(inChannels, outChannels, kernelSize,
                stride: stride,
                padding: padding,
                dilation: dilation,
                groups: groups,
                useBias: bias) :
            Conv1d(inChannels, outChannels, kernelSize,
                stride: stride,
                padding: padding,
                dilation: dilation,
                groups: groups,
                bias: bias);
        if (norm.Equals("group_norm", StringComparison.InvariantCultureIgnoreCase) || norm.Equals("time_group_norm", StringComparison.InvariantCultureIgnoreCase))
        {
            this.norm = GetNormModule(conv as Conv1d, causal, norm, normParams);
        }
    }

    /// <summary>
    /// Gets or sets the underlying convolutional layer.
    /// </summary>
    public Module<Tensor, Tensor> conv { get; set; }

    /// <summary>
    /// Gets or sets the normalization module applied after convolution.
    /// </summary>
    public Module<Tensor, Tensor> norm { get; set; }

    /// <summary>
    /// Performs the forward pass of the module, applying convolution followed by normalization.
    /// </summary>
    /// <param name="x">Input tensor of shape [B, C, T] where B is batch size, C is channels, and T is time steps.</param>
    /// <returns>Output tensor after convolution and normalization.</returns>
    /// <exception cref="ArgumentException">Thrown when input tensor does not have the expected shape.</exception>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        ValidateInputShape(x);

        var convOut = conv.forward(x);
        if (norm is not null)
        {
            var normOut = norm.forward(convOut);
            return normOut.MoveToOuterDisposeScope();
        }

        return convOut.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Disposes the managed resources used by the module.
    /// </summary>
    /// <param name="disposing">True to dispose managed resources, false otherwise.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            conv?.Dispose();
            norm?.Dispose();
        }
        base.Dispose(disposing);
    }

    /// <summary>
    /// Creates and returns the appropriate normalization module based on the specified parameters.
    /// </summary>
    /// <param name="conv">The convolution layer to normalize.</param>
    /// <param name="causal">Whether the convolution is causal.</param>
    /// <param name="norm">The type of normalization to apply.</param>
    /// <param name="normParams">Additional parameters for the normalization.</param>
    /// <returns>A normalization module compatible with the specified parameters.</returns>
    /// <exception cref="ArgumentException">Thrown when normalization parameters are invalid.</exception>
    private static Module<Tensor, Tensor> GetNormModule(
        Conv1d conv,
        bool causal,
        string norm,
        Dictionary<string, object>? normParams)
    {
        normParams ??= new Dictionary<string, object>();
        var eps = normParams.TryGetValue("eps", out var epsObj) ?
            Convert.ToSingle(epsObj) : 1e-5f;

        switch (norm.ToLowerInvariant())
        {
            case "layer_norm":
                if (conv == null)
                {
                    throw new ArgumentException("Cannot determine output channels for layer norm");
                }

                return new ConvLayerNorm((int)conv.out_channels, eps);

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

                return GroupNorm(1, conv.out_channels, eps, affine: true);

            case "weight_norm":
            case "none":
                return Identity();

            default:
                throw new ArgumentException($"Unsupported normalization: {norm}");
        }
    }

    /// <summary>
    /// Validates all input parameters for the convolution and normalization layers.
    /// </summary>
    /// <param name="inChannels">Number of input channels to validate.</param>
    /// <param name="outChannels">Number of output channels to validate.</param>
    /// <param name="kernelSize">Kernel size to validate.</param>
    /// <param name="stride">Stride value to validate.</param>
    /// <param name="padding">Padding value to validate.</param>
    /// <param name="dilation">Dilation value to validate.</param>
    /// <param name="groups">Number of groups to validate.</param>
    /// <param name="norm">Normalization type to validate.</param>
    /// <param name="bias">Bias flag to validate.</param>
    /// <param name="casual">Casual flag to validate.</param>
    /// <exception cref="ArgumentException">Thrown when any parameter is invalid.</exception>
    private static void ValidateParameters(
        int inChannels, int outChannels, int kernelSize,
        int stride, int padding, long dilation, int groups,
        string norm, bool bias, bool casual)
    {
        Debug.WriteLine($"Validating NormConv parameters: inchannels: {inChannels}, outchannels: {outChannels}, kernelSize: {kernelSize}, stride: {stride}, padding: {padding}, dilation: {dilation}, groups: {groups}, bias={bias}, casual={casual}, norm: {norm}");
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

        if (!ValidNorms.Contains(norm.ToLowerInvariant()))
        {
            throw new ArgumentException(
                $"Invalid normalization '{norm}'. Valid options are: {string.Join(", ", ValidNorms)}");
        }
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