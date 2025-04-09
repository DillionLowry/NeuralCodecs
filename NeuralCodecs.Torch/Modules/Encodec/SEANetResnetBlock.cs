using static NeuralCodecs.Torch.Utils.TorchUtils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Residual block for SEANet model
/// </summary>
public class SEANetResnetBlock : Module<Tensor, Tensor>
{
    public readonly Module<Tensor, Tensor> shortcut;
    private readonly Module<Tensor, Tensor> block;

    /// <summary>
    /// Initialize a SEANet residual block with the given parameters
    /// </summary>
    /// <param name="dimension">Input/output dimension</param>
    /// <param name="kernelSizes">Kernel sizes for each conv layer</param>
    /// <param name="dilations">Dilation factors for each conv layer</param>
    /// <param name="activation">Activation function name</param>
    /// <param name="activationParams">Parameters for activation function</param>
    /// <param name="norm">Normalization method</param>
    /// <param name="normParams">Parameters for normalization</param>
    /// <param name="causal">Whether to use causal convolutions</param>
    /// <param name="padMode">Padding mode</param>
    /// <param name="compress">Compression factor for hidden dimension</param>
    /// <param name="trueSkip">Whether to use true skip connection</param>
    public SEANetResnetBlock(
        int dimension,
        int[]? kernelSizes = null,
        long[]? dilations = null,
        string activation = "ELU",
        Dictionary<string, object>? activationParams = null,
        string norm = "weight_norm",
        Dictionary<string, object>? normParams = null,
        bool causal = false,
        string padMode = "reflect",
        int compress = 2,
        bool trueSkip = true) : base($"SEANetResnetBlock")
    {
        kernelSizes ??= new[] { 3, 1 };
        dilations ??= new long[] { 1, 1 };
        activationParams ??= new Dictionary<string, object> { { "alpha", 1.0f } };
        normParams ??= new Dictionary<string, object>();

        ValidateParameters(dimension, kernelSizes, dilations, compress);

        var act = GetActivation(activation, activationParams);
        var hidden = dimension / compress;
        var blockLayers = new List<Module<Tensor, Tensor>>();

        for (int i = 0; i < kernelSizes.Length; i++)
        {
            var inChs = i == 0 ? dimension : hidden;
            var outChs = i == kernelSizes.Length - 1 ? dimension : hidden;

            blockLayers.Add(act);
            blockLayers.Add(new SConv1d(
                inChs, outChs,
                kernelSize: kernelSizes[i],
                dilation: dilations[i],
                causal: causal,
                normType: norm,
                normParams: normParams,
                padMode: padMode));
        }

        block = Sequential(blockLayers);
        shortcut = trueSkip ?
            Identity() as Module<Tensor, Tensor> :
            new SConv1d(dimension, dimension, 1, normType: norm,
                normParams: normParams, causal: causal, padMode: padMode);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();

        var shortcutOut = shortcut.forward(x);
        var blockOut = block.forward(x);

        return add(shortcutOut, blockOut).MoveToOuterDisposeScope();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            block?.Dispose();
            shortcut?.Dispose();
        }
        base.Dispose(disposing);
    }

    private static void ValidateParameters(
                int dim, int[] kernelSizes, long[] dilations, int compress)
    {
        if (dim <= 0)
        {
            throw new ArgumentException($"Dimension must be positive, got {dim}");
        }

        if (compress <= 0)
        {
            throw new ArgumentException($"Compress factor must be positive, got {compress}");
        }

        if (kernelSizes.Length != dilations.Length)
        {
            throw new ArgumentException(
                "Number of kernel sizes must match number of dilations, " +
                $"got {kernelSizes.Length} vs {dilations.Length}");
        }

        if (kernelSizes.Any(k => k <= 0))
        {
            throw new ArgumentException(
                $"Kernel sizes must be positive, got [{string.Join(", ", kernelSizes)}]");
        }

        if (dilations.Any(d => d <= 0))
        {
            throw new ArgumentException(
                $"Dilations must be positive, got [{string.Join(", ", dilations)}]");
        }
    }
}