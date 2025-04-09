using NeuralCodecs.Torch.Utils;
using System.Collections.Immutable;
using System.Diagnostics;
using TorchSharp.Modules;
using static NeuralCodecs.Torch.Utils.TorchUtils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// SEANet encoder implementation for audio encoding.
/// </summary>
public class SEANetEncoder : Module<Tensor, Tensor>
{
    private int[] _ratios;
    public readonly Sequential layers;

    /// <summary>
    /// Initialize SEANet encoder
    /// </summary>
    /// <param name="channels">Input audio channels</param>
    /// <param name="dimension">Intermediate representation dimension</param>
    /// <param name="nFilters">Base width for the model</param>
    /// <param name="nResidualLayers">Number of residual layers</param>
    /// <param name="ratios">Kernel size and stride ratios</param>
    /// <param name="activation">Activation function</param>
    /// <param name="activationParams">Parameters for activation function</param>
    /// <param name="norm">Normalization method</param>
    /// <param name="normParams">Parameters for normalization</param>
    /// <param name="kernelSize">Kernel size for initial convolution</param>
    /// <param name="lastKernelSize">Kernel size for final convolution</param>
    /// <param name="residualKernelSize">Kernel size for residual layers</param>
    /// <param name="dilationBase">Base for dilation increase</param>
    /// <param name="causal">Whether to use causal convolution</param>
    /// <param name="padMode">Padding mode</param>
    /// <param name="trueSkip">Whether to use true skip connection</param>
    /// <param name="compress">Compression factor for residual branches</param>
    /// <param name="lstm">Number of LSTM layers</param>
    public SEANetEncoder(
        int channels = 1,
        int dimension = 128,
        int nFilters = 32,
        int nResidualLayers = 1,
        int[] ratios = null,
        string activation = "ELU",
        Dictionary<string, object> activationParams = null,
        string norm = "weight_norm",
        Dictionary<string, object> normParams = null,
        int kernelSize = 7,
        int lastKernelSize = 7,
        int residualKernelSize = 3,
        int dilationBase = 2,
        bool causal = false,
        string padMode = "reflect",
        bool trueSkip = false,
        int compress = 2,
        int lstm = 2) : base("SEANetEncoder")
    {
        ratios ??= new[] { 8, 5, 4, 2 };
        _ratios = ratios.Reverse().ToArray();
        Dimension = dimension;

        activationParams ??= new Dictionary<string, object> { { "alpha", 1.0f } };
        normParams ??= new Dictionary<string, object>();

        ValidateParameters(dimension, nFilters, kernelSize, lastKernelSize, residualKernelSize);
        Debug.WriteLine($"SEANetEncoder: channels={channels} dimension={dimension} nFilters={nFilters} kernelSize={kernelSize} lastKernelSize={lastKernelSize} residualKernelSize={residualKernelSize}");

        var act = GetActivation(activation, activationParams);
        var mult = 1;

        var modules = new List<Module<Tensor, Tensor>>();

        // Initial convolution
        modules.Add(new SConv1d(
            channels, mult * nFilters, kernelSize, causal: causal,
            normType: norm, normParams: normParams, padMode: padMode));

        // Downsample to raw audio scale
        for (int i = 0; i < _ratios.Length; i++)
        {
            // Add residual layers
            for (int j = 0; j < nResidualLayers; j++)
            {
                modules.Add(new SEANetResnetBlock(
                    mult * nFilters,
                    kernelSizes: new[] { residualKernelSize, 1 },
                    dilations: new[] { (long)Math.Pow(dilationBase, j), 1 },
                    activation: activation,
                    activationParams: activationParams,
                    norm: norm,
                    normParams: normParams,
                    causal: causal,
                    padMode: padMode,
                    compress: compress,
                    trueSkip: trueSkip));
            }

            // Add downsampling layers
            modules.Add(act);
            modules.Add(new SConv1d(
                mult * nFilters,
                mult * nFilters * 2,
                kernelSize: _ratios[i] * 2,
                stride: _ratios[i],
                causal: causal,
                normType: norm,
                normParams: normParams,
                padMode: padMode));

            mult *= 2;
        }

        // Add LSTM layers if specified
        if (lstm > 0)
        {
            modules.Add(new SLSTM(mult * nFilters, numLayers: lstm));
        }

        // Final layers
        modules.Add(act);
        modules.Add(new SConv1d(
            mult * nFilters,
            dimension,
            lastKernelSize,
            causal: causal,
            normType: norm,
            normParams: normParams,
            padMode: padMode));

        layers = Sequential(modules);
        RegisterComponents();
    }

    public int Dimension { get; }
    public int TotalRatio => _ratios.Aggregate((a, b) => a * b); // TODO: rename hoplength

    public override Tensor forward(Tensor x)
    {
        ValidateInputShape(x);
        // Ensure input is on the same device as the model
        x = x.to(this.parameters().First().device);
        return layers.forward(x);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            layers?.Dispose();
        }
        base.Dispose(disposing);
    }

    private static void ValidateParameters(
                int dimension, int nFilters, int kernelSize,
        int lastKernelSize, int residualKernelSize)
    {
        if (dimension <= 0)
        {
            throw new ArgumentException($"Dimension must be positive, got {dimension}");
        }

        if (nFilters <= 0)
        {
            throw new ArgumentException($"Number of filters must be positive, got {nFilters}");
        }

        if (kernelSize <= 0 || kernelSize % 2 == 0)
        {
            throw new ArgumentException($"Kernel size must be positive odd number, got {kernelSize}");
        }

        if (lastKernelSize <= 0 || lastKernelSize % 2 == 0)
        {
            throw new ArgumentException($"Last kernel size must be positive odd number, got {lastKernelSize}");
        }

        if (residualKernelSize <= 0 || residualKernelSize % 2 == 0)
        {
            throw new ArgumentException($"Residual kernel size must be positive odd number, got {residualKernelSize}");
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