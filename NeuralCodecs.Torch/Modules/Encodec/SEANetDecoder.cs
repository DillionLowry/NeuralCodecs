using System.Diagnostics;
using TorchSharp.Modules;
using static NeuralCodecs.Torch.Utils.TorchUtils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// SEANet decoder implementation for audio decoding.
/// </summary>
public class SEANetDecoder : Module<Tensor, Tensor>
{
    private readonly int[] _ratios;
    private readonly Sequential layers;

    /// <summary>
    /// Initialize SEANet decoder.
    /// </summary>
    /// <param name="channels">Output audio channels</param>
    /// <param name="dimension">Input dimension from encoder</param>
    /// <param name="nFilters">Base width for the model</param>
    /// <param name="nResidualLayers">Number of residual layers</param>
    /// <param name="ratios">Upsampling ratios</param>
    /// <param name="activation">Activation function</param>
    /// <param name="activationParams">Parameters for activation</param>
    /// <param name="finalActivation">Optional final activation</param>
    /// <param name="finalActivationParams">Parameters for final activation</param>
    /// <param name="norm">Normalization method</param>
    /// <param name="normParams">Parameters for normalization</param>
    /// <param name="kernelSize">Initial kernel size</param>
    /// <param name="lastKernelSize">Final kernel size</param>
    /// <param name="residualKernelSize">Kernel size for residual layers</param>
    /// <param name="dilationBase">Base for dilation increase</param>
    /// <param name="causal">Whether to use causal convolution</param>
    /// <param name="padMode">Padding mode</param>
    /// <param name="trueSkip">Whether to use true skip connections</param>
    /// <param name="compress">Compression factor for residual branches</param>
    /// <param name="lstm">Number of LSTM layers</param>
    /// <param name="trimRightRatio">Ratio for right padding trimming</param>
    public SEANetDecoder(
        int channels = 1,
        int dimension = 128,
        int nFilters = 32,
        int nResidualLayers = 1,
        int[] ratios = null,
        string activation = "ELU",
        Dictionary<string, object> activationParams = null,
        string finalActivation = null,
        Dictionary<string, object> finalActivationParams = null,
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
        int lstm = 2,
        float trimRightRatio = 1.0f) : base("SEANetDecoder")
    {
        ratios ??= new[] { 8, 5, 4, 2 };
        _ratios = ratios;

        ValidateParameters(
            dimension, channels, nFilters, kernelSize,
            lastKernelSize, residualKernelSize, trimRightRatio);

        activationParams ??= new Dictionary<string, object> { { "alpha", 1.0f } };
        normParams ??= new Dictionary<string, object>();

        var act = GetActivation(activation, activationParams);
        var mult = (int)Math.Pow(2, ratios.Length);

        var modules = new List<Module<Tensor, Tensor>>
        {
            new SConv1d(dimension, mult * nFilters, kernelSize,
                causal: causal, normType: norm,
                normParams: normParams, padMode: padMode)
        };

        // Add LSTM if specified
        if (lstm > 0)
        {
            modules.Add(new SLSTM(mult * nFilters, numLayers: lstm));
        }

        // Upsample to raw audio scale
        for (int i = 0; i < _ratios.Length; i++)
        {
            // Add upsampling layers
            modules.Add(act);
            modules.Add(new SConvTranspose1d(
                mult * nFilters,
                mult * nFilters / 2,
                kernelSize: _ratios[i] * 2,
                stride: _ratios[i],
                causal: causal,
                normType: norm,
                normParams: normParams,
                trimRightRatio: trimRightRatio));

            // Add residual layers
            Debug.WriteLine($"SEANetDecoder residual layers: {nResidualLayers}");
            for (int j = 0; j < nResidualLayers; j++)
            {
                modules.Add(new SEANetResnetBlock(
                    mult * nFilters / 2,
                    kernelSizes: new[] { residualKernelSize, 1 },
                    dilations: new long[] { (long)Math.Pow(dilationBase, j), 1 },
                    activation: activation,
                    activationParams: activationParams,
                    norm: norm,
                    normParams: normParams,
                    causal: causal,
                    padMode: padMode,
                    compress: compress,
                    trueSkip: trueSkip));
            }

            mult /= 2;
        }

        // Add final layers
        modules.Add(act);
        modules.Add(new SConv1d(
            nFilters,
            channels,
            lastKernelSize,
            causal: causal,
            normType: norm,
            normParams: normParams,
            padMode: padMode));

        // Add optional final activation
        if (!string.IsNullOrEmpty(finalActivation))
        {
            modules.Add(GetActivation(
                finalActivation,
                finalActivationParams ?? new Dictionary<string, object>()));
        }

        layers = Sequential(modules);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        ValidateInputShape(x);
        // Ensure input is on the same device as the model
        x = x.contiguous().to(float32).to(this.parameters().First().device);
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
                int dimension, int channels, int nFilters,
        int kernelSize, int lastKernelSize, int residualKernelSize,
        float trimRightRatio)
    {
        if (dimension <= 0)
        {
            throw new ArgumentException($"Dimension must be positive, got {dimension}");
        }

        if (channels <= 0)
        {
            throw new ArgumentException($"Channels must be positive, got {channels}");
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

        if (trimRightRatio is < 0 or > 1)
        {
            throw new ArgumentException($"Trim right ratio must be between 0 and 1, got {trimRightRatio}");
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