using NeuralCodecs.Core.Configuration;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using DeviceType = NeuralCodecs.Core.Configuration.DeviceType;

namespace NeuralCodecs.Torch.Utils;

public static class TorchUtils
{
    /// <summary>
    /// Create activation function from name and parameters
    /// </summary>
    public static Module<Tensor, Tensor> GetActivation(
        string name, Dictionary<string, object> parameters)
    {
        // Handle common activation functions
        switch (name.ToUpperInvariant())
        {
            case "RELU":
                return ReLU();

            case "GELU":
                return GELU();

            case "ELU":
                var alpha = parameters.TryGetValue("alpha", out var alphaObj)
                    ? Convert.ToSingle(alphaObj)
                    : 1.0f;
                return ELU(alpha);

            case "LEAKYRELU":
                var negativeSlope = parameters.TryGetValue("negative_slope", out var nsObj)
                    ? Convert.ToSingle(nsObj)
                    : 0.01f;
                return LeakyReLU(negativeSlope);

            case "TANH":
                return Tanh();

            case "SIGMOID":
                return Sigmoid();

            default:
                throw new ArgumentException($"Unsupported activation function: {name}");
        }
    }

    /// <summary>
    /// Performs interpolation on tensors.
    /// </summary>
    /// <param name="x">Input tensor for interpolation points</param>
    /// <param name="xp">Known data points</param>
    /// <param name="fp">Function values at known data points</param>
    /// <param name="dim">Dimension along which to interpolate</param>
    /// <param name="linear">Whether to use linear interpolation</param>
    /// <returns>Interpolated values</returns>
    public static Tensor Interp(Tensor x, Tensor xp, Tensor fp, long dim = -1, bool linear = true)
    {
        using var scope = torch.NewDisposeScope();
        // Move the interpolation dimension to the last axis
        x = x.movedim([dim], [-1]);
        xp = xp.movedim([dim], [-1]);
        fp = fp.movedim([dim], [-1]);
        var offset = torch.diff(fp) / torch.diff(xp);
        var slope = fp[TensorIndex.Ellipsis, ..^1] - (offset * xp[TensorIndex.Ellipsis, ..^1]);
        var indices = torch.searchsorted(xp, x, right: false);

        if (linear)
        {
            indices = torch.clamp(indices - 1, 0, offset.shape[^1] - 1);
        }
        else // constant
        {
            // Pad m and b to get constant values outside of xp range
            offset = torch.cat([torch.zeros_like(offset)[TensorIndex.Ellipsis, ..1], offset, torch.zeros_like(offset)[TensorIndex.Ellipsis, ..1]], dim: -1);
            slope = torch.cat([fp[TensorIndex.Ellipsis, ..1], slope, fp[TensorIndex.Ellipsis, ^1..]], dim: -1);
        }

        var values = (offset.gather(-1, indices) * x) + slope.gather(-1, indices);
        return values.movedim([-1], [dim]).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Gets the torch device based on the provided device configuration.
    /// </summary>
    /// <param name="device">The device configuration.</param>
    /// <returns>The torch device.</returns>
    /// <exception cref="InvalidOperationException">Thrown when CUDA is requested but not available.</exception>
    public static Device GetDevice(DeviceConfiguration device)
    {
        return device?.Type switch

        {
            DeviceType.CPU => CPU,

            DeviceType.CUDA when cuda.is_available() => new Device("CUDA", device.Index),

            DeviceType.CUDA => throw new InvalidOperationException("CUDA requested but not available"),

            _ => CPU
        };
    }
}