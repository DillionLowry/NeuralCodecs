using NeuralCodecs.Core.Configuration;
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

            DeviceType.CUDA when cuda.is_available() => CUDA,

            DeviceType.CUDA => throw new InvalidOperationException("CUDA requested but not available"),

            _ => CPU
        };
    }
}