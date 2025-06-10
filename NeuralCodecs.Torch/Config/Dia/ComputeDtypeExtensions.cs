using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Config.Dia;

/// <summary>
/// Extensions for ComputeDtype enum.
/// </summary>
public static class ComputeDtypeExtensions
{
    /// <summary>
    /// Converts ComputeDtype to TorchSharp ScalarType.
    /// </summary>
    /// <param name="dtype">Compute data type</param>
    /// <returns>TorchSharp ScalarType</returns>
    public static ScalarType ToTorchDtype(this ComputeDtype dtype)
    {
        return dtype switch
        {
            ComputeDtype.Float32 => ScalarType.Float32,
            ComputeDtype.Float16 => ScalarType.Float16,
            ComputeDtype.BFloat16 => ScalarType.BFloat16,
            _ => throw new ArgumentException($"Unsupported compute dtype: {dtype}")
        };
    }
}