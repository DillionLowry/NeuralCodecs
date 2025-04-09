using TorchSharp;
using static TorchSharp.torch;
namespace NeuralCodecs.Torch.Utils;

/// <summary>
/// Extension methods for audio processing operations, particularly for frequency conversions
/// between Hertz and mel scales.
/// </summary>
public static class AudioExtensions
{
    /// <summary>
    /// Gets the dimensions of the tensor as a tuple (B, C, T)
    /// </summary>
    /// <param name="tensor">The input tensor</param>
    /// <returns>A tuple containing the dimensions (B, C, T)</returns>
    public static (int B, int C, int T) GetBCTDimensions(this Tensor tensor)
    {
        if (tensor.dim() != 3)
        {
            throw new ArgumentException("Tensor must have 3 dimensions (B,C,T)");
        }
        return ((int)tensor.size(0), (int)tensor.size(1), (int)tensor.size(2));
    }

    /// <summary>
    /// Converts frequencies from Hertz to mel scale. The mel scale is a perceptual scale
    /// where equal distances in mel correspond to equal perceived distances in pitch.
    /// </summary>
    /// <param name="hertz">Tensor containing frequencies in Hertz</param>
    /// <returns>Tensor containing frequencies in mel scale</returns>
    public static Tensor HertzToMel(this Tensor hertz)
    {
        return 2595.0f * log10(1.0f + hertz / 700.0f);
    }

    /// <summary>
    /// Converts frequencies from mel scale back to Hertz.
    /// </summary>
    /// <param name="mel">Tensor containing frequencies in mel scale</param>
    /// <returns>Tensor containing frequencies in Hertz</returns>
    public static Tensor MelToHertz(this Tensor mel)
    {
        return 700.0f * (pow(10.0f, mel / 2595.0f) - 1.0f);
    }

}