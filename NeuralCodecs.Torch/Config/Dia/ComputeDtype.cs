namespace NeuralCodecs.Torch.Config.Dia;

/// <summary>
/// Compute data types for model inference.
/// </summary>
public enum ComputeDtype
{
    /// <summary>
    /// 32-bit floating point
    /// </summary>
    Float32,

    /// <summary>
    /// 16-bit floating point
    /// </summary>
    Float16,

    /// <summary>
    /// Brain floating point (16-bit)
    /// </summary>
    BFloat16
}