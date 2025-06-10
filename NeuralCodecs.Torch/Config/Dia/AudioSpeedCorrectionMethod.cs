namespace NeuralCodecs.Torch.Config.Dia;

/// <summary>
/// Audio processing method options for speed and pitch adjustment
/// </summary>
public enum AudioSpeedCorrectionMethod
{
    /// <summary>
    /// No speed correction applied.
    /// </summary>
    None,

    /// <summary>
    /// Use TorchSharp-based linear interpolation
    /// </summary>
    TorchSharp,

    /// <summary>
    /// Hybrid speed correction combining TorchSharp and NAudio methods
    /// </summary>
    Hybrid,

    /// <summary>
    /// Use NAudio resampling for speed correction
    /// </summary>
    NAudioResampling,

    /// <summary>
    /// Create seperate outputs using all available methods (for comparison/testing)
    /// </summary>
    All
}