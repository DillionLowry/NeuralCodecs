using TorchSharp;

namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Parameters for Short-Time Fourier Transform (STFT) operations.
/// </summary>
public class STFTParams
{
    /// <summary>
    /// Gets or sets the window length in samples.
    /// </summary>
    public int WindowLength { get; set; }

    /// <summary>
    /// Gets or sets the hop length in samples.
    /// </summary>
    public int HopLength { get; set; }

    // TODO: Add WindowType enum
    /// <summary>
    /// Gets or sets the window type (e.g., hann, hamming).
    /// </summary>
    public string WindowType { get; set; }

    /// <summary>
    /// Gets or sets whether to match the stride to the hop length.
    /// </summary>
    public bool MatchStride { get; set; }

    /// <summary>
    /// Gets or sets the padding type for STFT.
    /// </summary>
    public PaddingModes PaddingMode { get; set; }

    /// <summary>
    /// Initializes a new instance of the STFTParams class.
    /// </summary>
    /// <param name="windowLength">Window length in samples.</param>
    /// <param name="hopLength">Hop length in samples.</param>
    /// <param name="windowType">Type of window function.</param>
    /// <param name="matchStride">Whether to match stride to hop length.</param>
    /// <param name="paddingType">Type of padding to apply.</param>
    public STFTParams(
        int? windowLength = null,
        int? hopLength = null,
        string windowType = null,
        bool? matchStride = null,
        PaddingModes? paddingType = null)
    {
        WindowLength = windowLength ?? 0;
        HopLength = hopLength ?? 0;
        WindowType = windowType;
        MatchStride = matchStride ?? false;
        PaddingMode = paddingType ?? PaddingModes.Reflect;
    }
}