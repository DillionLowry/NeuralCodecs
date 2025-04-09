using TorchSharp;

namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Parameters for short-time Fourier transform operations.
/// </summary>
public record STFTParams
{
    /// <summary>
    /// Creates a new instance of StftParams with specified parameters.
    /// </summary>
    /// <param name="windowLength">Length of the window in samples. If null, defaults to 2048.</param>
    /// <param name="hopLength">Hop length in samples. If null, defaults to windowLength/4.</param>
    /// <param name="windowType">Type of window to use. If null, defaults to "hann".</param>
    /// <param name="center">Whether to pad the signal. If null, defaults to true.</param>
    /// <param name="matchStride">Whether to match stride to hop length.</param>
    /// <param name="paddingMode">Padding mode to use. If null, defaults to "reflect".</param>
    public STFTParams(
        int? windowLength = null,
        int? hopLength = null,
        string? windowType = null,
        bool? center = null,
        bool? matchStride = null,
        PaddingModes? paddingMode = null)
    {
        WindowLength = windowLength ?? 2048;
        HopLength = hopLength ?? WindowLength / 4;
        WindowType = windowType ?? "hann";
        Center = center ?? true;
        MatchStride = matchStride ?? false;
        PaddingMode = paddingMode ?? PaddingModes.Reflect;
    }

    /// <summary>
    /// Length of the window in samples.
    /// </summary>
    public int WindowLength { get; init; }

    /// <summary>
    /// Hop length in samples.
    /// </summary>
    public int HopLength { get; init; }

    /// <summary>
    /// Type of window to use. Options: "hann", "hamming", "blackman", "bartlett", "ones", "sqrt_hann".
    /// </summary>
    public string WindowType { get; init; }

    /// <summary>
    /// Whether to pad the signal.
    /// </summary>
    public bool Center { get; init; }

    /// <summary>
    /// Gets or sets whether to match the stride to the hop length.
    /// </summary>
    public bool MatchStride { get; set; }

    /// <summary>
    /// Padding mode to use.
    /// </summary>
    public PaddingModes PaddingMode { get; init; }
}