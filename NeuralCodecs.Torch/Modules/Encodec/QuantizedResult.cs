using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Represents the result of audio quantization in the Encodec neural codec,
/// containing the quantized audio data, codebook indices, bandwidth metrics and optional penalties.
/// </summary>
public record QuantizedResult
{
    /// <summary>
    /// Gets the quantized audio tensor after encoding and quantization.
    /// Shape: [batch, channels, time]
    /// </summary>
    public required Tensor Quantized { get; init; }

    /// <summary>
    /// Gets the codebook indices tensor representing the quantized audio.
    /// Shape: [batch, num_codebooks, time]
    /// </summary>
    public required Tensor Codes { get; init; }

    /// <summary>
    /// Gets the bandwidth used for each batch item in kilobits per second (kb/s).
    /// Shape: [batch]
    /// </summary>
    public required Tensor Bandwidth { get; init; }

    /// <summary>
    /// Gets an optional penalty tensor that can be used during training.
    /// May contain commitment loss, codebook usage penalties, etc.
    /// </summary>
    public Tensor? Penalty { get; init; }

    /// <summary>
    /// Gets a dictionary of additional metrics computed during quantization.
    /// May include codebook usage statistics, perplexity measures, or other diagnostic information.
    /// </summary>
    public Dictionary<string, object> Metrics { get; init; } = new();
}