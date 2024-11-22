using NeuralCodecs.Core.Configuration;
using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch;

/// <summary>
/// Configuration class for the SNAC (Specialized Neural Audio Codec) model.
/// Contains parameters for both the encoder and decoder architectures.
/// </summary>
[method: JsonConstructor]
public class SNACConfig() : IModelConfig
{
    /// <summary>
    /// Gets or sets the compute device configuration for the model.
    /// </summary>
    [JsonIgnore]
    public DeviceConfiguration Device { get; set; } = DeviceConfiguration.CPU;

    /// <summary>
    /// Gets or sets the architecture identifier. Default is "snac".
    /// </summary>
    [JsonIgnore]
    public string Architecture { get; set; } = "snac";

    /// <summary>
    /// Gets or sets the version of the model configuration.
    /// </summary>
    [JsonIgnore]
    public string Version { get; set; } = "1.0";

    /// <summary>
    /// Gets or sets additional metadata associated with the model configuration.
    /// </summary>
    [JsonIgnore]
    public IDictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

    /// <summary>
    /// Gets or sets the audio sampling rate in Hz.
    /// </summary>
    [JsonPropertyName("sampling_rate")]
    public int SamplingRate { get; set; } = 44100;

    /// <summary>
    /// Gets or sets the dimensionality of the encoder's hidden layers.
    /// </summary>
    [JsonPropertyName("encoder_dim")]
    public int EncoderDim { get; set; } = 64;

    /// <summary>
    /// Gets or sets the stride rates for each layer in the encoder.
    /// </summary>
    [JsonPropertyName("encoder_rates")]
    public int[] EncoderRates { get; set; } = [3, 3, 7, 7];

    /// <summary>
    /// Gets or sets the dimensionality of the latent space. Null indicates automatic sizing.
    /// </summary>
    [JsonPropertyName("latent_dim")]
    public int? LatentDim { get; set; } = null;

    /// <summary>
    /// Gets or sets the dimensionality of the decoder's hidden layers.
    /// </summary>
    [JsonPropertyName("decoder_dim")]
    public int DecoderDim { get; set; } = 1536;

    /// <summary>
    /// Gets or sets the stride rates for each layer in the decoder.
    /// </summary>
    [JsonPropertyName("decoder_rates")]
    public int[] DecoderRates { get; set; } = [7, 7, 3, 3];

    /// <summary>
    /// Gets or sets the attention window size. Null disables attention.
    /// </summary>
    [JsonPropertyName("attn_window_size")]
    public int? AttnWindowSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the size of the vector quantization codebook.
    /// </summary>
    [JsonPropertyName("codebook_size")]
    public int CodebookSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the dimensionality of each codebook entry.
    /// </summary>
    [JsonPropertyName("codebook_dim")]
    public int CodebookDim { get; set; } = 8;

    /// <summary>
    /// Gets or sets the stride values for the vector quantization layers.
    /// </summary>
    [JsonPropertyName("vq_strides")]
    public int[] VQStrides { get; set; } = [8, 4, 2, 1];

    /// <summary>
    /// Gets or sets whether to apply noise during training.
    /// </summary>
    [JsonPropertyName("noise")]
    public bool Noise { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use depthwise convolutions.
    /// </summary>
    [JsonPropertyName("depthwise")]
    public bool Depthwise { get; set; } = true;

    /// <summary>
    /// Gets a predefined configuration for 44kHz audio processing.
    /// </summary>
    [JsonIgnore]
    public static SNACConfig SNAC44Khz => new();

    /// <summary>
    /// Gets a predefined configuration for 32kHz audio processing.
    /// </summary>
    [JsonIgnore]
    public static SNACConfig SNAC32Khz => new()
    {
        SamplingRate = 32000,
        EncoderDim = 64,
        EncoderRates = [2, 3, 8, 8],
        LatentDim = null,
        DecoderDim = 1536,
        DecoderRates = [8, 8, 3, 2],
        AttnWindowSize = 32,
        CodebookSize = 4096,
        CodebookDim = 8,
        VQStrides = [8, 4, 2, 1],
        Noise = true,
        Depthwise = true,
    };

    /// <summary>
    /// Gets a predefined configuration for 24kHz audio processing.
    /// </summary>
    [JsonIgnore]
    public static SNACConfig SNAC24Khz => new()
    {
        SamplingRate = 24000,
        EncoderDim = 48,
        EncoderRates = [2, 4, 8, 8],
        LatentDim = null,
        DecoderDim = 1024,
        DecoderRates = [8, 8, 4, 2],
        AttnWindowSize = null,
        CodebookSize = 4096,
        CodebookDim = 8,
        VQStrides = [4, 2, 1],
        Noise = true,
        Depthwise = true,
    };
}