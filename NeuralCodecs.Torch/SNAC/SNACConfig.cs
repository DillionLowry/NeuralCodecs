using NeuralCodecs.Core.Configuration;
using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch;

[method: JsonConstructor]
public class SNACConfig() : IModelConfig
{
    [JsonIgnore]
    public DeviceConfiguration Device { get; set; } = DeviceConfiguration.CPU;

    [JsonIgnore]
    public string Architecture { get; set; } = "snac";

    [JsonIgnore]
    public string Version { get; set; } = "1.0";

    [JsonIgnore]
    public IDictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

    [JsonPropertyName("sampling_rate")]
    public int SamplingRate { get; set; } = 44100;

    [JsonPropertyName("encoder_dim")]
    public int EncoderDim { get; set; } = 64;

    [JsonPropertyName("encoder_rates")]
    public int[] EncoderRates { get; set; } = [3, 3, 7, 7];

    [JsonPropertyName("latent_dim")]
    public int? LatentDim { get; set; } = null;

    [JsonPropertyName("decoder_dim")]
    public int DecoderDim { get; set; } = 1536;

    [JsonPropertyName("decoder_rates")]
    public int[] DecoderRates { get; set; } = [7, 7, 3, 3];

    [JsonPropertyName("attn_window_size")]
    public int? AttnWindowSize { get; set; } = 32;

    [JsonPropertyName("codebook_size")]
    public int CodebookSize { get; set; } = 4096;

    [JsonPropertyName("codebook_dim")]
    public int CodebookDim { get; set; } = 8;

    [JsonPropertyName("vq_strides")]
    public int[] VQStrides { get; set; } = [8, 4, 2, 1];

    [JsonPropertyName("noise")]
    public bool Noise { get; set; } = true;

    [JsonPropertyName("depthwise")]
    public bool Depthwise { get; set; } = true;

    [JsonIgnore]
    public static SNACConfig SNAC44Khz => new();

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