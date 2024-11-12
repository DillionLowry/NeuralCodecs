using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch.Codecs.SNAC;

public class SNACConfig
{
    [JsonPropertyName("sampling_rate")]
    public int SamplingRate { get; set; }

    [JsonPropertyName("encoder_dim")]
    public int EncoderDim { get; set; }

    [JsonPropertyName("encoder_rates")]
    public int[] EncoderRates { get; set; }

    [JsonPropertyName("latent_dim")]
    public int? LatentDim { get; set; }

    [JsonPropertyName("decoder_dim")]
    public int DecoderDim { get; set; }

    [JsonPropertyName("decoder_rates")]
    public int[] DecoderRates { get; set; }

    [JsonPropertyName("attn_window_size")]
    public int? AttnWindowSize { get; set; }

    [JsonPropertyName("codebook_size")]
    public int CodebookSize { get; set; }

    [JsonPropertyName("codebook_dim")]
    public int CodebookDim { get; set; }

    [JsonPropertyName("vq_strides")]
    public int[] VQStrides { get; set; }

    [JsonPropertyName("noise")]
    public bool Noise { get; set; }

    [JsonPropertyName("depthwise")]
    public bool Depthwise { get; set; }

    public static SNACConfig Large { get => large; set => large = value; }
    public static SNACConfig Small { get => small; set => small = value; }

    private static SNACConfig large = new()
    {
        SamplingRate = 44100,
        EncoderDim = 64,
        EncoderRates = [3, 3, 7, 7],
        LatentDim = null,
        DecoderDim = 1536,
        DecoderRates = [7, 7, 3, 3],
        AttnWindowSize = 32,
        CodebookSize = 4096,
        CodebookDim = 8,
        VQStrides = [8, 4, 2, 1],
        Noise = true,
        Depthwise = true,
    };

    private static SNACConfig small = new()
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