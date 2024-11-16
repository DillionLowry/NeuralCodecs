using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Utils;
using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch;

[method: JsonConstructor]
public class SNACConfig() : ModelConfig
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

    [JsonIgnore]
    public static SNACConfig Large { get => large; set => large = value; }

    [JsonIgnore]
    public static SNACConfig Small { get => small; set => small = value; }

    [JsonIgnore]
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

    [JsonIgnore]
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

    [JsonIgnore]
    private List<string> _errors = new();

    public override bool Validate()
    {
        return GetErrors().Count <= 0;
    }

    public virtual List<string> GetErrors()
    {
        _errors = new List<string>();

        if (SamplingRate <= 0)
            _errors.Add($"Invalid sampling rate: {SamplingRate}");

        if (EncoderDim <= 0)
            _errors.Add($"Invalid encoder dimension: {EncoderDim}");

        if (DecoderDim <= 0)
            _errors.Add($"Invalid decoder dimension: {DecoderDim}");

        if (EncoderRates.IsNullOrEmpty())
            _errors.Add("Missing encoder rates");
        else if (EncoderRates.Any(r => r <= 0))
            _errors.Add("Invalid encoder rate values");

        if (DecoderRates.IsNullOrEmpty())
            _errors.Add("Missing decoder rates");
        else if (DecoderRates.Any(r => r <= 0))
            _errors.Add("Invalid decoder rate values");

        if (AttnWindowSize <= 0)
            _errors.Add($"Invalid attention window size: {AttnWindowSize}");

        if (CodebookSize <= 0)
            _errors.Add($"Invalid codebook size: {CodebookSize}");

        if (CodebookDim <= 0)
            _errors.Add($"Invalid codebook dimension: {CodebookDim}");

        if (VQStrides.IsNullOrEmpty())
            _errors.Add("Missing VQ strides");
        else if (VQStrides.Any(s => s <= 0))
            _errors.Add("Invalid VQ stride values");

        return _errors;
    }
}