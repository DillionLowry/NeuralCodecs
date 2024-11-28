using NAudio.Codecs;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Torch.Config.SNAC;
using NeuralCodecs.Torch.Modules.DAC;
using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch.Config.DAC;

public class DACConfig: IModelConfig
{
    public DeviceConfiguration Device { get; set; } = DeviceConfiguration.CPU;

    public IDictionary<string, string> Metadata { get; set; }
    public string ModelBitrate { get; set; } = "8kbps";
    public string ModelType { get; set; } = "44khz";

    public string Version { get; set; } = "0.0.1";
    /// <summary>
    /// Gets or sets the architecture identifier. Default is "dac".
    /// </summary>
    [JsonPropertyName("model_type")]
    public string Architecture { get; set; } = "dac";

    // TODO: Double-check these defaults
    /// <summary>
    /// Gets or sets the dimensionality of each codebook entry.
    /// </summary>
    [JsonPropertyName("codebook_dim")]
    public int CodebookDim { get; set; } = 8;

    [JsonPropertyName("codebook_loss_weight")]
    public float CodebookLossWeight { get; set; } = 1.0f;

    /// <summary>
    /// Gets or sets the size of the vector quantization codebook.
    /// </summary>
    [JsonPropertyName("codebook_size")]
    public int CodebookSize { get; set; } = 1024;

    [JsonPropertyName("commitment_loss_weight")]
    public float CommitmentLossWeight { get; set; } = 0.25f;

    /// <summary>
    /// Gets or sets the dimensionality of the decoder's hidden layers.
    /// </summary>
    [JsonPropertyName("decoder_hidden_size")]
    public int DecoderDim { get; set; } = 1536;

    /// <summary>
    /// Gets or sets the stride rates for each layer in the decoder. AKA "upsampling rates"
    /// </summary>
    [JsonPropertyName("upsampling_ratios")]
    public int[] DecoderRates { get; set; } = [8, 8, 4, 2];

    /// <summary>
    /// Gets or sets the dimensionality of the encoder's hidden layers.
    /// </summary>
    [JsonPropertyName("encoder_hidden_size")]
    public int EncoderDim { get; set; } = 64;

    /// <summary>
    /// Gets or sets the stride rates for each layer in the encoder. AKA "downsampling rates"
    /// </summary>
    [JsonPropertyName("downsampling_ratios")]
    public int[] EncoderRates { get; set; } = [2, 4, 8, 8];

    [JsonPropertyName("hop_length")]
    public int HopLength { get; set; } = 512;

    [JsonPropertyName("n_codebooks")]
    public int NumCodebooks { get; set; } = 9;

    [JsonPropertyName("quantizer_dropout")]
    public float QuantizerDropout { get; set; } = 0.0f;

    /// <summary>
    /// Gets or sets the audio sampling rate in Hz.
    /// </summary>
    [JsonPropertyName("sampling_rate")]
    public int SamplingRate { get; set; } = 44100;

    public string Tag { get; set; } = "latest";
    [JsonPropertyName("torch_dtype")]
    public string TorchDataType { get; set; } = "float32";

    [JsonPropertyName("transformers_version")]
    public string TransformersVersion { get; set; }


    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the dimensionality of the latent space. Null indicates automatic sizing.
    /// </summary>
    [JsonPropertyName("latent_dim")]
    public int? LatentDim { get; set; } = null;

    [JsonIgnore]
    public static DACConfig DAC44khz => new();

    // TODO
    //[JsonIgnore]
    //public static DACConfig DAC44khz16kbps => new()
    //{
    //    ModelBitrate = "16kbps",
    //    ModelType = "44khz",
    //    NumCodebooks = 18,
    //    LatentDim = 128,
    //    Version = "1.0.0"
    //};

    [JsonIgnore]
    public static DACConfig DAC24khz => new()
    {
        SamplingRate = 24000,
        ModelType = "24khz",
        NumCodebooks = 32,
        EncoderRates = new[] { 2, 4, 5, 8 },
        DecoderRates = new[] { 8, 5, 4, 2 },
        Version = "0.0.4"
    };

    [JsonIgnore]
    public static DACConfig DAC16khz => new()
    {
        SamplingRate = 16000,
        ModelType = "16khz",
        NumCodebooks = 12,
        EncoderRates = new[] { 2, 4, 5, 8 },
        DecoderRates = new[] { 8, 5, 4, 2 },
        Version = "0.0.5"
    };



    public class MultiScaleSTFTLossConfig
    {
        public int[] WindowLengths { get; set; } = [2048, 512];
    }
    public class MelSpectrogramLossConfig
    {
        public int[] NMels { get; set; } = [5, 10, 20, 40, 80, 160, 320];
        public int[] WindowLengths { get; set; } = [32, 64, 128, 256, 512, 1024, 2048];
        public int[] MelFMin { get; set; } = [0, 0, 0, 0, 0, 0, 0];
        public int?[] MelFMax { get; set; } = [null, null, null, null, null, null, null];
        public float MelPow { get; set; } = 1.0f;
        public float ClampEps { get; set; } = 1.0e-5f;
        public float MagWeight { get; set; } = 0.0f;
    }
    public class DiscriminatorConfig
    {
        public int SampleRate { get; set; } = 44100;
        public int[] Rates { get; set; }
        public int[] Periods { get; set; } = [2, 3, 5, 7, 11];
        public int[] FFTLengths { get; set; } = [2048, 1024, 512];
        public float[][] Bands { get; set; } = new[]
        {
            new[] { 0.0f, 0.1f },
            new[] { 0.1f, 0.25f },
            new[] { 0.25f, 0.5f },
            new[] { 0.5f, 0.75f },
            new[] { 0.75f, 1.0f }
        };
    }
}