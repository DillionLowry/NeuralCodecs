using NeuralCodecs.Core.Configuration;
using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch.Config.Encodec;

public class EncodecConfig : IModelConfig
{
    [JsonIgnore]
    public static EncodecConfig Encodec24Khz => new()
    {
        SampleRate = 24000,
        Channels = 1,
        CodebookSize = 1024,
        CodebookDim = 128,
        HiddenSize = 128,
        Compress = 2,
        DilationGrowthRate = 2,
        KernelSize = 7,
        LastKernelSize = 7,
        ModelType = "encodec",
        NormType = "weight_norm",
        Normalize = false,
        NumFilters = 32,
        NumLstmLayers = 2,
        NumResidualLayers = 1,
        PadMode = "reflect",
        ResidualKernelSize = 3,
        TorchDtype = "float32",
        TransformersVersion = "4.31.0.dev0",
        TrimRightRatio = 1.0f,
        TargetBandwidths = new[] { 1.5f, 3.0f, 6.0f, 12.0f, 24.0f },
        UpsamplingRatios = new[] { 8, 5, 4, 2 },
        UseCausalConv = true
    };

    [JsonIgnore]
    public static EncodecConfig Encodec48Khz => new()
    {
        Channels = 2,
        ChunkLengthSeconds = 1.0f,
        CodebookDim = 128,
        CodebookSize = 1024,
        Compress = 2,
        DilationGrowthRate = 2,
        HiddenSize = 128,
        KernelSize = 7,
        LastKernelSize = 7,
        NormType = "time_group_norm",
        Normalize = true,
        NumFilters = 32,
        NumLstmLayers = 2,
        NumResidualLayers = 1,
        Overlap = 0.01f,
        PadMode = "reflect",
        ResidualKernelSize = 3,
        TorchDtype = "float32",
        TransformersVersion = "4.31.0.dev0",
        TrimRightRatio = 1.0f,
        Bandwidth = 6.0f,
        TargetBandwidths = new[] { 3.0f, 6.0f, 12.0f, 24.0f },
        UpsamplingRatios = new[] { 8, 5, 4, 2 },
        SampleRate = 48000,
        UseCausalConv = false
    };

    [JsonIgnore]
    public string Architecture { get; set; } = "encodec";

    public float? Bandwidth { get; set; } = 6.0f;

    [JsonPropertyName("channels")]
    public int Channels { get; set; } = 1;

    [JsonPropertyName("chunk_length_s")]
    public float? ChunkLengthSeconds { get; set; } = null;

    [JsonPropertyName("codebook_dim")]
    public int CodebookDim { get; set; } = 128;

    [JsonPropertyName("codebook_size")]
    public int CodebookSize { get; set; } = 1024;

    [JsonPropertyName("compress")]
    public int Compress { get; set; } = 2;

    [JsonIgnore]
    public DeviceConfiguration Device { get; set; } = DeviceConfiguration.CPU;

    [JsonPropertyName("dilation_growth_rate")]
    public int DilationGrowthRate { get; set; } = 2;

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 128;

    [JsonPropertyName("kernel_size")]
    public int KernelSize { get; set; } = 7;

    [JsonPropertyName("last_kernel_size")]
    public int LastKernelSize { get; set; } = 7;

    [JsonIgnore]
    public IDictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

    [JsonPropertyName("model_type")]
    public string ModelType { get; set; } = "encodec";

    [JsonPropertyName("normalize")]
    public bool Normalize { get; set; } = false;

    [JsonPropertyName("norm_type")]
    public string NormType { get; set; } = "weight_norm";

    [JsonPropertyName("num_filters")]
    public int NumFilters { get; set; } = 32;

    [JsonPropertyName("num_lstm_layers")]
    public int NumLstmLayers { get; set; } = 2;

    [JsonPropertyName("num_residual_layers")]
    public int NumResidualLayers { get; set; } = 1;

    [JsonPropertyName("overlap")]
    public float? Overlap { get; set; } = null;

    [JsonPropertyName("pad_mode")]
    public string PadMode { get; set; } = "reflect";

    [JsonPropertyName("residual_kernel_size")]
    public int ResidualKernelSize { get; set; } = 3;

    [JsonPropertyName("sampling_rate")]
    public int SampleRate { get; set; } = 24000;

    [JsonPropertyName("target_bandwidths")]
    public float[] TargetBandwidths { get; set; } = new[] { 1.5f, 3.0f, 6.0f, 12.0f, 24.0f };

    [JsonPropertyName("torch_dtype")]
    public string TorchDtype { get; set; } = "float32";

    [JsonPropertyName("transformers_version")]
    public string TransformersVersion { get; set; } = "4.31.0.dev0";

    [JsonPropertyName("trim_right_ratio")]
    public float TrimRightRatio { get; set; } = 1.0f;

    [JsonPropertyName("upsampling_ratios")]
    public int[] UpsamplingRatios { get; set; } = new[] { 8, 5, 4, 2 };

    [JsonPropertyName("use_causal_conv")]
    public bool UseCausalConv { get; set; } = true;

    [JsonIgnore]
    public string Version { get; set; } = "1.0.0";
}