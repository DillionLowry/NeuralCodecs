using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch.Config.Dia;

/// <summary>
/// Main configuration container for the Dia model architecture.
/// </summary>
public class DiaModelConfig
{
    /// <summary>
    /// Configuration for the encoder component.
    /// </summary>
    [JsonPropertyName("encoder")]
    public EncoderConfig Encoder { get; set; } = new EncoderConfig();

    /// <summary>
    /// Configuration for the decoder component.
    /// </summary>
    [JsonPropertyName("decoder")]
    public DecoderConfig Decoder { get; set; } = new DecoderConfig();

    /// <summary>
    /// Size of the source (text) vocabulary.
    /// </summary>
    [JsonPropertyName("src_vocab_size")]
    public int SrcVocabSize { get; set; } = 256;

    /// <summary>
    /// Size of the target (audio code) vocabulary.
    /// </summary>
    [JsonPropertyName("tgt_vocab_size")]
    public int TgtVocabSize { get; set; } = 1028;

    /// <summary>
    /// Dropout probability applied within the model.
    /// </summary>
    [JsonPropertyName("dropout")]
    public float Dropout { get; set; } = 0.0f;

    /// <summary>
    /// Epsilon value for normalization layers (e.g., LayerNorm).
    /// </summary>
    [JsonPropertyName("normalization_layer_epsilon")]
    public float NormalizationLayerEpsilon { get; set; } = 1.0e-5f;

    /// <summary>
    /// Data type for model weights (e.g., "float32", "bfloat16").
    /// </summary>
    [JsonPropertyName("weight_dtype")]
    public string WeightDtype { get; set; } = "float32";

    /// <summary>
    /// Minimum timescale for Rotary Positional Embeddings (RoPE).
    /// </summary>
    [JsonPropertyName("rope_min_timescale")]
    public int RopeMinTimescale { get; set; } = 1;

    /// <summary>
    /// Maximum timescale for Rotary Positional Embeddings (RoPE).
    /// </summary>
    [JsonPropertyName("rope_max_timescale")]
    public int RopeMaxTimescale { get; set; } = 10_000;

    /// <summary>
    /// Creates a new DiaModelConfig instance.
    /// </summary>
    /// <param name="encoder">Configuration for the encoder component.</param>
    /// <param name="decoder">Configuration for the decoder component.</param>
    /// <param name="srcVocabSize">Size of the source (text) vocabulary.</param>
    /// <param name="tgtVocabSize">Size of the target (audio code) vocabulary.</param>
    /// <param name="dropout">Dropout probability applied within the model.</param>
    /// <param name="normalizationLayerEpsilon">Epsilon value for normalization layers.</param>
    /// <param name="weightDtype">Data type for model weights.</param>
    /// <param name="ropeMinTimescale">Minimum timescale for Rotary Positional Embeddings.</param>
    /// <param name="ropeMaxTimescale">Maximum timescale for Rotary Positional Embeddings.</param>
    public DiaModelConfig(
        EncoderConfig encoder = null,
        DecoderConfig decoder = null,
        int srcVocabSize = 256,
        int tgtVocabSize = 1028,
        float dropout = 0.0f,
        float normalizationLayerEpsilon = 1.0e-5f,
        string weightDtype = "float32",
        int ropeMinTimescale = 1,
        int ropeMaxTimescale = 10_000)
    {
        Encoder = encoder ?? new();
        Decoder = decoder ?? new();

        if (srcVocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(srcVocabSize), "Must be greater than 0");
        if (tgtVocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(tgtVocabSize), "Must be greater than 0");
        if (dropout < 0.0f || dropout >= 1.0f) throw new ArgumentOutOfRangeException(nameof(dropout), "Must be in range [0, 1)");
        if (normalizationLayerEpsilon <= 0.0f) throw new ArgumentOutOfRangeException(nameof(normalizationLayerEpsilon), "Must be greater than 0");

        SrcVocabSize = srcVocabSize;
        TgtVocabSize = tgtVocabSize;
        Dropout = dropout;
        NormalizationLayerEpsilon = normalizationLayerEpsilon;
        WeightDtype = weightDtype;
        RopeMinTimescale = ropeMinTimescale;
        RopeMaxTimescale = ropeMaxTimescale;
    }
}