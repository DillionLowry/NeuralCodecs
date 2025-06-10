using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch.Config.Dia;

/// <summary>
/// Configuration for the decoder component of the Dia model.
/// </summary>
public class DecoderConfig
{
    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    [JsonPropertyName("n_layer")]
    public int NLayer { get; set; } = 18;

    /// <summary>
    /// Embedding dimension.
    /// </summary>
    [JsonPropertyName("n_embd")]
    public int NEmbedding { get; set; } = 2048;

    /// <summary>
    /// Hidden dimension size in the MLP layers.
    /// </summary>
    [JsonPropertyName("n_hidden")]
    public int NHidden { get; set; } = 8192;

    /// <summary>
    /// Number of query heads for grouped-query self-attention.
    /// </summary>
    [JsonPropertyName("gqa_query_heads")]
    public int GqaQueryHeads { get; set; } = 16;

    /// <summary>
    /// Number of key/value heads for grouped-query self-attention.
    /// </summary>
    [JsonPropertyName("kv_heads")]
    public int KvHeads { get; set; } = 4;

    /// <summary>
    /// Dimension per query head for grouped-query self-attention.
    /// </summary>
    [JsonPropertyName("gqa_head_dim")]
    public int GqaHeadDim { get; set; } = 128;

    /// <summary>
    /// Number of query heads for cross-attention.
    /// </summary>
    [JsonPropertyName("cross_query_heads")]
    public int CrossQueryHeads { get; set; } = 16;

    /// <summary>
    /// Dimension per cross-attention head.
    /// </summary>
    [JsonPropertyName("cross_head_dim")]
    public int CrossHeadDim { get; set; } = 128;

    /// <summary>
    /// Creates a new DecoderConfig instance.
    /// </summary>
    /// <param name="nLayer">Number of transformer layers.</param>
    /// <param name="nEmbedding">Embedding dimension.</param>
    /// <param name="nHidden">Hidden dimension size in the MLP layers.</param>
    /// <param name="gqaQueryHeads">Number of query heads for grouped-query self-attention.</param>
    /// <param name="kvHeads">Number of key/value heads for grouped-query self-attention.</param>
    /// <param name="gqaHeadDim">Dimension per query head for grouped-query self-attention.</param>
    /// <param name="crossQueryHeads">Number of query heads for cross-attention.</param>
    /// <param name="crossHeadDim">Dimension per cross-attention head.</param>
    public DecoderConfig(
        int nLayer = 18,
        int nEmbedding = 2048,
        int nHidden = 8192,
        int gqaQueryHeads = 16,
        int kvHeads = 4,
        int gqaHeadDim = 128,
        int crossQueryHeads = 16,
        int crossHeadDim = 128)
    {
        if (nLayer <= 0) throw new ArgumentOutOfRangeException(nameof(nLayer), "Must be greater than 0");
        if (nEmbedding <= 0) throw new ArgumentOutOfRangeException(nameof(nEmbedding), "Must be greater than 0");
        if (nHidden <= 0) throw new ArgumentOutOfRangeException(nameof(nHidden), "Must be greater than 0");
        if (gqaQueryHeads <= 0) throw new ArgumentOutOfRangeException(nameof(gqaQueryHeads), "Must be greater than 0");
        if (kvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(kvHeads), "Must be greater than 0");
        if (gqaHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(gqaHeadDim), "Must be greater than 0");
        if (crossQueryHeads <= 0) throw new ArgumentOutOfRangeException(nameof(crossQueryHeads), "Must be greater than 0");
        if (crossHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(crossHeadDim), "Must be greater than 0");

        if (gqaQueryHeads % kvHeads != 0)
        {
            throw new ArgumentException($"GQA query heads ({gqaQueryHeads}) must be divisible by KV heads ({kvHeads})");
        }

        NLayer = nLayer;
        NEmbedding = nEmbedding;
        NHidden = nHidden;
        GqaQueryHeads = gqaQueryHeads;
        KvHeads = kvHeads;
        GqaHeadDim = gqaHeadDim;
        CrossQueryHeads = crossQueryHeads;
        CrossHeadDim = crossHeadDim;
    }
}