using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch.Config.Dia;

/// <summary>
/// Configuration for the encoder component of the Dia model.
/// </summary>
public class EncoderConfig
{
    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    [JsonPropertyName("n_layer")]
    public int NLayer { get; set; } = 12;

    /// <summary>
    /// Embedding dimension.
    /// </summary>
    [JsonPropertyName("n_embd")]
    public int NEmbedding { get; set; } = 1024;

    /// <summary>
    /// Hidden dimension size in the MLP layers.
    /// </summary>
    [JsonPropertyName("n_hidden")]
    public int NHidden { get; set; } = 4096;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    [JsonPropertyName("n_head")]
    public int NHead { get; set; } = 16;

    /// <summary>
    /// Dimension per attention head.
    /// </summary>
    [JsonPropertyName("head_dim")]
    public int HeadDim { get; set; } = 128;

    /// <summary>
    /// Creates a new EncoderConfig instance.
    /// </summary>
    /// <param name="nLayer">Number of transformer layers.</param>
    /// <param name="nEmbedding">Embedding dimension.</param>
    /// <param name="nHidden">Hidden dimension size in the MLP layers.</param>
    /// <param name="nHead">Number of attention heads.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    public EncoderConfig(int nLayer = 12, int nEmbedding = 1024, int nHidden = 4096, int nHead = 16, int headDim = 128)
    {
        if (nLayer <= 0) throw new ArgumentOutOfRangeException(nameof(nLayer), "Must be greater than 0");
        if (nEmbedding <= 0) throw new ArgumentOutOfRangeException(nameof(nEmbedding), "Must be greater than 0");
        if (nHidden <= 0) throw new ArgumentOutOfRangeException(nameof(nHidden), "Must be greater than 0");
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead), "Must be greater than 0");
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim), "Must be greater than 0");

        NLayer = nLayer;
        NEmbedding = nEmbedding;
        NHidden = nHidden;
        NHead = nHead;
        HeadDim = headDim;
    }
}