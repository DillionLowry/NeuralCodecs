using NeuralCodecs.Torch.Config.Dia;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Transformer Encoder Stack.
/// Processes text input through multiple transformer encoder layers.
/// Each layer consists of self-attention and feed-forward components.
/// </summary>
public class Encoder : Module<Tensor, EncoderInferenceState, Tensor>
{
    private readonly Embedding embedding;
    private readonly ModuleList<EncoderLayer> layers;
    private readonly RMSNorm norm;
    private readonly ScalarType _computeDtype;

    /// <summary>
    /// Initializes a new instance of the <see cref="Encoder"/> class, which represents the encoder component of a
    /// transformer-based model.
    /// </summary>
    /// <param name="config">The configuration object containing model and encoder-specific settings,
    /// such as vocabulary size, embedding dimensions, and number of layers.</param>
    /// <param name="computeDtype">The data type used for computations, such as <see cref="ScalarType.Float32"/>
    /// or <see cref="ScalarType.Float16"/>.</param>
    public Encoder(DiaConfig config, ScalarType computeDtype)
        : base(nameof(Encoder))
    {
        var modelConfig = config.Model;
        var encConfig = config.Model.Encoder;
        _computeDtype = computeDtype;

        embedding = Embedding(
            num_embeddings: modelConfig.SrcVocabSize,
            embedding_dims: encConfig.NEmbedding,
            device: Utils.TorchUtils.GetDevice( config.Device),
            dtype: computeDtype

        );

        layers = new();
        for (int i = 0; i < encConfig.NLayer; i++)
        {
            layers.Add(new EncoderLayer(config, computeDtype));
        }

        norm = new RMSNorm(
            encConfig.NEmbedding,
            eps: modelConfig.NormalizationLayerEpsilon,
            elementwiseAffine: true,
            dtype: ScalarType.Float32
        );

        RegisterComponents();
    }

    /// <summary>
    /// Processes the input tensor through the embedding layer, a sequence of transformer layers,
    /// and a normalization layer, producing the final output tensor.
    /// </summary>
    /// <param name="xIds">A tensor containing input token IDs  used to look up embeddings in the
    /// embedding layer.</param>
    /// <param name="state">The inference state used to maintain context across transformer layers.</param>
    /// <returns>A tensor representing the processed output after applying the embedding, transformer layers,
    /// and normalization. The output tensor is in the specified compute data type.</returns>
    public override Tensor forward(Tensor xIds, EncoderInferenceState state)
    {
        using var scope = NewDisposeScope();

        var x = embedding.forward(xIds);

        for (int i = 0; i < layers.Count; i++)
        {
            x = layers[i].forward(x, state);
        }
        x = norm.forward(x).to(_computeDtype);

        return x.MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            embedding?.Dispose();
            layers?.Dispose();
            norm?.Dispose();
        }

        base.Dispose(disposing);
    }
}