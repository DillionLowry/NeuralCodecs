using NeuralCodecs.Torch.Config.Dia;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Transformer Encoder Layer using DenseGeneral.
/// </summary>
public class EncoderLayer : Module<Tensor, EncoderInferenceState, int, Tensor>
{

    private readonly RMSNorm pre_sa_norm;
    private readonly SelfAttention self_attention;
    private readonly RMSNorm post_sa_norm;
    private readonly MlpBlock mlp;
    private readonly int _embedDim;
    private readonly ScalarType _computeDtype;

    /// <summary>
    /// Creates a new EncoderLayer.
    /// </summary>
    /// <param name="config">Model configuration</param>
    /// <param name="computeDtype">Computation data type</param>
    public EncoderLayer(DiaConfig config, ScalarType computeDtype)
        : base(nameof(EncoderLayer))
    {
        var modelConfig = config.Model;
        var encConfig = config.Model.Encoder;
        _embedDim = encConfig.NEmbedding;
        _computeDtype = computeDtype;

        pre_sa_norm = new RMSNorm(
            normalizedShape: _embedDim,
            eps: modelConfig.NormalizationLayerEpsilon,
            elementwiseAffine: true,
            dtype: ScalarType.Float32
        );

        self_attention = new SelfAttention(
            config,
            qEmbedDim: _embedDim,
            kvEmbedDim: _embedDim,
            numQueryHeads: encConfig.NHead,
            numKvHeads: encConfig.NHead,
            headDim: encConfig.HeadDim,
            computeDtype: computeDtype,
            outEmbedDim: _embedDim
        );

        post_sa_norm = new RMSNorm(
            normalizedShape: _embedDim,
            eps: modelConfig.NormalizationLayerEpsilon,
            elementwiseAffine: true,
            dtype: ScalarType.Float32
            );

        mlp = new MlpBlock(
            embedDim: _embedDim,
            intermediateDim: encConfig.NHidden,
            computeDtype: computeDtype
        );

        RegisterComponents();
    }

    /// <summary>
    /// Processes the input tensor through the encoder layer, applying self-attention and a feed-forward network.
    /// </summary>
    /// <param name="x">The input tensor with shape [batch size, sequence length, embedding size].</param>
    /// <param name="state">The encoder inference state, which provides positional information and attention masks.</param>
    /// <param name="currIndex">The current index in the sequence being processed. Defaults to 0.</param>
    /// <returns>A tensor with the same shape as the input, representing the output of the encoder layer after applying
    /// self-attention and the feed-forward network.</returns>
    /// <exception cref="ArgumentException">Thrown if the shape of <paramref name="x"/> does not match the expected shape  [batch size, sequence length,
    /// embedding size].</exception>
    public override Tensor forward(Tensor x, EncoderInferenceState state, int currIndex = 0)
    {
        using var scope = NewDisposeScope();
        var expectedShape = new long[] { x.shape[0], x.shape[1], _embedDim };
        if (!x.shape.SequenceEqual(expectedShape))
        {
            throw new ArgumentException($"Expected shape {string.Join(",", expectedShape)}, got {string.Join(",", x.shape)}");
        }

        var residual = x;

        // Self-attention with pre-normalization
        var xNorm = pre_sa_norm.forward(x).to(_computeDtype);

        var saOut = self_attention.forward(
            x: xNorm,
            qPositions: state.Positions,
            attnMask: state.AttentionMask,
            currentIdx: currIndex
        );
        x = residual.add(saOut);

        // MLP with pre-normalization
        residual = x;
        xNorm = post_sa_norm.forward(x.to(_computeDtype, non_blocking: true));
        var mlpOut = mlp.forward(xNorm);
        x = residual.add(mlpOut);

        return x.MoveToOuterDisposeScope();
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            pre_sa_norm.Dispose();
            self_attention.Dispose();
            post_sa_norm.Dispose();
            mlp.Dispose();
        }
        base.Dispose(disposing);
    }
}