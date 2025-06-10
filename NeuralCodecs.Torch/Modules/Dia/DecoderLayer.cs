using NeuralCodecs.Torch.Config.Dia;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Transformer Decoder Layer using DenseGeneral.
/// </summary>
public class DecoderLayer : Module<Tensor, DecoderInferenceState, KVCache?, KVCache?, bool, int, Tensor>
{
    private readonly RMSNorm pre_sa_norm;
    private readonly SelfAttention self_attention;
    private readonly RMSNorm pre_ca_norm;
    private readonly CrossAttention cross_attention;
    private readonly RMSNorm pre_mlp_norm;
    private readonly MlpBlock mlp;
    private readonly ScalarType _computeDtype;

    /// <summary>
    /// Gets the cross-attention module of this decoder layer.
    /// </summary>
    public CrossAttention CrossAttention => cross_attention;

    /// <summary>
    /// Creates a new DecoderLayer.
    /// </summary>
    /// <param name="config">Model configuration</param>
    /// <param name="computeDtype">Computation data type</param>
    public DecoderLayer(DiaConfig config, ScalarType computeDtype)
        : base(nameof(DecoderLayer))
    {
        var modelConfig = config.Model;
        var decConfig = config.Model.Decoder;
        var encConfig = config.Model.Encoder;
        var decEmbedDim = decConfig.NEmbedding;
        var encEmbedDim = encConfig.NEmbedding;
        _computeDtype = computeDtype;

        // Normalization layers
        pre_sa_norm = new RMSNorm(
            normalizedShape: decEmbedDim,
            eps: modelConfig.NormalizationLayerEpsilon,
            elementwiseAffine: true,
            dtype: ScalarType.Float32
        );

        pre_ca_norm = new RMSNorm(
            normalizedShape: decEmbedDim,
            eps: modelConfig.NormalizationLayerEpsilon,
            elementwiseAffine: true,
            dtype: ScalarType.Float32
        );

        pre_mlp_norm = new RMSNorm(
            normalizedShape: decEmbedDim,
            eps: modelConfig.NormalizationLayerEpsilon,
            elementwiseAffine: true,
            dtype: ScalarType.Float32
        );

        // Self-attention (GQA) with Causal Masking
        self_attention = new SelfAttention(
            config,
            qEmbedDim: decEmbedDim,
            kvEmbedDim: decEmbedDim,
            numQueryHeads: decConfig.GqaQueryHeads,
            numKvHeads: decConfig.KvHeads,
            headDim: decConfig.GqaHeadDim,
            computeDtype: computeDtype,
            outEmbedDim: decEmbedDim
        );

        // Cross-attention (Multi-Head Attention)
        cross_attention = new CrossAttention(
            config,
            qEmbedDim: decEmbedDim,
            kvEmbedDim: encEmbedDim,
            numQueryHeads: decConfig.CrossQueryHeads,
            numKvHeads: decConfig.CrossQueryHeads,
            headDim: decConfig.CrossHeadDim,
            computeDtype: computeDtype,
            outEmbedDim: decEmbedDim
        );

        mlp = new MlpBlock(
            embedDim: decEmbedDim,
            intermediateDim: decConfig.NHidden,
            computeDtype: computeDtype
        );

        RegisterComponents();
    }

    /// <summary>
    /// Processes the input tensor through the decoder, updating the inference state as needed.
    /// </summary>
    /// <param name="x">The input tensor to be processed by the decoder.</param>
    /// <param name="state">The current inference state, which will be updated during processing.</param>
    /// <returns>A tensor representing the output of the decoder after processing the input.</returns>
    public Tensor forward(Tensor x, DecoderInferenceState state) =>
        forward(x, state, null, null, false);

    /// <summary>
    /// Processes the input tensor through a decoder layer, applying self-attention, cross-attention,  and a
    /// feed-forward network with pre-normalization at each stage.
    /// </summary>
    /// <param name="x">The input tensor to the decoder layer.</param>
    /// <param name="state">The current decoder inference state, including positional information and attention masks.</param>
    /// <param name="selfAttnCache">An optional cache for self-attention key-value pairs. </param>
    /// <param name="crossAttnCache">An optional cache for cross-attention key-value pairs. </param>
    /// <param name="prefill">A boolean value indicating whether the method is in prefill mode. If <see langword="true"/>,  causal masking
    /// is applied during self-attention to ensure autoregressive behavior.</param>
    /// <param name="currentIndex">The current decoding step index. This is used to select the appropriate positions and attention masks for
    /// the current step in the decoding process.</param>
    /// <returns>A tensor representing the output of the decoder layer after applying self-attention, cross-attention, and
    /// the feed-forward network.</returns>
    public override Tensor forward(
        Tensor x,
        DecoderInferenceState state,
        KVCache? selfAttnCache,
        KVCache? crossAttnCache,
        bool prefill = false,
        int currentIndex = 0)
    {
        using var scope = NewDisposeScope();
        var residual = x;

        var attnMask = state.CausalAttentionMask[TensorIndex.None, TensorIndex.None, currentIndex];
        var xNorm = pre_sa_norm.forward(x).to(_computeDtype);

        var saOut = self_attention.forward(
            x: xNorm,
            qPositions: state.DecoderPositions,
            attnMask: attnMask,
            cache: selfAttnCache,
            prefill: prefill,
            isCausal: prefill, // Only use causal masking during prefill
            currentIdx: currentIndex
        );

        x = residual.add(saOut);

        residual = x;
        xNorm = pre_ca_norm.forward(x).to(_computeDtype);

        // Updated cross-attention call to match new signature
        var caOut = cross_attention.forward(
            Xq: xNorm,
            qPositions: state.DecoderPositions,
            kvPositions: state.EncoderPositions,
            attnMask: state.CrossAttentionMask,
            cache: crossAttnCache
        );

        x = residual.add(caOut);

        residual = x;
        xNorm = pre_mlp_norm.forward(x).to(_computeDtype);
        var mlpOut = mlp.forward(xNorm);
        x = residual.add(mlpOut);

        return x.MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            pre_sa_norm?.Dispose();
            self_attention?.Dispose();
            pre_ca_norm?.Dispose();
            cross_attention?.Dispose();
            pre_mlp_norm?.Dispose();
            mlp?.Dispose();
        }
        base.Dispose(disposing);
    }
}