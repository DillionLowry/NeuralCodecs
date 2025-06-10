using NeuralCodecs.Torch.Config.Dia;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Transformer Decoder Stack.
/// </summary>
public class Decoder : Module<Tensor, DecoderInferenceState, Tensor>
{
    private readonly int _numChannels;
    private readonly int _numLayers;
    private readonly ModuleList<Embedding> embeddings;
    private readonly ModuleList<DecoderLayer> layers;
    private readonly RMSNorm norm;
    private readonly DenseGeneral logits_dense;
    private readonly ScalarType _computeDtype;

    /// <summary>
    /// Creates a new Decoder.
    /// </summary>
    /// <param name="config">Model configuration</param>
    /// <param name="computeDtype">Computation data type</param>
    public Decoder(DiaConfig config, ScalarType computeDtype)
        : base(nameof(Decoder))
    {
        var modelConfig = config.Model;
        var dataConfig = config.Data;
        _numChannels = dataConfig.Channels;
        _numLayers = modelConfig.Decoder.NLayer;
        _computeDtype = computeDtype;

        embeddings = new();
        for (int i = 0; i < _numChannels; i++)
        {
            embeddings.Add(Embedding(
                num_embeddings: modelConfig.TgtVocabSize,
                embedding_dims: modelConfig.Decoder.NEmbedding,
                dtype: computeDtype
            ));
        }

        layers = new();
        for (int i = 0; i < _numLayers; i++)
        {
            layers.Add(new DecoderLayer(config, computeDtype));
        }

        norm = new RMSNorm(
            normalizedShape: modelConfig.Decoder.NEmbedding,
            eps: modelConfig.NormalizationLayerEpsilon,
            elementwiseAffine: true,
            dtype: ScalarType.Float32
        );

        logits_dense = new DenseGeneral(
            inShapes: new[] { modelConfig.Decoder.NEmbedding },
            outFeatures: new[] { _numChannels, modelConfig.TgtVocabSize },
            axis: new[] { -1 },
            weightDtype: computeDtype
        );

        RegisterComponents();
    }

    /// <summary>
    /// Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
    /// </summary>
    /// <param name="encOut">Encoder output tensor [B, S, E] - B=batch, S=source length, E=embedding dim</param>
    /// <param name="encPositions">Encoder position indices [B, S]</param>
    /// <param name="kPaddingMask"> Optional padding mask for keys [B, S]</param>
    /// <returns>List of KVCache for each layer</returns>
    public List<KVCache> PrecomputeCrossAttnCache(Tensor encOut, Tensor encPositions, Tensor? kPaddingMask = null)
    {
        using var scope = NewDisposeScope();

        var perLayerKvCache = layers.Select(layer =>
        {
            var crossAttnModule = layer.CrossAttention;

            // Project to key/value spaces
            var kProj = crossAttnModule.k_proj.forward(encOut);
            var vProj = crossAttnModule.v_proj.forward(encOut);

            // Apply rotary embeddings to keys
            var kEmb = crossAttnModule._rotaryEmb.forward(kProj, encPositions);

            // Transpose for attention computation
            var k = kEmb.transpose(1, 2);
            var v = vProj.transpose(1, 2);

            if (kPaddingMask is not null)
            {
                k = k.masked_fill(~kPaddingMask.unsqueeze(1).unsqueeze(3), 0.0);
            }

            return KVCache.FromKV(k.MoveToOuterDisposeScope(), v.MoveToOuterDisposeScope());
        }).ToList();

        return perLayerKvCache;
    }

    /// <summary>
    /// Performs a single decoding step, managing KV caches layer by layer.
    /// </summary>
    /// <param name="tgtIds_Bx1xC">Target token IDs [B, 1, C] - B=batch, C=channels</param>
    /// <param name="state">Decoder inference state</param>
    /// <param name="currentIndex"></param>
    /// <returns>Logits for the current step [B, 1, C, V] - V=vocab size</returns>
    public Tensor DecodeStep(Tensor tgtIds_Bx1xC, DecoderInferenceState state, int currentIndex)
    {
        using var scope = NewDisposeScope();
        Tensor x = null;

        // Extract tokens, get embeddings, and sum them for each channel
        for (int i = 0; i < _numChannels; i++)
        {
            var channelTokens = tgtIds_Bx1xC[TensorIndex.Ellipsis, i];
            var channelEmbed = embeddings[i].forward(channelTokens);
            x = x is null ? channelEmbed : x.add(channelEmbed);
        }

        // Process through decoder layers
        for (int i = 0; i < layers.Count; i++)
        {
            var selfCache = state.SelfAttentionCache[i];
            var crossCache = state.CrossAttentionCache[i];

            x = layers[i].forward(
                x!,
                state,
                selfAttnCache: selfCache,
                crossAttnCache: crossCache,
                currentIndex: currentIndex
            );
        }

        // Final normalization and project to logits
        x = norm.forward(x!);
        var logits_Bx1xCxV = logits_dense.forward(x);

        return logits_Bx1xCxV.to(_computeDtype).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Forward pass for the Decoder stack, managing KV caches.
    /// </summary>
    /// <param name="tgtIds_BxTxC">Target token IDs [B, T, C] - B=batch, T=time, C=channels</param>
    /// <param name="state">Decoder inference state</param>
    /// <returns>Logits [B, T, C, V] - V=vocab size</returns>
    public override Tensor forward(Tensor tgtIds_BxTxC, DecoderInferenceState state)
    {
        using var scope = NewDisposeScope();
        var numChannelsIn = tgtIds_BxTxC.size(-1);
        if (numChannelsIn != _numChannels)
        {
            throw new ArgumentException($"Input channels mismatch. Expected {_numChannels}, got {numChannelsIn}");
        }

        Tensor x = null;

        // Extract tokens, get embeddings, and sum them for each channel
        for (int i = 0; i < _numChannels; i++)
        {
            var channelTokens = tgtIds_BxTxC[TensorIndex.Ellipsis, i];
            var channelEmbed = embeddings[i].forward(channelTokens);
            x = x is null ? channelEmbed : x.add(channelEmbed);
        }

        // Process through decoder layers
        for (int i = 0; i < _numLayers; i++)
        {
            var selfCache = state.SelfAttentionCache[i];
            var crossCache = state.CrossAttentionCache[i];

            x = layers[i].forward(
                x!,
                state,
                selfAttnCache: selfCache,
                crossAttnCache: crossCache,
                prefill: true
            );
        }

        // Final normalization and project to logits
        x = norm.forward(x!);
        var logits_BxTxCxV = logits_dense.forward(x);

        return logits_BxTxCxV.to(_computeDtype).MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            embeddings?.Dispose();
            layers?.Dispose();
            norm?.Dispose();
            logits_dense?.Dispose();
        }
        base.Dispose(disposing);
    }
}