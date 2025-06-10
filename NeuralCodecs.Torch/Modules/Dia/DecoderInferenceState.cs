using NeuralCodecs.Torch.Config.Dia;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Parameters specifically for decoder inference.
/// </summary>
public class DecoderInferenceState : Module
{
    /// <summary>
    /// Device for tensor operations.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// Data type for computation.
    /// </summary>
    public ScalarType Dtype { get; }

    /// <summary>
    /// Encoder output (shape: [B, S, E])
    /// B=2 (unconditional + conditional paths)
    /// S=source sequence length
    /// E=embedding dimension
    /// </summary>
    public Tensor EncoderOutput { get; }

    /// <summary>
    /// Encoder positions (shape: [B, S])
    /// Position indices for source sequence
    /// </summary>
    public Tensor EncoderPositions { get; }

    /// <summary>
    /// Decoder positions (shape: [B, T])
    /// Position indices for target sequence
    /// T varies dynamically during decoding
    /// </summary>
    public Tensor DecoderPositions { get; private set; }

    /// <summary>
    /// Self-attention cache for each decoder layer.
    /// </summary>
    public List<KVCache> SelfAttentionCache { get; }

    /// <summary>
    /// Cross-attention cache for each decoder layer.
    /// </summary>
    public List<KVCache> CrossAttentionCache { get; }

    /// <summary>
    /// Causal attention mask used to enforce autoregressive behavior in sequence processing.
    /// </summary>
    public Tensor CausalAttentionMask { get; }

    /// <summary>
    /// Cross-attention mask for decoder-encoder attention.
    /// </summary>
    public Tensor CrossAttentionMask { get; }

    private DecoderInferenceState(
        Device device,
        ScalarType dtype,
        Tensor encoderOutput,
        Tensor encoderPositions,
        Tensor decoderPositions,
        List<KVCache> selfAttentionCache,
        List<KVCache> crossAttentionCache,
        Tensor causalAttentionMask,
        Tensor crossAttentionMask)
        : base(nameof(DecoderInferenceState))
    {
        Device = device;
        Dtype = dtype;
        EncoderOutput = encoderOutput;
        EncoderPositions = encoderPositions;
        DecoderPositions = decoderPositions;
        SelfAttentionCache = selfAttentionCache;
        CrossAttentionCache = crossAttentionCache;
        CausalAttentionMask = causalAttentionMask;
        CrossAttentionMask = crossAttentionMask;
    }

    /// <summary>
    /// Creates a new DecoderInferenceState from the given configuration and encoder state.
    /// </summary>
    /// <param name="config">Model configuration</param>
    /// <param name="encoderState">Encoder inference state</param>
    /// <param name="encoderOutput">Encoder output tensor</param>
    /// <param name="decoderCrossAttentionCache">Cross-attention cache for each decoder layer</param>
    /// <param name="computeType">Computation data type</param>
    /// <param name="maxGenerationLength">Maximum generation length (optional)</param>
    /// <returns>New DecoderInferenceState</returns>
    public static DecoderInferenceState New(
        DiaConfig config,
        EncoderInferenceState encoderState,
        Tensor encoderOutput,
        List<KVCache> decoderCrossAttentionCache,
        ScalarType computeType,
        int? maxGenerationLength = null)
    {
        var device = encoderOutput.device;
        var maxAudioLen = maxGenerationLength ?? config.Data.AudioLength;
        var batchSize = (int)encoderOutput.shape[0] / 2;

        // Create decoder positions [B, 1] - initial position 0
        var decoderPositions = full([2 * batchSize, 1], 0, ScalarType.Int32, device: device);
        var causalMask = tril(ones([maxAudioLen, maxAudioLen], ScalarType.Bool, device: device));

        // Create cross-attention mask
        var decMask = ones([2 * batchSize, 1], dtype: ScalarType.Bool, device: device);
        var crossAttnMask = AttentionMaskUtils.CreateAttentionMask(decMask, encoderState.PaddingMask, device, isCausal: false);

        // Create self-attention cache for each decoder layer
        var selfAttentionCache = new List<KVCache>();
        for (int i = 0; i < config.Model.Decoder.NLayer; i++)
        {
            selfAttentionCache.Add(new KVCache(
                batchSize,
                config.Model.Decoder.KvHeads,
                maxAudioLen,
                config.Model.Decoder.GqaHeadDim,
                computeType,
                device));
        }

        return new DecoderInferenceState(
            device: device,
            dtype: computeType,
            encoderOutput: encoderOutput,
            encoderPositions: encoderState.Positions,
            decoderPositions: decoderPositions,
            selfAttentionCache: selfAttentionCache,
            crossAttentionCache: decoderCrossAttentionCache,
            causalAttentionMask: causalMask,
            crossAttentionMask: crossAttnMask);
    }

    /// <summary>
    /// Prepares the decoder state for a specific decoding step.
    /// </summary>
    /// <param name="stepFrom">Starting step index</param>
    /// <param name="stepTo">Optional ending step index (defaults to stepFrom + 1)</param>
    public void PrepareStep(int stepFrom, int? stepTo = null)
    {
        stepTo ??= stepFrom + 1;

        // Dispose previous positions tensor to prevent memory leaks
        DecoderPositions?.Dispose();

        DecoderPositions = arange(stepFrom, stepTo, dtype: int32, device: Device)
            .unsqueeze_(0);
    }

    /// <summary>
    /// Moves the current instance and its associated resources to the outer disposal scope.
    /// </summary>
    /// <remarks>This method ensures that the resources managed by the current instance, including
    /// <see cref="EncoderOutput"/>, <see cref="EncoderPositions"/>, and <see cref="DecoderPositions"/>, are moved
    /// to the outer disposal scope. The Cache tensors are not moved, as they should be detached from the current scope.
    /// </remarks>
    /// <returns>The current <see cref="DecoderInferenceState"/> instance, allowing for method chaining.</returns>
    public DecoderInferenceState MoveToOuterDisposeScope()
    {
        EncoderOutput.MoveToOuterDisposeScope();
        EncoderPositions.MoveToOuterDisposeScope();
        DecoderPositions.MoveToOuterDisposeScope();
        CrossAttentionMask.MoveToOuterDisposeScope();
        // cache should be detached from the current scope
        return this;
    }

    /// <summary>
    /// Releases resources used by this instance
    /// </summary>
    /// <param name="disposing">True when called from Dispose, false when called from finalizer</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // Dispose tensors
            EncoderOutput?.Dispose();
            EncoderPositions?.Dispose();
            DecoderPositions?.Dispose();
            CrossAttentionMask?.Dispose();
            // Don't dispose cross-attention cache as it might be shared
        }

        base.Dispose(disposing);
    }
}