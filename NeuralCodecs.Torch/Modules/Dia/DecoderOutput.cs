using NeuralCodecs.Torch.Config.Dia;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Container for decoder generation output.
/// Manages token storage and manipulation during generation.
/// </summary>
public class DecoderOutput : Module
{
    /// <summary>
    /// Generated token tensor (shape: [L, C])
    /// where L is max audio length and C is number of channels
    /// </summary>
    public Tensor GeneratedTokens { get; }

    /// <summary>
    /// The step at which the prefill completes.
    /// </summary>
    public int[] PrefillSteps { get; private set; }

    private DecoderOutput(Tensor generatedTokens) : base(nameof(DecoderOutput))
    {
        GeneratedTokens = generatedTokens;
        PrefillSteps = [];
    }

    /// <summary>
    /// Creates a new DecoderOutput with the specified configuration and device.
    /// </summary>
    /// <param name="batchSize"></param>
    /// <param name="config">Model configuration</param>
    /// <param name="device">Device for tensor operations</param>
    /// <returns>New DecoderOutput</returns>
    public static DecoderOutput New(int batchSize, DiaConfig config, Device device)
    {
        int maxAudioLen = config.Data.AudioLength;

        // Initialize token tensor with -1 (uninitialized)
        var generatedTokens = full(
            new long[] { batchSize, maxAudioLen, config.Data.Channels },
            -1,
            ScalarType.Int32,
            device: device);

        return new DecoderOutput(generatedTokens);
    }

    /// <summary>
    /// Gets tokens at the specified step range.
    /// </summary>
    /// <param name="stepFrom">Starting step index</param>
    /// <param name="stepTo">Optional ending step index (defaults to stepFrom + 1)</param>
    /// <returns>Tensor slice of tokens, shape [stepTo-stepFrom, C]</returns>
    public Tensor GetTokensAt(int stepFrom, int? stepTo = null)
    {
        stepTo ??= stepFrom + 1;
        var tokens = GeneratedTokens.index(TensorIndex.Colon, TensorIndex.Slice(stepFrom, stepTo), TensorIndex.Colon);

        return tokens;
    }

    /// <summary>
    /// Updates a single token in the output.
    /// </summary>
    /// <param name="decOut">New token tensor (shape: [C])</param>
    /// <param name="step">Step index to update</param>
    /// <param name="applyMask">Whether to apply masking for invalid tokens</param>
    public void UpdateOne(Tensor decOut, int step, bool applyMask = false)
    {
        var dec = decOut.to(GeneratedTokens.dtype);
        if (applyMask)
        {
            using var stepSlice = GeneratedTokens.index(TensorIndex.Colon, step, TensorIndex.Colon);
            using var mask = stepSlice.eq(-1);
            var updatedValues = where(mask, dec, stepSlice);

            GeneratedTokens[TensorIndex.Colon, step, TensorIndex.Colon] = updatedValues;
        }
        else
        {
            GeneratedTokens[TensorIndex.Colon, step, TensorIndex.Colon] = dec;
        }
    }

    /// <summary>
    /// Prefills the output with initial tokens.
    /// </summary>
    /// <param name="decOut">Token tensor to prefill with (shape: [L, C])</param>
    /// <param name="prefillSteps">The step (or steps if batching) at which the prefill completes</param>
    public void Prefill(Tensor decOut, int[] prefillSteps)
    {
        var length = decOut.shape[1];
        GeneratedTokens[TensorIndex.Colon, TensorIndex.Slice(stop:length), TensorIndex.Colon] = decOut.MoveToOuterDisposeScope();
        PrefillSteps = prefillSteps;
    }

    /// <summary>
    /// Moves the current instance's <see cref="GeneratedTokens"/> tensor to the outer dispose scope.
    /// </summary>
    /// <returns>The current <see cref="DecoderOutput"/> instance, allowing for method chaining.</returns>
    public DecoderOutput MoveToOuterDisposeScope()
    {
        GeneratedTokens.MoveToOuterDisposeScope();
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
            GeneratedTokens?.Dispose();
        }
        base.Dispose(disposing);
    }
}