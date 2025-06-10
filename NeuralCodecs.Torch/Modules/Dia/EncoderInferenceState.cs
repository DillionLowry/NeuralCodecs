using NeuralCodecs.Torch.Config.Dia;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Parameters specifically for encoder inference.
/// Manages state required for encoding text input in both conditional and unconditional paths.
/// </summary>
public class EncoderInferenceState : Module
{
    /// <summary>
    /// Maximum sequence length.
    /// </summary>
    public int MaxSequenceLength { get; }

    /// <summary>
    /// Device for tensor operations.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// Position indices (shape: [B, S])
    /// B=2 (unconditional + conditional paths)
    /// S=source text sequence length
    /// Contains position IDs from 0 to S-1
    /// </summary>
    public Tensor Positions { get; }

    /// <summary>
    /// Padding mask (shape: [B, S])
    /// Boolean tensor where True indicates valid tokens, False for padding
    /// Used to mask out padding tokens during attention
    /// </summary>
    public Tensor PaddingMask { get; }

    /// <summary>
    /// Attention mask (shape: [B, 1, S, S])
    /// Combines padding information into attention-compatible format
    /// Value -inf for masked positions, 0 for valid positions
    /// </summary>
    public Tensor AttentionMask { get; }

    internal EncoderInferenceState(
        int maxSequenceLength,
        Device device,
        Tensor positions,
        Tensor paddingMask,
        Tensor attentionMask) : base(nameof(EncoderInferenceState))
    {
        MaxSequenceLength = maxSequenceLength;
        Device = device;
        Positions = positions;
        PaddingMask = paddingMask;
        AttentionMask = attentionMask;
    }

    /// <summary>
    /// Creates a new instance of <see cref="EncoderInferenceState"/> configured for the specified input data and device.
    /// </summary>
    /// <remarks>The method initializes the encoder inference state by calculating positional encodings,  padding
    /// masks, and attention masks based on the provided configuration and input tensor.  The resulting state is ready for
    /// use in encoder-based inference tasks.</remarks>
    /// <param name="config">The configuration object containing parameters such as text length and padding value.</param>
    /// <param name="condSrc">A tensor representing the conditional source input. This tensor is used to determine the device,
    /// padding mask, and other state initialization parameters.</param>
    /// <returns>An initialized <see cref="EncoderInferenceState"/> object configured with the specified input data,  including
    /// positional encodings, padding masks, and attention masks.</returns>
    public static EncoderInferenceState New(DiaConfig config, Tensor condSrc)
    {
        var device = condSrc.device;
        var textLength = config.Data.TextLength;

        var positions = arange(textLength, dtype: float32, device: device).unsqueeze(0);
        var paddingMask = (condSrc.squeeze(1) != config.Data.TextPadValue).to(device).repeat_interleave(2, dim: 0);

        var attentionMask = AttentionMaskUtils.CreateAttentionMask(
            paddingMask, paddingMask, device, isCausal: false);

        return new EncoderInferenceState(
            maxSequenceLength: textLength,
            device: device,
            positions: positions,
            paddingMask: paddingMask,
            attentionMask: attentionMask);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            Positions?.Dispose();
            PaddingMask?.Dispose();
            AttentionMask?.Dispose();
        }

        base.Dispose(disposing);
    }
}