using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Creates attention masks for self and cross attention.
/// </summary>
public static class AttentionMaskUtils
{
    /// <summary>
    /// Creates the attention mask (self or cross) mimicking JAX segment ID logic.
    /// </summary>
    /// <param name="qPaddingMask1d">Query padding mask (shape: [B, Tq])</param>
    /// <param name="kPaddingMask1d">Key padding mask (shape: [B, Tk])</param>
    /// <param name="device">Device to create the mask on</param>
    /// <param name="isCausal">Whether to apply causal masking</param>
    /// <returns>Attention mask (shape: [B, 1, Tq, Tk])</returns>
    public static Tensor CreateAttentionMask(
        Tensor qPaddingMask1d,
        Tensor kPaddingMask1d,
        Device device,
        bool isCausal = false)
    {
        using var scope = NewDisposeScope();

        // Reshape masks for broadcasting
        var pMaskQ = qPaddingMask1d.unsqueeze(2);
        var pMaskK = kPaddingMask1d.unsqueeze(1);

        var nonPadAttendsNonPad = pMaskQ.logical_and(pMaskK);
        var padAttendsPad = pMaskQ.logical_not().logical_and(pMaskK.logical_not());

        var mask = nonPadAttendsNonPad.logical_or(padAttendsPad);

        if (isCausal)
        {
            var causalMask3d = tril(ones_like(mask[0], dtype: ScalarType.Bool, device: device));
            var causalMask = mask.logical_and(causalMask3d);
            return causalMask.unsqueeze(1).MoveToOuterDisposeScope();
        }
        else
        {
            return mask.unsqueeze(1).MoveToOuterDisposeScope();
        }
    }
}