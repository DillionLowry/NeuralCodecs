using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Utility methods for audio processing in the Dia model.
/// </summary>
public static class AudioUtils
{
    /// <summary>
    /// Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
    /// Negative t_idx => BOS; t_idx >= T => PAD.
    /// </summary>
    /// <param name="B">Batch size</param>
    /// <param name="T">Sequence length</param>
    /// <param name="C">Number of channels</param>
    /// <param name="delayPattern">Delay pattern array</param>
    /// <returns>Tuple containing t_idx_BxTxC and indices_BTCx3</returns>
    public static (Tensor t_idx_BxTxC, Tensor indices_BTCx3) BuildDelayIndices(int B, int T, int C, int[] delayPattern)
    {
        using var scope = NewDisposeScope();
        var delay_arr = tensor(delayPattern, dtype: int32);

        var t_idx_BxT = broadcast_to(
            arange(T, dtype: int32)[TensorIndex.None, TensorIndex.Colon],
            B, T
        );
        var t_idx_BxTx1 = t_idx_BxT[TensorIndex.Ellipsis, TensorIndex.None];
        var t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C);

        var b_idx_BxTxC = broadcast_to(
             arange(B, dtype: int32).view(B, 1, 1),
             B, T, C
         );
        var c_idx_BxTxC = broadcast_to(
            arange(C, dtype: int32).view(1, 1, C),
            B, T, C
        );

        // Clamp time indices to [0..T-1]
        var t_clamped_BxTxC = clamp(t_idx_BxTxC, 0, T - 1);

        var indices_BTCx3 = stack(
            [
                b_idx_BxTxC.reshape(-1),
                t_clamped_BxTxC.reshape(-1),
                c_idx_BxTxC.reshape(-1),
            ],
            dim: 1
        ).to(int64);  // Ensure indices are long type for indexing

        return (t_idx_BxTxC.MoveToOuterDisposeScope(), indices_BTCx3.MoveToOuterDisposeScope());
    }

    /// <summary>
    /// Applies the delay pattern to batched audio tokens using precomputed indices,
    /// inserting BOS where t_idx LT 0 and PAD where t_idx GTE T.
    /// </summary>
    /// <param name="audio_BxTxC">Audio tokens with shape B, T, C</param>
    /// <param name="padValue">The padding token</param>
    /// <param name="bosValue">The BOS token</param>
    /// <param name="precomp">Precomputed indices from BuildDelayIndices</param>
    /// <returns>Delayed audio tokens with shape B, T, C</returns>
    public static Tensor ApplyAudioDelay(Tensor audio_BxTxC, int padValue, int bosValue,
            (Tensor t_idx_BxTxC, Tensor indices_BTCx3) precomp)
    {
        using var scope = NewDisposeScope();
        var device = audio_BxTxC.device;
        var (t_idx_BxTxC, indices_BTCx3) = precomp;

        var t_idx_device = t_idx_BxTxC.to(device);
        var indices_device = indices_BTCx3.to(device);

        // Gather using flat indices
        var gathered_flat = audio_BxTxC[indices_device[TensorIndex.Colon, 0],
                                            indices_device[TensorIndex.Colon, 1],
                                            indices_device[TensorIndex.Colon, 2]];

        // Reshape to original tensor shape
        var gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape);

        // Create masks
        var mask_bos = t_idx_device < 0;
        var mask_pad = t_idx_device >= audio_BxTxC.shape[1];

        // Create scalar tensors
        var bos_tensor = tensor(bosValue, dtype: audio_BxTxC.dtype, device: device);
        var pad_tensor = tensor(padValue, dtype: audio_BxTxC.dtype, device: device);

        var result_intermediate = where(mask_bos, bos_tensor, gathered_BxTxC);
        var result_BxTxC = where(mask_pad, pad_tensor, result_intermediate);

        return result_BxTxC.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Precompute indices for the revert operation.
    /// </summary>
    /// <param name="B">Batch size</param>
    /// <param name="T">Sequence length</param>
    /// <param name="C">Number of channels</param>
    /// <param name="delayPattern">Delay pattern array</param>
    /// <returns>
    /// Tuple containing:
    /// - t_idx_BxTxC: Time indices plus delay, shape B, T, C
    /// - indices_BTCx3: Flattened indices for gathering, shape [B*T*C, 3]
    /// </returns>
    public static (Tensor t_idx_BxTxC, Tensor indices_BTCx3) BuildRevertIndices(int B, int T, int C, int[] delayPattern)
    {
        // delayArr: Shape [C] - holds delay pattern values
        var delayArr = tensor(delayPattern, int32);

        // Create time indices [B, T, 1] - fixing dtype to match Python
        // t_idx_BT1: Shape [B, T, 1] - holds indices 0...T-1 for each batch
        var t_idx_BT1 = broadcast_to(arange(T).unsqueeze(0), B, T);
        t_idx_BT1 = t_idx_BT1.unsqueeze(-1);

        // Apply delay pattern for each channel B, T, C
        // For revert, we add the delay rather than subtract it
        // t_idx_BxTxC: Shape B, T, C - holds t + delay[c] for each position
        var t_idx_BxTxC = minimum(
            t_idx_BT1 + delayArr.view(1, 1, C),  // Add delay pattern
            tensor(T - 1)                       // Clamp to max time index
        );

        var b_idx_BxTxC = broadcast_to(arange(B).view(B, 1, 1), B, T, C);
        var c_idx_BxTxC = broadcast_to(arange(C).view(1, 1, C), B, T, C);

        // Stack indices for gather operation [B*T*C, 3]
        // indices_BTCx3: Shape [B*T*C, 3] - each row is [batch_idx, time_idx, channel_idx]
        var indices_BTCx3 = stack(new[] {
            b_idx_BxTxC.reshape(-1),
            t_idx_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1)
        }, dim: 1).to(int64);  // Cast to Int64 for indexing

        return (t_idx_BxTxC, indices_BTCx3);
    }

    /// <summary>
    /// Reverts a delay pattern from batched audio tokens using precomputed indices.
    /// </summary>
    /// <param name="audio_BxTxC">Input delayed audio tensor, shape B, T, C</param>
    /// <param name="padValue">Padding value for out-of-bounds indices</param>
    /// <param name="precomp">Precomputed revert indices tuple</param>
    /// <param name="T">Original sequence length before padding</param>
    /// <returns>Reverted audio tensor with same shape as input B, T, C</returns>
    public static Tensor RevertAudioDelay(Tensor audio_BxTxC, int padValue,
        (Tensor t_idx_BxTxC, Tensor indices_BTCx3) precomp, int T)
    {
        using var scope = NewDisposeScope();
        var (t_idx_BxTxC, indices_BTCx3) = precomp;
        var device = audio_BxTxC.device;

        // Move precomputed indices to the same device as audio_BxTxC
        var t_idx_device = t_idx_BxTxC.to(device);
        var indices_device = indices_BTCx3.to(device);

        // Gather values
        var gathered_flat = audio_BxTxC.index(
           indices_device.index(TensorIndex.Colon, 0),
           indices_device.index(TensorIndex.Colon, 1),
           indices_device.index(TensorIndex.Colon, 2));

        // Reshape back to original dimensions
        var gathered_BxTxC = gathered_flat.view(audio_BxTxC.size());

        // Create pad_tensor on the correct device
        var pad_tensor = tensor(padValue, dtype: audio_BxTxC.dtype, device: device);
        var T_val = tensor(T, device: device);

        // Apply mask
        var result_BxTxC = where(t_idx_device.ge(T_val), pad_tensor, gathered_BxTxC);

        return result_BxTxC.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Decodes audio codes into a tensor representation of audio data using the specified model.
    /// </summary>
    /// <remarks>This method uses the provided <see cref="DAC"/> model to first convert the audio
    /// codes into a quantized representation and then decode the quantized data into audio. Ensure that the input
    /// tensor contains exactly one frame to avoid an <see cref="ArgumentException"/>.</remarks>
    /// <param name="model">The <see cref="DAC"/> model used to decode the audio codes.</param>
    /// <param name="audioCodes">A tensor containing the audio codes to decode. Must contain exactly one frame (i.e., <c>audioCodes.size(0)
    /// == 1</c>).</param>
    /// <returns>A <see cref="Tensor"/> representing the decoded audio data.</returns>
    /// <exception cref="ArgumentException">Thrown if <paramref name="audioCodes"/> does not contain exactly one frame.</exception>
    public static Tensor Decode(Torch.Models.DAC model, Tensor audioCodes)
    {
        if (audioCodes.size(0) != 1)
        {
            throw new ArgumentException($"Expected one frame, got {audioCodes.size(0)}");
        }

        try
        {
            var quantized = model.FromCodes(audioCodes);
            var decodedAudio = model.Decode(quantized);

            return decodedAudio;
        }
        catch (Exception e)
        {
            Console.WriteLine($"Error in decode method: {e.Message}");
            throw;
        }
    }
}