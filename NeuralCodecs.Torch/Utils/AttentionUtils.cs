using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Utils;

/// <summary>
/// General-purpose attention utilities for various attention mechanisms.
/// </summary>
public static class AttentionUtils
{    
    /// <summary>
    /// Computes scaled dot-product attention matching PyTorch's API.
    /// </summary>
    /// <param name="query">Query tensor [batch_size, num_heads, query_length, head_dim].</param>
    /// <param name="key">Key tensor [batch_size, num_heads, key_length, head_dim] or [batch_size, num_kv_heads, key_length, head_dim] for GQA.</param>
    /// <param name="value">Value tensor [batch_size, num_heads, key_length, head_dim] or [batch_size, num_kv_heads, key_length, head_dim] for GQA.</param>
    /// <param name="attentionMask">Optional attention mask. False values are masked out.</param>
    /// <param name="dropoutRate">Dropout probability for attention weights (0.0 = no dropout).</param>
    /// <param name="isCausal">Whether to apply causal masking (lower triangular mask).</param>
    /// <param name="scale">Scaling factor. If null, uses 1/sqrt(head_dim).</param>
    /// <param name="enableGqa">Whether to enable Grouped Query Attention (repeats key/value to match query heads).</param>
    /// <param name="gqaGroups">Number of repetitions for GQA. If null, computed automatically from head dimensions.</param>
    /// <returns>Attention output tensor [batch_size, num_heads, query_length, head_dim].</returns>
    public static Tensor ScaledDotProductAttention(
        Tensor query,
        Tensor key,
        Tensor value,
        Tensor? attentionMask = null,
        float dropoutRate = 0.0f,
        bool isCausal = false,
        float? scale = null,
        bool enableGqa = false,
        int? gqaGroups = null)
    {
        long headDim = query.shape[3];
        float actualScale = scale ?? (1.0f / MathF.Sqrt(headDim));

        if (enableGqa)
        {
            key = key.repeat_interleave(gqaGroups ?? (query.shape[1] / key.shape[1]), dim: 1);
            value = value.repeat_interleave(gqaGroups ?? (query.shape[1] / value.shape[1]), dim: 1);
        }

        // Compute attention scores
        var scores = matmul(query, key.transpose(-1, -2)) * actualScale;

        // Apply causal mask if needed
        if (isCausal)
        {
            long queryLen = query.shape[2];
            long keyLen = key.shape[2];
            var causalMask = tril(ones(queryLen, keyLen, dtype: ScalarType.Bool, device: query.device));
            scores.masked_fill_(~causalMask, float.NegativeInfinity);
        }

        // Apply attention mask if provided
        if (attentionMask is not null)
        {
            scores.masked_fill_(~attentionMask, float.NegativeInfinity);
        }


        // Apply softmax to get attention weights
        var attentionWeights = functional.softmax(scores, dim: -1);

        // Apply dropout if specified
        if (dropoutRate > 0.0f)
        {
            attentionWeights = functional.dropout(attentionWeights, p: dropoutRate, training: true);
        }

        // Compute output
        return matmul(attentionWeights, value);
    }

    /// <summary>
    /// Creates a simple causal attention mask.
    /// </summary>
    /// <param name="sequenceLength">Length of the sequence.</param>
    /// <param name="device">Device to create the mask on.</param>
    /// <param name="dtype">Data type for the mask.</param>
    /// <returns>Causal mask tensor [sequence_length, sequence_length].</returns>
    public static Tensor CreateCausalMask(long sequenceLength, Device device, ScalarType dtype = ScalarType.Bool)
    {
        return tril(ones(sequenceLength, sequenceLength, dtype: dtype, device: device));
    }

    /// <summary>
    /// Creates a combined attention mask from padding and optional causal masking.
    /// </summary>
    /// <param name="queryPaddingMask">Query padding mask [batch_size, query_length].</param>
    /// <param name="keyPaddingMask">Key padding mask [batch_size, key_length].</param>
    /// <param name="isCausal">Whether to apply causal masking.</param>
    /// <param name="device">Device to create the mask on.</param>
    /// <returns>Combined attention mask [batch_size, 1, query_length, key_length].</returns>
    public static Tensor CreateCombinedMask(
        Tensor queryPaddingMask,
        Tensor keyPaddingMask,
        bool isCausal,
        Device device)
    {
        using var scope = NewDisposeScope();

        // Reshape masks for broadcasting
        var queryMask = queryPaddingMask.unsqueeze(2);  // [B, T_q, 1]
        var keyMask = keyPaddingMask.unsqueeze(1);      // [B, 1, T_k]

        // Create base mask where both query and key positions are valid
        var mask = queryMask.logical_and(keyMask);      // [B, T_q, T_k]

        if (isCausal)
        {
            long queryLen = queryMask.shape[1];
            long keyLen = keyMask.shape[2];
            var causalMask = CreateCausalMask(Math.Max(queryLen, keyLen), device)
                .narrow(0, 0, queryLen)
                .narrow(1, 0, keyLen);
            mask = mask.logical_and(causalMask);
        }

        return mask.unsqueeze(1).MoveToOuterDisposeScope(); // [B, 1, T_q, T_k]
    }

    /// <summary>
    /// Applies rotary positional embeddings to query and key tensors.
    /// </summary>
    /// <param name="tensor">Input tensor [batch_size, num_heads, seq_length, head_dim].</param>
    /// <param name="cos">Cosine values for rotary embeddings.</param>
    /// <param name="sin">Sine values for rotary embeddings.</param>
    /// <returns>Tensor with rotary embeddings applied.</returns>
    public static Tensor ApplyRotaryEmbedding(Tensor tensor, Tensor cos, Tensor sin)
    {
        // Split the last dimension into two halves
        long headDim = tensor.shape[3];
        long halfDim = headDim / 2;
        var x1 = tensor.narrow(-1, 0, halfDim);
        var x2 = tensor.narrow(-1, halfDim, halfDim);

        // Apply rotary transformation: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
        var rotatedX1 = (x1 * cos) - (x2 * sin);
        var rotatedX2 = (x1 * sin) + (x2 * cos);

        return cat(new[] { rotatedX1, rotatedX2 }, dim: -1);
    }

    /// <summary>
    /// Computes multi-head self-attention with flexible options.
    /// </summary>
    /// <param name="input">Input tensor [batch_size, seq_length, embed_dim].</param>
    /// <param name="queryProj">Query projection layer.</param>
    /// <param name="keyProj">Key projection layer.</param>
    /// <param name="valueProj">Value projection layer.</param>
    /// <param name="outputProj">Output projection layer.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="attentionMask">Optional attention mask.</param>
    /// <param name="isCausal">Whether to apply causal masking.</param>
    /// <param name="dropoutRate">Dropout rate for attention weights.</param>
    /// <returns>Output tensor [batch_size, seq_length, embed_dim].</returns>
    public static Tensor MultiHeadSelfAttention(
        Tensor input,
        Module<Tensor, Tensor> queryProj,
        Module<Tensor, Tensor> keyProj,
        Module<Tensor, Tensor> valueProj,
        Module<Tensor, Tensor> outputProj,
        int numHeads,
        Tensor? attentionMask = null,
        bool isCausal = false,
        float dropoutRate = 0.0f)
    {
        long batchSize = input.shape[0];
        long seqLength = input.shape[1];
        long embedDim = input.shape[2];
        long headDim = embedDim / numHeads;

        // Project to Q, K, V
        var query = queryProj.forward(input).view(batchSize, seqLength, numHeads, headDim).transpose(1, 2);
        var key = keyProj.forward(input).view(batchSize, seqLength, numHeads, headDim).transpose(1, 2);
        var value = valueProj.forward(input).view(batchSize, seqLength, numHeads, headDim).transpose(1, 2);

        // Apply attention
        var attnOutput = ScaledDotProductAttention(query, key, value, attentionMask, dropoutRate, isCausal);

        // Reshape and project output
        attnOutput = attnOutput.transpose(1, 2).contiguous().view(batchSize, seqLength, embedDim);
        return outputProj.forward(attnOutput);
    }
}