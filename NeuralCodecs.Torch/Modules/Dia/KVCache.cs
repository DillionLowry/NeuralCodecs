using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// Key-Value cache for attention layers.
/// </summary>
public class KVCache : Module
{
    /// <summary>
    /// Key cache (shape: [B, H, L, D])
    /// </summary>
    public Tensor K { get; private set; }

    /// <summary>
    /// Value cache (shape: [B, H, L, D])
    /// </summary>
    public Tensor V { get; private set; }

    /// <summary>
    /// Creates a new KVCache with the specified dimensions.
    /// </summary>
    /// <param name="batchSize"></param>
    /// <param name="numHeads">Number of attention heads</param>
    /// <param name="maxLen">Maximum sequence length</param>
    /// <param name="headDim">Dimension of each head</param>
    /// <param name="dtype">Data type for the cache</param>
    /// <param name="device">Device for the cache</param>
    /// <param name="k">Optional existing key cache</param>
    /// <param name="v">Optional existing value cache</param>
    public KVCache(
        int batchSize,
        int numHeads,
        int maxLen,
        int headDim,
        ScalarType dtype,
        Device device,
        Tensor? k = null,
        Tensor? v = null) : base("KVCache")
    {
        K = k ?? zeros([2 * batchSize, numHeads, maxLen, headDim], dtype: dtype, device: device);
        V = v ?? zeros([2 * batchSize, numHeads, maxLen, headDim], dtype: dtype, device: device);
        register_buffer("k", K);
        register_buffer("v", V);
    }

    /// <summary>
    /// Creates a KVCache from existing key and value tensors.
    /// </summary>
    /// <param name="k">Key tensor</param>
    /// <param name="v">Value tensor</param>
    /// <returns>New KVCache</returns>
    public static KVCache FromKV(Tensor k, Tensor v)
    {
        return new KVCache(
            batchSize: (int)(k.shape[0] / 2),
            numHeads: (int)k.shape[1],
            maxLen: (int)k.shape[2],
            headDim: (int)k.shape[3],
            dtype: k.dtype,
            device: k.device,
            k: k,
            v: v);
    }

    /// <summary>
    /// Updates the cache with new key and value tensors.
    /// </summary>
    /// <param name="k">New key tensor (shape: [B, H, 1, D])</param>
    /// <param name="v">New value tensor (shape: [B, H, 1, D])</param>
    /// <param name="currentIdx">Current cache index</param>
    /// <returns>Tuple of updated key and value caches</returns>
    public (Tensor kCache, Tensor vCache) Update(Tensor k, Tensor v, int? currentIdx)
    {
        var index = currentIdx is null ? TensorIndex.None : TensorIndex.Slice(currentIdx.Value, currentIdx.Value + 1);
        K[TensorIndex.Colon, TensorIndex.Colon, index, TensorIndex.Colon] = k;
        V[TensorIndex.Colon, TensorIndex.Colon, index, TensorIndex.Colon] = v;
        return (K, V);
    }

    /// <summary>
    /// Prefills the cache with initial key and value tensors.
    /// </summary>
    /// <param name="k">Key tensor to prefill with (shape: [B, H, T, D])</param>
    /// <param name="v">Value tensor to prefill with (shape: [B, H, T, D])</param>
    /// <returns>Tuple of the prefilled key and value tensors</returns>
    public void Prefill(Tensor k, Tensor v)
    {
        var prefillLen = k.size(2);
        K[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(stop: prefillLen), TensorIndex.Colon] = k;
        V[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(stop: prefillLen), TensorIndex.Colon] = v;
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            K?.Dispose();
            V?.Dispose();
        }
        base.Dispose(disposing);
    }
}