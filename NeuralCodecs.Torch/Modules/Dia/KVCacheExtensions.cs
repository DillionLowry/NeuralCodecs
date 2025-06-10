namespace NeuralCodecs.Torch.Modules.Dia;

public static class KVCacheExtensions
{
    /// <summary>
    /// Disposes all caches in a collection
    /// </summary>
    /// <param name="caches">The collection of caches to dispose</param>
    public static void Dispose(this List<KVCache> caches)
    {
        foreach (var cache in caches)
        {
            cache?.Dispose();
        }
    }

    /// <summary>
    /// Clones all caches in a collection
    /// </summary>
    /// <param name="caches">The collection of caches to clone</param>
    public static List<KVCache> Clone(this List<KVCache> caches)
    {
        var cloned = new List<KVCache>();
        foreach (var cache in caches)
        {
            var k = cache.K.clone();
            var v = cache.V.clone();
            cloned.Add(KVCache.FromKV(k, v));
        }
        return cloned;
    }

    /// <summary>
    /// Resets all caches in a collection, calling zero_ on each cache's K and V tensors.
    /// </summary>
    /// <param name="caches">The collection of caches to reset</param>
    public static void Reset(this List<KVCache> caches)
    {
        foreach (var cache in caches)
        {
            cache?.K?.zero_();
            cache?.V?.zero_();
        }
    }
}