using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.AudioTools;

// Helper class for managing window functions
/// <summary>
/// Provides window functions for signal processing operations.
/// </summary>
public static class WindowFunctions
{
    private static readonly Dictionary<(string, int, string), Tensor> Cache = new();

    /// <summary>
    /// Gets a window function of the specified type and length.
    /// </summary>
    /// <param name="windowType">Type of window (hann, hamming, blackman, bartlett, or ones).</param>
    /// <param name="length">Length of the window in samples.</param>
    /// <param name="device">Device where the window tensor should be created.</param>
    /// <returns>A tensor containing the window function.</returns>
    public static Tensor GetWindow(string windowType, int length, string device)
    {
        var key = (windowType, length, device);
        if (!Cache.ContainsKey(key))
        {
            Tensor window = windowType switch
            {
                "hann" => torch.hann_window(length),
                "hamming" => torch.hamming_window(length),
                "blackman" => torch.blackman_window(length),
                "bartlett" => torch.bartlett_window(length),
                "ones" => torch.ones(length),
                _ => throw new ArgumentException($"Unsupported window type: {windowType}")
            };
            Cache[key] = window.to(device);
        }
        return Cache[key];
    }

    /// <summary>
    /// Clears the window function cache.
    /// </summary>
    public static void ClearCache()
    {
        foreach (var window in Cache.Values)
        {
            window.Dispose();
        }
        Cache.Clear();
    }
}