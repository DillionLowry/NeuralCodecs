using System.Threading;

namespace NeuralCodecs.Core.Utils;

/// <summary>
/// Provides extension methods for Stream and Collection types.
/// </summary>
internal static class Extensions
{
    /// <summary>
    /// Asynchronously copies data from the source stream to the destination stream with progress reporting.
    /// </summary>
    /// <param name="source">The source stream to copy from.</param>
    /// <param name="destination">The destination stream to copy to.</param>
    /// <param name="bufferSize">The size of the buffer to use for copying.</param>
    /// <param name="progress">Optional progress reporter for bytes read.</param>
    /// <param name="ct">Cancellation token to cancel the operation.</param>
    /// <returns>A task representing the asynchronous copy operation.</returns>
    public static async Task CopyToAsync(
        this Stream source,
        Stream destination,
        int bufferSize,
        IProgress<long>? progress = null,
        CancellationToken ct = default)
    {
        var buffer = new byte[bufferSize];
        int bytesRead;
        long totalRead = 0;

        while ((bytesRead = await source.ReadAsync(buffer, ct)) != 0)
        {
            await destination.WriteAsync(buffer.AsMemory(0, bytesRead), ct);
            totalRead += bytesRead;
            progress?.Report(bytesRead);
        }
    }

    /// <summary>
    /// Asynchronously copies data from the source stream to the destination stream with progress reporting.
    /// </summary>
    /// <param name="source">The source stream to copy from.</param>
    /// <param name="destination">The destination stream to copy to.</param>
    /// <param name="progress">Progress reporter for bytes read.</param>
    /// <param name="ct">Cancellation token to cancel the operation.</param>
    /// <returns>A task representing the asynchronous copy operation.</returns>
    public static async Task CopyToAsync(
        this Stream source,
        Stream destination,
        IProgress<long>? progress = null,
        CancellationToken ct = default)
    {
        var buffer = new byte[81920];
        int bytesRead;
        while ((bytesRead = await source.ReadAsync(buffer, ct)) != 0)
        {
            await destination.WriteAsync(buffer.AsMemory(0, bytesRead), ct);
            progress?.Report(bytesRead);
        }
    }

    /// <summary>
    /// Determines whether the specified collection is null or contains no elements.
    /// </summary>
    /// <typeparam name="T">The type of elements in the collection.</typeparam>
    /// <param name="collection">The collection to check.</param>
    /// <returns>True if the collection is null or empty; otherwise, false.</returns>
    public static bool IsNullOrEmpty<T>(this ICollection<T>? collection)
    {
        return collection == null || collection.Count == 0;
    }
}