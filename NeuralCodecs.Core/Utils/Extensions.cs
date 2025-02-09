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
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        if (bufferSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(bufferSize));

        var buffer = new byte[bufferSize];
        long totalBytesRead = 0;

        while (true)
        {
            int bytesRead = await source.ReadAsync(buffer, ct).ConfigureAwait(false);
            if (bytesRead == 0) break; // End of stream

            await destination.WriteAsync(buffer.AsMemory(0, bytesRead), ct).ConfigureAwait(false);

            totalBytesRead += bytesRead;
            progress?.Report(totalBytesRead);

            if (ct.IsCancellationRequested) break;
        }

        // Ensure writes are flushed
        await destination.FlushAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Asynchronously copies data from the source stream to the destination stream with progress reporting.
    /// </summary>
    /// <param name="source">The source stream to copy from.</param>
    /// <param name="destination">The destination stream to copy to.</param>
    /// <param name="progress">Progress reporter for bytes read.</param>
    /// <param name="ct">Cancellation token to cancel the operation.</param>
    /// <returns>A task representing the asynchronous copy operation.</returns>
    public static Task CopyToAsync(
        this Stream source,
        Stream destination,
        IProgress<long>? progress = null,
        CancellationToken ct = default)
    {
        return CopyToAsync(source, destination, 81920, progress, ct);
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