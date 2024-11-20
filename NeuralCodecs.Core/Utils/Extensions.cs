namespace NeuralCodecs.Core.Utils;

/// <summary>
/// Extension methods for streams and collections
/// </summary>
internal static class Extensions
{
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

    public static bool IsNullOrEmpty<T>(this ICollection<T>? collection)
    {
        return collection == null || collection.Count == 0;
    }

    public static async Task CopyToAsync(this Stream source, Stream destination, IProgress<long> progress, CancellationToken cancellationToken)
    {
        var buffer = new byte[81920];
        int bytesRead;
        while ((bytesRead = await source.ReadAsync(buffer, cancellationToken)) != 0)
        {
            await destination.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
            progress?.Report(bytesRead);
        }
    }
}