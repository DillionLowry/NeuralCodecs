using NeuralCodecs.Core.Models;

namespace NeuralCodecs.Core.Interfaces
{
    internal interface IMultiScaleCodec : IDisposable
    {
        EncodingConfig Config { get; }

        ValueTask<(float[] audio, int[][] codes)> EncodeAsync(
            float[] input,
            CancellationToken ct = default);

        ValueTask<float[]> DecodeAsync(
            int[][] codes,
            CancellationToken ct = default);

        ValueTask<CompressionInfo> GetCompressionInfoAsync(
            int[][] codes,
            int originalLength,
            CancellationToken ct = default);
    }
}