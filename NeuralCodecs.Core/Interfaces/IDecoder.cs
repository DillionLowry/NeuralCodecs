namespace NeuralCodecs.Core.Interfaces
{
    internal interface IDecoder : IDisposable
    {
        int InputChannels { get; }
        int OutputChannels { get; }
        int[] Rates { get; }

        ValueTask<float[]> DecodeAsync(float[] latents, CancellationToken ct = default);

        ValueTask<float[][]> DecodeBatchAsync(float[][] latents, CancellationToken ct = default);
    }
}