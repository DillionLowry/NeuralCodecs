namespace NeuralCodecs.Core.Interfaces
{
    internal interface IEncoder : IDisposable
    {
        int InputChannels { get; }
        int OutputChannels { get; }
        int[] Strides { get; }

        ValueTask<float[]> EncodeAsync(float[] input, CancellationToken ct = default);
        //ValueTask<float[][]> EncodeBatchAsync(float[][] inputs, CancellationToken ct = default);
    }
}