namespace NeuralCodecs.Core.Interfaces
{
    internal interface IResidualVectorQuantizer : IDisposable
    {
        int NumCodebooks { get; }
        int[] Strides { get; }

        ValueTask<(float[] quantized, int[][] codes)> QuantizeAsync(
            float[] inputs,
            CancellationToken ct = default);

        ValueTask<float[]> DequantizeAsync(
            int[][] codes,
            CancellationToken ct = default);
    }
}