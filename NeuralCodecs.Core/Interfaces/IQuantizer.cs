namespace NeuralCodecs.Core.Interfaces
{
    internal interface IQuantizer : IDisposable
    {
        int InputDim { get; }
        int CodebookSize { get; }
        int CodebookDim { get; }

        ValueTask<(float[] quantized, int[] indices)> QuantizeAsync(
            float[] inputs,
            CancellationToken ct = default);

        ValueTask<float[]> DequantizeAsync(
            int[] indices,
            CancellationToken ct = default);
    }
}