using NeuralCodecs.Core.Models;

namespace NeuralCodecs.Core.Interfaces
{
    public interface IAudioProcessor : IDisposable
    {
        string Name { get; }

        ValueTask<float[]> ProcessAsync(float[] input, AudioFormat format, CancellationToken ct = default);
    }
}