using NAudio.Wave;

namespace NeuralCodecs.Core.Interfaces
{
    public interface IAudioFileReader : IDisposable
    {
        WaveFormat WaveFormat { get; }
        long Length { get; }
        int Read(byte[] buffer, int offset, int count);
    }
}
