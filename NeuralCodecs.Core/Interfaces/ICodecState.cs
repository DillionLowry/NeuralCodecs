using NeuralCodecs.Core.Models;

namespace NeuralCodecs.Core.Interfaces
{
    internal interface ICodecState
    {
        CodecParameters Parameters { get; }
        CompressionInfo? LastCompressionInfo { get; }
        TimeSpan TotalProcessingTime { get; }
        long TotalSamplesProcessed { get; }
        bool IsInitialized { get; }
    }
}