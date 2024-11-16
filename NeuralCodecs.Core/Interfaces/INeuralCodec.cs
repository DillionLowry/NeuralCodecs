using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Models;
using System.Runtime.CompilerServices;

namespace NeuralCodecs.Core.Interfaces
{
    /// <summary>
    /// Base interface for neural codec models
    /// </summary>
    public interface INeuralCodec
    {
        ModelConfig Config { get; }
        string Architecture { get; }
        Device Device { get; }

        void Save(string path);

        void Load(string path);

        void LoadWeights(string path);
    }
}