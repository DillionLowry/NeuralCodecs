using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Models;
using System.Runtime.CompilerServices;
using System.Text;

namespace NeuralCodecs.Core.Interfaces
{
    /// <summary>
    /// Base interface for neural codec models
    /// </summary>
    public interface INeuralCodec : IDisposable
    {
        ModelConfig Config { get; }

        //void Save(string path);

        void LoadWeights(string path);
    }
}