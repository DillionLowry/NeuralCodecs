using NeuralCodecs.Core.Models;
using System.Text.Json.Serialization;

namespace NeuralCodecs.Core.Interfaces
{
    public interface IModelConfig
    {
        Device Device { get; set; }
        int SamplingRate { get; set; }
        string Architecture { get; set; }
        string Version { get; set; }
        IDictionary<string, string> Metadata { get; set; }
    }
}