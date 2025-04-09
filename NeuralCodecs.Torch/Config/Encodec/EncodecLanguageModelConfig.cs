using NeuralCodecs.Core.Configuration;

namespace NeuralCodecs.Torch.Config.Encodec;

/// <summary>
/// Configuration for the LanguageModel.
/// </summary>
public class EncodecLanguageModelConfig : IModelConfig
{
    public DeviceConfiguration Device { get; set; } = DeviceConfiguration.CPU;
    public int SampleRate { get; set; } = 24000;
    public string Architecture { get; set; } = "language_model";
    public string Version { get; set; } = "1.0";
    public IDictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

    // Default constructor parameters from LanguageModel
    public int NumCodebooks { get; set; } = 32;
    public int CodebookSize { get; set; } = 1024;
    public int Dimension { get; set; } = 200;
    public int NumHeads { get; set; } = 8;
    public int NumLayers { get; set; } = 5;
    public float HiddenScale { get; set; } = 4.0f;
    public float MaxPeriod { get; set; } = 10000.0f;
    public int PastContext { get; set; } = 1000;
    public bool Gelu { get; set; } = true;
    public bool NormIn { get; set; } = true;
    public float Dropout { get; set; } = 0.0f;
}
