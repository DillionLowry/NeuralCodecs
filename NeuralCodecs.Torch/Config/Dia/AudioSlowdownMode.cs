namespace NeuralCodecs.Torch.Config.Dia;

/// <summary>
/// Audio slowdown mode options for Dia TTS.
/// </summary>
public enum AudioSlowdownMode
{
    /// <summary>
    /// Use a fixed static slowdown factor.
    /// </summary>
    Static,

    /// <summary>
    /// Use dynamic slowdown based on text length.
    /// </summary>
    Dynamic
}