namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Contains basic information about an audio file or stream.
/// </summary>
public class AudioInfo
{
    public float Duration { get; set; }
    public int NumFrames { get; set; }
    public float SampleRate { get; set; }
}