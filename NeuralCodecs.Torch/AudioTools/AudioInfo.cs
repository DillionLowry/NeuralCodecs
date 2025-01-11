namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Contains basic information about an audio file or stream.
/// </summary>
public class AudioInfo
{
    public float SampleRate { get; set; }

    public int NumFrames { get; set; }

    public float Duration { get; set; } //= NumFrames / SampleRate;
}