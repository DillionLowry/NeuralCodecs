using static TorchSharp.torch;

namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Extension methods for AudioSignal DSP operations.
/// Based on Descript's audiotools python library.
/// </summary>
public static class AudioSignalExtensions
{
    private static readonly DSP dsp = new DSP();

    /// <summary>
    /// Collects overlapping windows into a single tensor.
    /// </summary>
    public static AudioSignal CollectWindows(
        this AudioSignal signal,
        float windowDuration,
        float hopDuration,
        bool preprocess = true)
    {
        return dsp.CollectWindows(signal, windowDuration, hopDuration, preprocess);
    }

    /// <summary>
    /// Applies a high-pass filter to the audio signal.
    /// </summary>
    public static AudioSignal HighPass(
        this AudioSignal signal,
        Tensor cutoffs,
        int zeros = 51)
    {
        return dsp.HighPass(signal, cutoffs, zeros);
    }

    /// <summary>
    /// Applies a low-pass filter to the audio signal.
    /// </summary>
    public static AudioSignal LowPass(
        this AudioSignal signal,
        Tensor cutoffs,
        int zeros = 51)
    {
        return dsp.LowPass(signal, cutoffs, zeros);
    }

    /// <summary>
    /// Masks frequencies in the audio signal.
    /// </summary>
    public static AudioSignal MaskFrequencies(
        this AudioSignal signal,
        Tensor fminHz,
        Tensor fmaxHz,
        float val = 0.0f)
    {
        return dsp.MaskFrequencies(signal, fminHz, fmaxHz, val);
    }

    /// <summary>
    /// Masks timesteps in the audio signal.
    /// </summary>
    public static AudioSignal MaskTimesteps(
        this AudioSignal signal,
        Tensor tminS,
        Tensor tmaxS,
        float val = 0.0f)
    {
        return dsp.MaskTimesteps(signal, tminS, tmaxS, val);
    }

    /// <summary>
    /// Overlaps and adds the audio signal.
    /// </summary>
    public static AudioSignal OverlapAndAdd(
        this AudioSignal signal,
        float hopDuration)
    {
        return dsp.OverlapAndAdd(signal, hopDuration);
    }

    /// <summary>
    /// Applies preemphasis to the audio signal.
    /// </summary>
    public static AudioSignal Preemphasis(
        this AudioSignal signal,
        float coef = 0.85f)
    {
        return dsp.Preemphasis(signal, coef);
    }

    /// <summary>
    /// Generates overlapping windows from the audio signal.
    /// </summary>
    public static IEnumerable<AudioSignal> Windows(
        this AudioSignal signal,
        float windowDuration,
        float hopDuration,
        bool preprocess = true)
    {
        return dsp.Windows(signal, windowDuration, hopDuration, preprocess);
    }
}