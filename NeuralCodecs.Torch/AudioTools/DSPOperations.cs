using System.Numerics;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.AudioTools;

public class DSPOperations
{
    private int _originalBatchSize;
    private int _originalNumChannels;
    private int _paddedSignalLength;

    private (int windowLength, int hopLength) PreprocessSignalForWindowing(
        AudioSignal signal,
        float windowDuration,
        float hopDuration)
    {
        _originalBatchSize = signal.BatchSize;
        _originalNumChannels = signal.NumChannels;

        int windowLength = (int)(windowDuration * signal.SampleRate);
        int hopLength = (int)(hopDuration * signal.SampleRate);

        // Ensure window length is multiple of hop length
        if (windowLength % hopLength != 0)
        {
            int factor = windowLength / hopLength;
            windowLength = factor * hopLength;
        }

        signal.ZeroPad(hopLength, hopLength);
        _paddedSignalLength = signal.SignalLength;

        return (windowLength, hopLength);
    }

    public IEnumerable<AudioSignal> Windows(
        AudioSignal signal,
        float windowDuration,
        float hopDuration,
        bool preprocess = true)
    {
        int windowLength = 0, hopLength = 0;

        if (preprocess)
        {
            (windowLength, hopLength) = PreprocessSignalForWindowing(signal, windowDuration, hopDuration);
        }

        signal.AudioData = signal.AudioData.reshape(-1, 1, signal.SignalLength);

        for (int b = 0; b < signal.BatchSize; b++)
        {
            int i = 0;
            while (true)
            {
                int startIdx = i * hopLength;
                i++;
                int endIdx = startIdx + windowLength;
                if (endIdx > signal.SignalLength)
                    break;

                var windowSlice = signal[b].AudioData.narrow(-1, startIdx, windowLength);
                yield return new AudioSignal(windowSlice, signal.SampleRate);
            }
        }
    }

    public AudioSignal CollectWindows(
        AudioSignal signal,
        float windowDuration,
        float hopDuration,
        bool preprocess = true)
    {
        int windowLength = 0, hopLength = 0;

        if (preprocess)
        {
            (windowLength, hopLength) = PreprocessSignalForWindowing(signal, windowDuration, hopDuration);
        }

        // Unfold operation using TorchSharp
        var unfolded = torch.nn.functional.unfold(
            signal.AudioData.reshape(-1, 1, 1, signal.SignalLength),
            kernel_size: windowLength,
            stride: hopLength);

        // Reshape to (batch * channels * num_windows, 1, window_length)
        unfolded = unfolded.permute(0, 2, 1).reshape(-1, 1, windowLength);
        signal.AudioData = unfolded;

        return signal;
    }

    public AudioSignal OverlapAndAdd(
        AudioSignal signal,
        float hopDuration)
    {
        int hopLength = (int)(hopDuration * signal.SampleRate);
        int windowLength = signal.SignalLength;

        var (nb, nch) = (_originalBatchSize, _originalNumChannels);

        // Reshape for folding operation
        var unfolded = signal.AudioData.reshape(nb * nch, -1, windowLength).permute(0, 2, 1);

        // Fold operation
        var folded = torch.nn.functional.fold(
            unfolded,
            output_size: _paddedSignalLength,
            kernel_size: windowLength,
            stride: hopLength);

        // Create normalization tensor
        var norm = torch.ones_like(unfolded);
        var normFolded = torch.nn.functional.fold(
            norm,
            output_size: _paddedSignalLength,
            kernel_size: windowLength,
            stride: hopLength);

        // Normalize
        folded /= normFolded;

        // Reshape back
        folded = folded.reshape(nb, nch, -1);
        signal.AudioData = folded;

        // Remove padding
        signal.Trim(hopLength, hopLength);

        return signal;
    }

    public AudioSignal LowPass(
        AudioSignal signal,
        Tensor cutoffs,
        int zeros = 51)
    {
        cutoffs = cutoffs.to_type(torch.float32) / signal.SampleRate;
        var filtered = torch.zeros_like(signal.AudioData);

        // Apply filtering for each cutoff frequency
        for (int i = 0; i < cutoffs.size(0); i++)
        {
            var cutoff = cutoffs[i];

            // Create FIR filter
            var n = torch.arange(-zeros / 2, zeros / 2 + 1);
            var h = 2 * cutoff.item<float>() * torch.sinc(2 * cutoff * n);

            // Apply Hamming window
            var window = 0.54f - 0.46f * torch.cos(2 * (float)Math.PI * (n + zeros / 2) / zeros);
            h *= window;

            // Normalize
            h /= h.sum();

            // Reshape for conv1d
            h = h.reshape(1, 1, -1).to(signal.Device);

            // Apply filtering
            filtered[i] = torch.nn.functional.conv1d(
                signal.AudioData[i].unsqueeze(0),
                h,
                padding: zeros / 2
            ).squeeze(0);
        }

        signal.AudioData = filtered;
        signal.StftData = null;

        return signal;
    }

    public AudioSignal HighPass(
        AudioSignal signal,
        Tensor cutoffs,
        int zeros = 51)
    {
        cutoffs = cutoffs.to_type(torch.float32) / signal.SampleRate;
        var filtered = torch.zeros_like(signal.AudioData);

        for (int i = 0; i < cutoffs.size(0); i++)
        {
            var cutoff = cutoffs[i];

            // Create FIR filter (high-pass by spectral inversion of low-pass)
            var n = torch.arange(-zeros / 2, zeros / 2 + 1);
            var h = 2 * cutoff.item<float>() * torch.sinc(2 * cutoff * n);
            h = -h;  // Invert for high-pass
            h[zeros / 2] += 1;  // Add impulse

            // Apply Hamming window
            var window = 0.54f - 0.46f * torch.cos(2 * (float)Math.PI * (n + zeros / 2) / zeros);
            h *= window;

            // Normalize
            h /= h.abs().sum();

            // Reshape for conv1d
            h = h.reshape(1, 1, -1).to(signal.Device);

            // Apply filtering
            filtered[i] = torch.nn.functional.conv1d(
                signal.AudioData[i].unsqueeze(0),
                h,
                padding: zeros / 2
            ).squeeze(0);
        }

        signal.AudioData = filtered;
        signal.StftData = null;

        return signal;
    }

    public AudioSignal MaskFrequencies(
        AudioSignal signal,
        Tensor fminHz,
        Tensor fmaxHz,
        float val = 0.0f)
    {
        var (mag, phase) = (signal.Magnitude, signal.Phase);

        // Ensure tensor dimensions match
        fminHz = fminHz.expand(mag.shape);
        fmaxHz = fmaxHz.expand(mag.shape);

        if (torch.any(fminHz >= fmaxHz).item<bool>())
        {
            throw new ArgumentException("fminHz must be less than fmaxHz");
        }

        // Build frequency mask
        int nbins = (int)mag.size(-2);
        var binsHz = torch.linspace(0, signal.SampleRate / 2, nbins, device: signal.Device);
        var binsExpanded = binsHz.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            .expand(signal.BatchSize, 1, -1, mag.size(-1));

        var mask = (fminHz <= binsExpanded) & (binsExpanded < fmaxHz);
        mask = mask.to(signal.Device);

        // Apply mask
        mag = mag.masked_fill(mask, val);
        phase = phase.masked_fill(mask, val);

        signal.StftData = mag * torch.exp(Complex.ImaginaryOne * phase);

        return signal;
    }

    public AudioSignal MaskTimesteps(
        AudioSignal signal,
        Tensor tminS,
        Tensor tmaxS,
        float val = 0.0f)
    {
        var (mag, phase) = (signal.Magnitude, signal.Phase);

        // Ensure tensor dimensions match
        tminS = tminS.expand(mag.shape);
        tmaxS = tmaxS.expand(mag.shape);

        if (torch.any(tminS >= tmaxS).item<bool>()) // todo : check this
        {
            throw new ArgumentException("tminS must be less than tmaxS");
        }

        // Build time mask
        int nt = (int)mag.size(-1);
        var binsT = torch.linspace(0, signal.SignalDuration, nt, device: signal.Device);
        var binsExpanded = binsT.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            .expand(signal.BatchSize, 1, mag.size(-2), -1);

        var mask = (tminS <= binsExpanded) & (binsExpanded < tmaxS);

        // Apply mask
        mag = mag.masked_fill(mask, val);
        phase = phase.masked_fill(mask, val);

        signal.StftData = mag * torch.exp(Complex.ImaginaryOne * phase);

        return signal;
    }

    public AudioSignal Preemphasis(AudioSignal signal, float coef = 0.85f)
    {
        // Create pre-emphasis filter [1, -coef]
        var kernel = torch.tensor(new float[] { 1, -coef, 0 })
            .reshape(1, 1, -1)
            .to(signal.Device);

        var x = signal.AudioData.reshape(-1, 1, signal.SignalLength);
        x = torch.nn.functional.conv1d(x, kernel, padding: 1);

        signal.AudioData = x.reshape(signal.BatchSize, signal.NumChannels, -1);

        return signal;
    }
}