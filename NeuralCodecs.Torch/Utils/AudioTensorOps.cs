using NeuralCodecs.Torch.AudioTools;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Utils;

public static class AudioTensorOps
{
    /// <summary>
    /// Computes STFT for raw audio tensor
    /// </summary>
    public static Tensor ComputeSTFT(
        Tensor audio,
        int windowLength,
        int hopLength,
        string windowType = "hann",
        bool center = true)
    {
        // Normalize shape to (batch, channels, time)
        if (audio.dim() == 1)
        {
            audio = audio.unsqueeze(0).unsqueeze(0);
        }
        else if (audio.dim() == 2)
        {
            audio = audio.unsqueeze(audio.size(0) <= 4 ? 0 : 1);
        }

        var window = AudioSignal.GetWindow(windowType, windowLength, audio.device.ToString()); // TODO: Device string

        // Reshape for stft
        var (batchSize, channels, samples) = (audio.size(0), audio.size(1), audio.size(2));
        var flattened = audio.reshape(-1, samples);

        var stft = torch.stft(
            flattened,
            n_fft: windowLength,
            hop_length: hopLength,
            window: window,
            center: center,
            return_complex: true
        );

        // Reshape back to (batch, channels, freq, time)
        var (freqBins, timeSteps) = (stft.size(1), stft.size(2));
        return stft.reshape(batchSize, channels, freqBins, timeSteps);
    }

    /// <summary>
    /// Creates a mel filterbank matrix
    /// </summary>
    private static Tensor CreateMelFilterbank(
        int sampleRate,
        int nMels,
        int nFft,
        float fMin,
        float? fMax = null)
    {
        fMax ??= sampleRate / 2.0f;

        // Create frequencies array
        var fftFreqs = linspace(0f, sampleRate / 2f, nFft / 2 + 1);

        // Convert Hz to mel
        var mels = from_array(new[] { fMin, fMax.Value }).HertzToMel();
        var melPoints = linspace(mels[0].item<float>(), mels[1].item<float>(), nMels + 2);
        var hzPoints = melPoints.MelToHertz();

        // Create filterbank matrix
        var filterbank = zeros(nMels, nFft / 2 + 1);

        // Populate filters
        for (int i = 0; i < nMels; i++)
        {
            var filter = new float[nFft / 2 + 1];
            for (int j = 0; j < filter.Length; j++)
            {
                var freq = fftFreqs[j].item<float>();
                var leftHz = hzPoints[i].item<float>();
                var centerHz = hzPoints[i + 1].item<float>();
                var rightHz = hzPoints[i + 2].item<float>();

                // Lower slope
                if (freq >= leftHz && freq < centerHz)
                {
                    filter[j] = (freq - leftHz) / (centerHz - leftHz);
                }
                // Upper slope
                else if (freq >= centerHz && freq < rightHz)
                {
                    filter[j] = (rightHz - freq) / (rightHz - centerHz);
                }
            }
            filterbank[i] = from_array(filter);
        }

        // Normalize filters
        var enorm = 2.0f / (hzPoints[2..] - hzPoints[..^2]);
        filterbank *= enorm.unsqueeze(1);

        return filterbank;
    }

    // TODO: further testing
    /// <summary>
    /// Computes mel spectrogram for raw audio tensor
    /// </summary>
    public static Tensor ComputeMelSpectrogram(
            Tensor audio,
            int sampleRate,
            int nMels,
            float melFmin = 0.0f,
            float? melFmax = null,
            int windowLength = 2048,
            int hopLength = 512,
            string windowType = "hann",
            bool center = true,
            Device device = null)
    {
        using var scope = NewDisposeScope();

        // Compute STFT
        var stft = ComputeSTFT(
            audio,
            windowLength,
            hopLength,
            windowType,
            center
        );

        // Get magnitudes
        var magnitudes = abs(stft);

        // Create mel filterbank
        var (nFft, nFreqs) = GetFftLengths(windowLength);
        var melBasis = CreateMelFilterbank(
            sampleRate,
            nMels,
            nFft,
            melFmin,
            melFmax
        ).to(audio.device);

        // Apply mel filterbank
        // (batch, channels, freq, time) -> (batch, channels, time, freq)
        magnitudes = magnitudes.transpose(2, 3);

        // Matrix multiply with mel filterbank
        var melSpec = magnitudes.matmul(melBasis.t());

        // Return to original shape
        melSpec = melSpec.transpose(2, 3);

        return melSpec.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Computes the Fourier transform lengths needed for mel filterbank computation
    /// </summary>
    private static (int nFft, int nFreqs) GetFftLengths(int windowLength)
    {
        var nFft = Math.Max(windowLength, 2048);
        // Get number of unique FFT frequencies
        var nFreqs = nFft / 2 + 1;
        return (nFft, nFreqs);
    }
}