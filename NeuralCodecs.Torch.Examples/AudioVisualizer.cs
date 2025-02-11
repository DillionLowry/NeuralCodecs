using NAudio.Wave;
using SkiaSharp;
using Spectre.Console;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Examples;

public class AudioVisualizer
{
    /// <summary>
    /// Compares the original audio file with the encoded audio file by generating spectrograms and saving the comparison image.
    /// </summary>
    /// <param name="originalPath">Path to the original audio file.</param>
    /// <param name="encodedPath">Path to the encoded audio file.</param>
    /// <param name="sampleRate">Sample rate for loading the audio files.</param>
    /// <param name="comparisonSavePath">Path to save the comparison spectrogram image. Default is "spectrogram_comparison.png".</param>
    public static void CompareAudioSpectrograms(string originalPath, string encodedPath, int sampleRate, string comparisonSavePath = "spectrogram_comparison.png")
    {
        var (originalSpec, encodedSpec) = CreateSpectrograms(originalPath, encodedPath, sampleRate);

        SaveComparisonSpectrogram(originalSpec, encodedSpec, comparisonSavePath);
    }

    /// <summary>
    /// Compares the original audio file with the encoded audio file by generating spectrograms and saving individual and difference images.
    /// </summary>
    /// <param name="originalPath">Path to the original audio file.</param>
    /// <param name="encodedPath">Path to the encoded audio file.</param>
    /// <param name="sampleRate">Sample rate for loading the audio files.</param>
    /// <param name="originalSavePath">Path to save the original spectrogram image.</param>
    /// <param name="encodedSavePath">Path to save the encoded spectrogram image.</param>
    /// <param name="diffSavePath">Path to save the difference spectrogram image.</param>
    public static void CompareAudioSpectrograms(string originalPath, string encodedPath, int sampleRate, string originalSavePath, string encodedSavePath, string diffSavePath)
    {
        var (originalSpec, encodedSpec) = CreateSpectrograms(originalPath, encodedPath, sampleRate);

        // Save visual comparisons
        SaveSpectrogram(originalSpec, originalSavePath);
        SaveSpectrogram(encodedSpec, encodedSavePath);
        SaveDifference(originalSpec, encodedSpec, diffSavePath);
    }

    private static (Tensor originalSpec, Tensor encodedSpec) CreateSpectrograms(string originalPath, string encodedPath, int sampleRate)
    {
        // Load audio files
        var original = LoadAudioTensor(originalPath, sampleRate);
        var encoded = LoadAudioTensor(encodedPath, sampleRate);

        // Generate spectrograms
        var originalSpec = GenerateMelSpectrogram(original, sampleRate);
        var encodedSpec = GenerateMelSpectrogram(encoded, sampleRate);

        // Match sizes
        var minTimeFrames = Math.Min(originalSpec.size(0), encodedSpec.size(0));
        var minMels = Math.Min(originalSpec.size(1), encodedSpec.size(1));

        originalSpec = originalSpec.narrow(0, 0, minTimeFrames).narrow(1, 0, minMels);
        encodedSpec = encodedSpec.narrow(0, 0, minTimeFrames).narrow(1, 0, minMels);

        return (originalSpec, encodedSpec);
    }

    private static Tensor LoadAudioTensor(string path, int sampleRate)
    {
        var audio = LoadAudio(path, sampleRate);

        return tensor(audio, dtype: ScalarType.Float32)
            .reshape(1, 1, -1); // [batch, channels, samples]
    }

    public static float[] LoadAudio(string path, int targetSampleRate)
    {
        using var audioFile = new AudioFileReader(path);
        var buffer = new List<float>();
        var readBuffer = new float[audioFile.WaveFormat.SampleRate * 4];
        int samplesRead;
        while ((samplesRead = audioFile.Read(readBuffer, 0, readBuffer.Length)) > 0)
        {
            buffer.AddRange(readBuffer.Take(samplesRead));
        }
        // Convert to mono, resample if necessary
        if (audioFile.WaveFormat.Channels > 1)
        {
            buffer = ConvertToMono(buffer, audioFile.WaveFormat.Channels);
        }
        if (audioFile.WaveFormat.SampleRate != targetSampleRate)
        {
            buffer = Resample(buffer.ToArray(), audioFile.WaveFormat.SampleRate, targetSampleRate).ToList();
        }
        return buffer.ToArray();
    }

    public static List<float> ConvertToMono(List<float> input, int channels)
    {
        var monoBuffer = new List<float>();
        for (int i = 0; i < input.Count; i += channels)
        {
            float sum = 0;
            for (int ch = 0; ch < channels; ch++)
            {
                sum += input[i + ch];
            }
            monoBuffer.Add(sum / channels);
        }
        return monoBuffer;
    }

    private static float[] Resample(float[] input, int sourceSampleRate, int targetSampleRate)
    {
        var ratio = (double)targetSampleRate / sourceSampleRate;
        var outputLength = (int)(input.Length * ratio);
        var output = new float[outputLength];

        for (int i = 0; i < outputLength; i++)
        {
            var position = i / ratio;
            var index = (int)position;
            var fraction = position - index;

            if (index >= input.Length - 1)
            {
                output[i] = input[input.Length - 1];
            }
            else
            {
                output[i] = (float)((1 - fraction) * input[index] +
                           fraction * input[index + 1]);
            }
        }

        return output;
    }

    private static Tensor ComputeSTFT(Tensor audio, int nFft, int hopLength)
    {
        var window = hann_window(nFft);

        // Pad signal
        var padSize = nFft / 2;
        var paddedAudio = nn.functional.pad(audio,
            new long[] { padSize, padSize }, PaddingModes.Reflect);

        // Prepare frames
        var frames = new List<Tensor>();
        for (int i = 0; i < paddedAudio.size(-1) - nFft; i += hopLength)
        {
            var frame = paddedAudio.narrow(-1, i, nFft) * window;
            frames.Add(frame);
        }

        var stacked = stack(frames.ToArray(), 1);

        // Apply FFT
        var fft = torch.fft.rfft(stacked, nFft);
        return fft;
    }

    private static Tensor GenerateMelSpectrogram(Tensor audio, int sampleRate,
        int nFft = 2048, int hopLength = 512, int nMels = 80,
        float fMin = 0.0f, float fMax = 8000.0f)
    {
        var stft = ComputeSTFT(audio, nFft, hopLength);

        // Convert to power spectrogram and reshape
        var power = stft.abs().pow(2);

        // Permute to get frequency bins as last dimension
        power = power.squeeze(0).transpose(0, 1);  // Now [frames, freq_bins]

        // Create and apply Mel filterbank
        var melBasis = CreateMelFilterbank(sampleRate, nFft, nMels, fMin, fMax);
        var melSpec = matmul(power, melBasis.transpose(0, 1));  // [frames, n_mels]

        // Convert to log scale
        return log10(melSpec + 1e-9f);
    }

    private static Tensor CreateMelFilterbank(int sampleRate, int nFft, int nMels,
        float fMin, float fMax)
    {
        // Convert Hz to Mel scale
        float HzToMel(float hz) =>
            2595.0f * (float)Math.Log10(1.0f + hz / 700.0f);

        float MelToHz(float mel) =>
            700.0f * ((float)Math.Pow(10, mel / 2595.0f) - 1.0f);

        // Create Mel points
        var melPoints = linspace(
            HzToMel(fMin),
            HzToMel(fMax),
            nMels + 2
        );

        // Convert back to Hz
        var hzPointsData = new float[melPoints.shape[0]];
        for (int i = 0; i < melPoints.shape[0]; i++)
        {
            hzPointsData[i] = MelToHz(melPoints[i].item<float>());
        }
        var hzPoints = tensor(hzPointsData);

        // Convert to FFT bins
        var bins = (hzPoints * (nFft + 1) / sampleRate).round().to(ScalarType.Int64);

        // Create filterbank matrix
        var fbank = zeros(nMels, nFft / 2 + 1);

        // Fill the filterbank matrix
        for (int i = 0; i < nMels; i++)
        {
            for (long j = bins[i].item<long>(); j < bins[i + 2].item<long>(); j++)
            {
                if (j < bins[i + 1].item<long>())
                {
                    fbank[i, j] = (j - bins[i].item<long>()) /
                                 (float)(bins[i + 1].item<long>() - bins[i].item<long>());
                }
                else
                {
                    fbank[i, j] = (bins[i + 2].item<long>() - j) /
                                 (float)(bins[i + 2].item<long>() - bins[i + 1].item<long>());
                }
            }
        }

        return fbank;
    }

    private static void SaveSpectrogram(Tensor spec, string path)
    {
        // Convert to numpy-like array and transpose to get time on x-axis
        var data = spec.cpu().detach().squeeze().transpose(0, 1).to(ScalarType.Float32);

        // Normalize to 0-255 range
        var min = data.min().item<float>();
        var max = data.max().item<float>();
        var normalized = ((data - min) / (max - min) * 255).to(ScalarType.Byte);

        // width is time dimension
        var width = (int)normalized.size(1);  // time
        var height = (int)normalized.size(0);  // frequency

        using var bitmap = new SKBitmap(width, height);
        using var canvas = new SKCanvas(bitmap);

        // Apply viridis-like colormap
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var value = normalized[y, x].item<byte>();
                var color = ApplyColormap(value);
                bitmap.SetPixel(x, height - 1 - y, new SKColor(color.R, color.G, color.B)); // Flip Y to put low frequencies at bottom
            }
        }

        using var image = SKImage.FromBitmap(bitmap);
        using var skData = image.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.OpenWrite(path);
        skData.SaveTo(stream);
    }

    private static void SaveComparisonSpectrogram(Tensor originalSpec, Tensor encodedSpec, string path)
    {
        var diffSpec = (originalSpec - encodedSpec).abs();

        // Process each spectrogram to get consistent dimensions
        var specs = new[] { originalSpec, encodedSpec, diffSpec };
        var processedSpecs = specs.Select(spec =>
        {
            var data = spec.cpu().detach().squeeze().transpose(0, 1).to(ScalarType.Float32);
            var min = data.min().item<float>();
            var max = data.max().item<float>();
            return ((data - min) / (max - min) * 255).to(ScalarType.Byte);
        }).ToArray();

        // Get dimensions
        var width = (int)processedSpecs[0].size(1);  // time
        var height = (int)processedSpecs[0].size(0);  // frequency
        var totalHeight = height * 3;  // Stack three spectrograms

        using var bitmap = new SKBitmap(width, totalHeight);
        using var canvas = new SKCanvas(bitmap);

        // Process each spectrogram into its section of the image
        for (int specIndex = 0; specIndex < 3; specIndex++)
        {
            var currentSpec = processedSpecs[specIndex];
            var yOffset = specIndex * height;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var value = currentSpec[y, x].item<byte>();
                    var color = ApplyColormap(value);
                    // Calculate position in combined image, maintaining Y-flip for each section
                    var combinedY = yOffset + (height - 1 - y);
                    bitmap.SetPixel(x, combinedY, new SKColor(color.R, color.G, color.B));
                }
            }
        }

        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.OpenWrite(path);
        data.SaveTo(stream);
    }

    private static void SaveDifference(Tensor spec1, Tensor spec2, string path)
    {
        var diff = (spec1 - spec2).abs();
        SaveSpectrogram(diff, path);
    }

    private static (byte R, byte G, byte B) ApplyColormap(byte value)
    {
        // Simple approximation of viridis colormap
        float normalized = value / 255.0f;
        return (
            R: (byte)(normalized * 255),
            G: (byte)(Math.Sin(normalized * Math.PI) * 255),
            B: (byte)((1 - normalized) * 255)
        );
    }
}