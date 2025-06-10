using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace NeuralCodecs.Core.Utils;

internal static class NAudioUtils
{
    private static Func<string, WaveFileReader> _readerFactory = path => new WaveFileReader(path);

    /// <summary>
    /// Loads audio from a file and converts it to float array format
    /// </summary>
    /// <param name="path">Path to the audio file</param>
    /// <param name="targetSampleRate">Desired sample rate, will resample if needed</param>
    /// <param name="format">Format of the output data (interleaved or deinterleaved)</param>
    /// <param name="mono">Whether to convert to mono (single channel)</param>
    /// <param name="bitDepth">Bit depth (16 or 32)</param>
    /// <returns>Audio data as float array with values from -1.0 to 1.0</returns>
    public static float[] LoadAudio(string path, int targetSampleRate,
        AudioInterleaving format = AudioInterleaving.Deinterleaved, bool mono = true, int bitDepth = 16)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            throw new FileNotFoundException("Input file does not exist", path);
        }

        using var reader = CreateReader(path);
        return LoadAudioFromReader(reader, targetSampleRate, format, mono, bitDepth);
    }

    /// <summary>
    /// Loads stereo audio from a file into a float array
    /// </summary>
    /// <param name="path">Path to the audio file</param>
    /// <param name="targetSampleRate">Desired sample rate</param>
    /// <param name="format">Format of the output data (interleaved or deinterleaved)</param>
    /// <param name="bitDepth">Bit depth (16 or 32)</param>
    /// <returns>Stereo audio data as float array</returns>
    public static float[] LoadStereoAudio(string path, int targetSampleRate,
        AudioInterleaving format = AudioInterleaving.Interleaved, int bitDepth = 16)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            throw new FileNotFoundException("Input file does not exist", path);
        }

        using var reader = CreateReader(path);
        return LoadStereoAudioFromReader(reader, targetSampleRate, format, bitDepth);
    }

    /// <summary>
    /// Loads stereo audio in deinterleaved format to a multidimensional array [channel, sample]
    /// </summary>
    /// <param name="filePath">Path to the audio file</param>
    /// <param name="targetSampleRate">Desired sample rate</param>
    /// <param name="bitDepth">Bit depth (16 or 32)</param>
    /// <returns>Stereo audio as [2, samples] float array</returns>
    public static float[,] LoadStereoAudioMultidimensional(string filePath, int targetSampleRate, int bitDepth = 16)
    {
        if (bitDepth is not 16 and not 32)
        {
            throw new ArgumentException("Bit depth must be either 16 or 32");
        }

        using var reader = CreateReader(filePath);

        if (reader.WaveFormat.Channels != 2)
        {
            throw new ArgumentException("Input file must be stereo (2 channels)");
        }

        if (reader.WaveFormat.BitsPerSample != bitDepth)
        {
            throw new ArgumentException($"Input file has {reader.WaveFormat.BitsPerSample} bits per sample, but {bitDepth} was requested");
        }

        if (reader.WaveFormat.SampleRate != targetSampleRate)
        {
            throw new NotImplementedException("Resampling for multidimensional stereo is not implemented");
        }

        int bytesPerSample = bitDepth / 8;
        long totalSamplesPerChannel = reader.Length / (bytesPerSample * 2);
        float[,] result = new float[2, (int)totalSamplesPerChannel];

        byte[] audioData = new byte[reader.Length];
        if (reader.Read(audioData, 0, audioData.Length) == 0)
        {
            throw new ArgumentException("Input file is empty");
        }

        if (bitDepth == 16)
        {
            // Convert value range -32768 to 32767 => -1.0 to 1.0
            for (long i = 0; i < totalSamplesPerChannel; i++)
            {
                int leftOffset = (int)(i * 2 * bytesPerSample);
                short leftSample = BitConverter.ToInt16(audioData, leftOffset);
                result[0, (int)i] = leftSample / 32768f;

                // Get right channel sample and convert to float
                int rightOffset = (int)((i * 2 * bytesPerSample) + bytesPerSample);
                short rightSample = BitConverter.ToInt16(audioData, rightOffset);
                result[1, (int)i] = rightSample / 32768f;
            }
        }
        else
        {
            // Convert value range -2147483648 to 2147483647 => -1.0 to 1.0
            for (long i = 0; i < totalSamplesPerChannel; i++)
            {
                int leftOffset = (int)(i * 2 * bytesPerSample);
                int leftSample = BitConverter.ToInt32(audioData, leftOffset);
                result[0, (int)i] = leftSample / 2147483648f;

                int rightOffset = (int)((i * 2 * bytesPerSample) + bytesPerSample);
                int rightSample = BitConverter.ToInt32(audioData, rightOffset);
                result[1, (int)i] = rightSample / 2147483648f;
            }
        }

        return result;
    }

    /// <summary>
    /// Resamples audio using NAudio's high-quality resampler
    /// </summary>
    public static float[] Resample(float[] input, int channels, int sourceSampleRate, int targetSampleRate)
    {
        var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(sourceSampleRate, channels);
        var resampler = new WdlResamplingSampleProvider(
            new FloatArraySampleProvider(waveFormat, input),
            targetSampleRate
        );

        var outputLength = (int)((long)input.Length * targetSampleRate / sourceSampleRate);
        if (outputLength % channels != 0)
        {
            outputLength += channels - (outputLength % channels);
        }

        var output = new float[outputLength];
        resampler.Read(output, 0, output.Length);

        return output;
    }

    /// <summary>
    /// Saves audio data to a WAV file
    /// </summary>
    /// <param name="path">Output file path</param>
    /// <param name="buffer">Audio data as float array</param>
    /// <param name="sampleRate">Sample rate in Hz</param>
    /// <param name="channels">Number of channels</param>
    /// <param name="format">Format of the input data (interleaved or deinterleaved)</param>
    /// <param name="bitDepth">Bit depth (16 or 32)</param>
    public static void SaveAudio(string path, float[] buffer, int sampleRate, int channels,
        AudioInterleaving format = AudioInterleaving.Interleaved, int bitDepth = 16)
    {
        if (buffer == null)
        {
            throw new ArgumentNullException(nameof(buffer));
        }

        if (channels <= 0)
        {
            throw new ArgumentException("Channel count must be greater than 0", nameof(channels));
        }

        if (bitDepth is not 16 and not 32)
        {
            throw new ArgumentException("Bit depth must be either 16 or 32");
        }

        if (channels == 2 && format == AudioInterleaving.Deinterleaved)
        {
            buffer = AudioUtils.DeinterleaveToInterleave(buffer);
        }

        var waveFormat = bitDepth == 16
            ? new WaveFormat(sampleRate, channels)
            : WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels);

        using var writer = new WaveFileWriter(path, waveFormat);
        writer.WriteSamples(buffer, 0, buffer.Length);
    }

    /// <summary>
    /// Loads audio from a reader and converts it to float array format
    /// </summary>
    internal static float[] LoadAudioFromReader(WaveFileReader reader, int targetSampleRate,
        AudioInterleaving format = AudioInterleaving.Deinterleaved, bool mono = true, int bitDepth = 16)
    {
        if (bitDepth is not 16 and not 32)
        {
            throw new ArgumentException("Bit depth must be either 16 or 32");
        }

        if (reader.WaveFormat.BitsPerSample != bitDepth)
        {
            throw new ArgumentException($"Input file has {reader.WaveFormat.BitsPerSample} bits per sample, but {bitDepth} was requested");
        }

        var channels = reader.WaveFormat.Channels;
        var bytesPerSample = bitDepth / 8;
        var totalSamples = reader.Length / (bytesPerSample * channels);

        byte[] audioData = new byte[reader.Length];
        if (reader.Read(audioData, 0, audioData.Length) == 0)
        {
            throw new ArgumentException("Input file is empty");
        }

        float[] buffer = AudioUtils.AudioBytesToFloatArray(audioData, bitDepth, totalSamples, channels);

        if (mono && channels > 1)
        {
            buffer = AudioUtils.ConvertToMono(buffer, channels);
            channels = 1;
        }

        if (reader.WaveFormat.SampleRate != targetSampleRate)
        {
            buffer = NAudioUtils.Resample(buffer, channels, reader.WaveFormat.SampleRate, targetSampleRate);
        }

        if (channels == 2 && format == AudioInterleaving.Deinterleaved)
        {
            buffer = AudioUtils.InterleaveToDeinterleave(buffer);
        }

        return buffer;
    }

    /// <summary>
    /// Loads stereo audio from a reader into a float array
    /// </summary>
    internal static float[] LoadStereoAudioFromReader(WaveFileReader reader, int targetSampleRate,
        AudioInterleaving format = AudioInterleaving.Interleaved, int bitDepth = 16)
    {
        if (bitDepth is not 16 and not 32)
        {
            throw new ArgumentException("Bit depth must be either 16 or 32");
        }

        if (reader.WaveFormat.Channels != 2)
        {
            throw new ArgumentException("Input file must be stereo (2 channels)");
        }

        if (reader.WaveFormat.BitsPerSample != bitDepth)
        {
            throw new ArgumentException($"Input file has {reader.WaveFormat.BitsPerSample} bits per sample, but {bitDepth} was requested");
        }

        var bytesPerSample = bitDepth / 8;
        var totalSamplesPerChannel = reader.Length / (bytesPerSample * 2);

        byte[] audioData = new byte[reader.Length];
        if (reader.Read(audioData, 0, audioData.Length) == 0)
        {
            throw new ArgumentException("Input file is empty");
        }

        float[] buffer = AudioUtils.AudioBytesToFloatArray(audioData, bitDepth, totalSamplesPerChannel * 2, 2);

        if (reader.WaveFormat.SampleRate != targetSampleRate)
        {
            buffer = NAudioUtils.Resample(buffer, 2, reader.WaveFormat.SampleRate, targetSampleRate);
        }

        if (format == AudioInterleaving.Deinterleaved)
        {
            buffer = AudioUtils.InterleaveToDeinterleave(buffer);
        }

        return buffer;
    }

    /// <summary>
    /// Sets a custom reader factory for creating WaveFileReader instances
    /// </summary>
    internal static void SetReaderFactory(Func<string, WaveFileReader> factory)
    {
        _readerFactory = factory;
    }

    /// <summary>
    /// Creates a wave file reader using the configured factory
    /// </summary>
    internal static WaveFileReader CreateReader(string path)
    {
        return _readerFactory(path);
    }

    /// <summary>
    /// Helper class for providing float arrays as NAudio sample providers
    /// </summary>
    public class FloatArraySampleProvider : ISampleProvider
    {
        private readonly float[] _samples;
        private int _position;

        public FloatArraySampleProvider(WaveFormat waveFormat, float[] samples)
        {
            WaveFormat = waveFormat;
            _samples = samples;
            _position = 0;
        }

        public WaveFormat WaveFormat { get; }

        public int Read(float[] buffer, int offset, int count)
        {
            var availableSamples = Math.Min(count, _samples.Length - _position);
            if (availableSamples > 0)
            {
                Array.Copy(_samples, _position, buffer, offset, availableSamples);
                _position += availableSamples;
            }
            return availableSamples;
        }
    }
}