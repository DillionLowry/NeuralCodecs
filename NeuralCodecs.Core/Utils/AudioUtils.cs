using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace NeuralCodecs.Core.Utils
{
    public static class AudioUtils
    {
        private static Func<string, WaveFileReader> _readerFactory = path => new WaveFileReader(path);

        /// <summary>
        /// Converts multi-channel audio to mono by averaging channels
        /// </summary>
        public static float[] ConvertToMono(float[] input, int channels)
        {
            var monoLength = input.Length / channels;
            var output = new float[monoLength];

            for (int i = 0; i < monoLength; i++)
            {
                float sum = 0;
                int inputOffset = i * channels;
                for (int ch = 0; ch < channels; ch++)
                {
                    sum += input[inputOffset + ch];
                }
                output[i] = sum / channels;
            }

            return output;
        }

        /// <summary>
        /// Converts a list of multi-channel audio samples to mono
        /// </summary>
        public static List<float> ConvertToMono(List<float> input, int channels)
        {
            var monoBuffer = new List<float>(input.Count / channels);
            for (int i = 0; i < input.Count; i += channels)
            {
                float sum = 0;
                for (int ch = 0; ch < channels && i + ch < input.Count; ch++)
                {
                    sum += input[i + ch];
                }
                monoBuffer.Add(sum / channels);
            }
            return monoBuffer;
        }

        /// <summary>
        /// Converts deinterleaved stereo audio to interleaved format (LRLRLRLR...)
        /// </summary>
        public static float[] DeinterleaveToInterleave(float[] deinterleavedData)
        {
            var samplesPerChannel = deinterleavedData.Length / 2;
            var result = new float[deinterleavedData.Length];

            for (int i = 0; i < samplesPerChannel; i++)
            {
                result[i * 2] = deinterleavedData[i];                         // Left channel
                result[(i * 2) + 1] = deinterleavedData[i + samplesPerChannel]; // Right channel
            }

            return result;
        }

        /// <summary>
        /// Deinterleaves a WAV file's bytes to arrange channels sequentially
        /// </summary>
        /// <param name="inputFilePath">Path to the stereo audio file</param>
        /// <returns>Deinterleaved byte array</returns>
        public static byte[] DeinterleaveWavFile(string inputFilePath)
        {
            using var reader = CreateReader(inputFilePath);

            // Ensure we have a stereo file
            if (reader.WaveFormat.Channels != 2)
            {
                throw new ArgumentException("Input file must be stereo (2 channels)");
            }

            // Get total number of samples (per channel)
            int bytesPerSample = reader.WaveFormat.BitsPerSample / 8;
            long totalSamples = reader.Length / (bytesPerSample * 2);

            // Create buffer for the deinterleaved data
            byte[] deinterleavedData = new byte[reader.Length];

            // Create buffer for reading the original file
            byte[] originalData = new byte[reader.Length];
            if (reader.Read(originalData, 0, originalData.Length)==0)
            {
                throw new ArgumentException("Input file is empty");
            }

            // Deinterleave the data
            long leftChannelOffset = 0;
            long rightChannelOffset = totalSamples * bytesPerSample;

            for (long i = 0; i < totalSamples; i++)
            {
                // Calculate source positions
                long sourceOffset = i * bytesPerSample * 2;

                // Copy left channel sample
                Buffer.BlockCopy(
                    originalData,
                    (int)sourceOffset,
                    deinterleavedData,
                    (int)leftChannelOffset,
                    bytesPerSample);
                leftChannelOffset += bytesPerSample;

                // Copy right channel sample
                Buffer.BlockCopy(
                    originalData,
                    (int)(sourceOffset + bytesPerSample),
                    deinterleavedData,
                    (int)rightChannelOffset,
                    bytesPerSample);
                rightChannelOffset += bytesPerSample;
            }

            return deinterleavedData;
        }

        /// <summary>
        /// Converts interleaved stereo audio to deinterleaved format (LLLLL...RRRRR...)
        /// </summary>
        public static float[] InterleaveToDeinterleave(float[] interleavedData)
        {
            var samplesPerChannel = interleavedData.Length / 2;
            var result = new float[interleavedData.Length];

            for (int i = 0; i < samplesPerChannel; i++)
            {
                result[i] = interleavedData[i * 2];                         // Left channel
                result[i + samplesPerChannel] = interleavedData[(i * 2) + 1]; // Right channel
            }

            return result;
        }
        public static float[,] InterleaveToDeinterleave2d(float[] interleavedData)
        {
            var samplesPerChannel = interleavedData.Length ;
            float[,] result = new float[2, samplesPerChannel];

            for (int i = 0; i < samplesPerChannel-1; i++)
            {
                result[0, i]= interleavedData[i];
                result[1, i] = interleavedData[i + 1];
            }

            return result;
        }
        /// <summary>
        /// Applies layer normalization to the input array across the channel dimension.
        /// </summary>
        /// <param name="input">Input array to normalize.</param>
        /// <param name="channels">Number of channels in the input.</param>
        /// <param name="weight">Optional scaling factors for each channel.</param>
        /// <param name="bias">Optional bias terms for each channel.</param>
        /// <returns>Normalized array with the same shape as the input.</returns>
        /// <exception cref="ArgumentException">Thrown when input length is not divisible by the number of channels.</exception>
        public static float[] LayerNorm(float[] input, int channels, float[]? weight = null, float[]? bias = null)
        {
            if (input.Length % channels != 0)
            {
                throw new ArgumentException("Input length must be divisible by number of channels");
            }

            int timeSteps = input.Length / channels;
            var output = new float[input.Length];

            for (int t = 0; t < timeSteps; t++)
            {
                var slice = new Span<float>(input, t * channels, channels);
                var normalized = Normalize(slice.ToArray());

                for (int c = 0; c < channels; c++)
                {
                    float value = normalized[c];
                    if (weight != null)
                    {
                        value *= weight[c];
                    }

                    if (bias != null)
                    {
                        value += bias[c];
                    }

                    output[(t * channels) + c] = value;
                }
            }

            return output;
        }

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
        /// Applies statistical normalization to standardize the input array.
        /// </summary>
        /// <param name="input">Array to normalize.</param>
        /// <param name="epsilon">Small constant for numerical stability. Defaults to 1e-5.</param>
        /// <returns>Normalized array with zero mean and unit variance.</returns>
        public static float[] Normalize(float[] input, float epsilon = 1e-5f)
        {
            float mean = 0;
            float variance = 0;

            // Calculate mean
            for (int i = 0; i < input.Length; i++)
            {
                mean += input[i];
            }

            mean /= input.Length;

            // Calculate variance
            for (int i = 0; i < input.Length; i++)
            {
                float diff = input[i] - mean;
                variance += diff * diff;
            }
            variance /= input.Length;

            // Normalize
            var output = new float[input.Length];
            float std = (float)Math.Sqrt(variance + epsilon);
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (input[i] - mean) / std;
            }

            return output;
        }

        /// <summary>
        /// Performs linear interpolation resampling
        /// </summary>
        /// <remarks>
        /// This is a simpler but lower quality resampler than WDL
        /// </remarks>
        public static float[] ResampleLinear(float[] input, int sourceSampleRate, int targetSampleRate)
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
                    output[i] = (float)(((1 - fraction) * input[index]) +
                               (fraction * input[index + 1]));
                }
            }

            return output;
        }

        /// <summary>
        /// Reshapes a flat array into a new shape while preserving the total number of elements.
        /// </summary>
        /// <param name="input">Input array to reshape.</param>
        /// <param name="shape">Target shape dimensions.</param>
        /// <returns>Array with the same data in the new shape.</returns>
        /// <exception cref="ArgumentException">Thrown when the new shape's total size doesn't match the input length.</exception>
        public static float[] Reshape(float[] input, params int[] shape)
        {
            int totalSize = 1;
            foreach (int dim in shape)
            {
                totalSize *= dim;
            }

            if (totalSize != input.Length)
            {
                throw new ArgumentException("New shape must have same total size as input");
            }

            return input.ToArray(); // Since we're working with flat arrays
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
                buffer = DeinterleaveToInterleave(buffer);
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

            float[] buffer = ConvertToFloat(audioData, bitDepth, totalSamples, channels);

            if (mono && channels > 1)
            {
                buffer = ConvertToMono(buffer, channels);
                channels = 1;
            }

            if (reader.WaveFormat.SampleRate != targetSampleRate)
            {
                buffer = ResampleUsingNAudio(buffer, channels, reader.WaveFormat.SampleRate, targetSampleRate);
            }

            if (channels == 2 && format == AudioInterleaving.Deinterleaved)
            {
                buffer = InterleaveToDeinterleave(buffer);
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

            float[] buffer = ConvertToFloat(audioData, bitDepth, totalSamplesPerChannel * 2, 2);

            if (reader.WaveFormat.SampleRate != targetSampleRate)
            {
                buffer = ResampleUsingNAudio(buffer, 2, reader.WaveFormat.SampleRate, targetSampleRate);
            }

            if (format == AudioInterleaving.Deinterleaved)
            {
                buffer = InterleaveToDeinterleave(buffer);
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
        /// Converts byte array audio data to float array
        /// </summary>
        private static float[] ConvertToFloat(byte[] audioData, int bitDepth, long totalSamples, int channels)
        {
            var result = new float[totalSamples];

            if (bitDepth == 16)
            {
                const float scale = 1.0f / 32768.0f;
                for (long i = 0; i < totalSamples; i++)
                {
                    int offset = (int)(i * 2);
                    result[i] = BitConverter.ToInt16(audioData, offset) * scale;
                }
            }
            else // 32-bit
            {
                const float scale = 1.0f / 2147483648.0f;
                for (long i = 0; i < totalSamples; i++)
                {
                    int offset = (int)(i * 4);
                    result[i] = BitConverter.ToInt32(audioData, offset) * scale;
                }
            }

            return result;
        }

        /// <summary>
        /// Creates a wave file reader using the configured factory
        /// </summary>
        private static WaveFileReader CreateReader(string path)
        {
            return _readerFactory(path);
        }

        /// <summary>
        /// Resamples audio using NAudio's high-quality resampler
        /// </summary>
        private static float[] ResampleUsingNAudio(float[] input, int channels, int sourceSampleRate, int targetSampleRate)
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
        /// Helper class for loading float arrays as NAudio sample providers
        /// </summary>
        private class FloatArraySampleProvider : ISampleProvider
        {
            private readonly float[] _samples;
            private int _position;

            public FloatArraySampleProvider(WaveFormat waveFormat, float[] samples)
            {
                WaveFormat = waveFormat;
                _samples = samples;
            }

            public WaveFormat WaveFormat { get; }

            public int Read(float[] buffer, int offset, int count)
            {
                var availableSamples = Math.Min(count, _samples.Length - _position);
                Array.Copy(_samples, _position, buffer, offset, availableSamples);
                _position += availableSamples;
                return availableSamples;
            }
        }
    }
}