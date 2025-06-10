namespace NeuralCodecs.Core.Utils
{
    public static class AudioUtils
    {
        /// <summary>
        /// Converts 16 or 32-bit PCM byte array to float array
        /// </summary>
        /// <param name="audioData">The PCM audio data as a byte array</param>
        /// <param name="bitDepth">The bit depth of the audio (16 or 32)</param>
        /// <param name="totalSamples">Total number of samples in the audio data</param>
        /// <param name="channels">Number of audio channels</param>
        /// <returns>Audio data as a normalized float array</returns>
        public static float[] AudioBytesToFloatArray(byte[] audioData, int bitDepth, long totalSamples, int channels)
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
        /// Converts multi-channel audio to mono by averaging channels
        /// </summary>
        /// <param name="input">Multi-channel audio data</param>
        /// <param name="channels">Number of channels in the input audio</param>
        /// <returns>Mono audio data with averaged channels</returns>
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
        /// <param name="input">Multi-channel audio data as a list</param>
        /// <param name="channels">Number of channels in the input audio</param>
        /// <returns>Mono audio data with averaged channels as a list</returns>
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
        /// <param name="deinterleavedData">Deinterleaved stereo audio data (left channel followed by right channel)</param>
        /// <returns>Interleaved stereo audio data in LRLRLR format</returns>
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
            using var reader = NAudioUtils.CreateReader(inputFilePath);

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
            if (reader.Read(originalData, 0, originalData.Length) == 0)
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
        /// Converts float array to PCM byte array with specified bit depth
        /// </summary>
        /// <param name="audioData">Audio data as normalized float array</param>
        /// <param name="bitDepth">Target bit depth (16 or 32)</param>
        /// <param name="channels">Number of audio channels</param>
        /// <returns>PCM audio data as byte array</returns>
        /// <exception cref="ArgumentException">Thrown when bit depth is not 16 or 32</exception>
        public static byte[] FloatArrayToAudioBytes(float[] audioData, int bitDepth, int channels)
        {
            if (bitDepth is not 16 and not 32)
            {
                throw new ArgumentException("Bit depth must be either 16 or 32");
            }
            var bytesPerSample = bitDepth / 8;
            var byteArray = new byte[audioData.Length * bytesPerSample];
            if (bitDepth == 16)
            {
                for (int i = 0; i < audioData.Length; i++)
                {
                    short sample = (short)(audioData[i] * short.MaxValue);
                    Buffer.BlockCopy(BitConverter.GetBytes(sample), 0, byteArray, i * bytesPerSample, bytesPerSample);
                }
            }
            else // 32-bit
            {
                for (int i = 0; i < audioData.Length; i++)
                {
                    int sample = (int)(audioData[i] * int.MaxValue);
                    Buffer.BlockCopy(BitConverter.GetBytes(sample), 0, byteArray, i * bytesPerSample, bytesPerSample);
                }
            }
            return byteArray;
        }

        /// <summary>
        /// Converts interleaved stereo audio to deinterleaved format (LLLLL...RRRRR...)
        /// </summary>
        /// <param name="interleavedData">Interleaved stereo audio data in LRLRLR format</param>
        /// <returns>Deinterleaved stereo audio data (left channel followed by right channel)</returns>
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

        /// <summary>
        /// Converts interleaved audio data to a 2D array format for processing
        /// </summary>
        /// <param name="interleavedData">Interleaved audio data</param>
        /// <returns>2D array where first dimension represents channels and second represents samples</returns>
        public static float[,] InterleaveToDeinterleave2d(float[] interleavedData)
        {
            var samplesPerChannel = interleavedData.Length;
            float[,] result = new float[2, samplesPerChannel];

            for (int i = 0; i < samplesPerChannel - 1; i++)
            {
                result[0, i] = interleavedData[i];
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
        /// <param name="input">Input audio data to resample</param>
        /// <param name="sourceSampleRate">Original sample rate in Hz</param>
        /// <param name="targetSampleRate">Target sample rate in Hz</param>
        /// <returns>Resampled audio data</returns>
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
                    output[i] = input[^1];
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
    }
}