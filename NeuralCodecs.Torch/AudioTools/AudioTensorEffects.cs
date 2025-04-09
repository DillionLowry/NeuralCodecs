using static TorchSharp.torch;

namespace NeuralCodecs.Torch.AudioTools
{
    /// <summary>
    /// Provides a collection of audio processing effects for tensor and AudioSignal manipulation.
    /// Implements common digital signal processing algorithms including reverb, delay, modulation,
    /// and dynamic processing effects.
    /// </summary>
    public static partial class AudioEffects
    {
        /// <summary>
        /// Applies a compressor effect to an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="threshold">Threshold in dB.</param>
        /// <param name="ratio">Compression ratio.</param>
        /// <param name="attackTime">Attack time in seconds.</param>
        /// <param name="releaseTime">Release time in seconds.</param>
        /// <param name="makeupGain">Makeup gain in dB.</param>
        /// <returns>Processed audio tensor.</returns>
        public static Tensor ApplyCompressor(
            Tensor audio,
            int sampleRate,
            float threshold = -20.0f,
            float ratio = 4.0f,
            float attackTime = 0.005f,
            float releaseTime = 0.050f,
            float makeupGain = 0.0f)
        {
            var input = audio;

            // Convert threshold from dB to linear
            var thresholdLin = exp(threshold / 20.0f * (float)Math.Log(10));

            // Calculate time constants in samples
            var attackSamples = (int)(attackTime * sampleRate);
            var releaseSamples = (int)(releaseTime * sampleRate);

            // Initialize envelope follower
            var envelope = zeros_like(input);
            var currentLevel = zeros(input.size(0), input.size(1), device: input.device);

            // Apply envelope detection
            for (int n = 0; n < input.size(-1); n++)
            {
                var inputLevel = input.index(TensorIndex.Ellipsis, n).abs();

                // Calculate smoothing coefficients
                var attackGain = 1.0f - (float)Math.Exp(-1.0f / attackSamples);
                var releaseGain = 1.0f - (float)Math.Exp(-1.0f / releaseSamples);

                // Choose attack or release based on level
                var gain = where(inputLevel > currentLevel, attackGain, releaseGain);

                // Smooth the level
                currentLevel.add_(gain * (inputLevel - currentLevel));

                // Store in envelope
                envelope.index_put_(currentLevel, TensorIndex.Ellipsis, n);
            }

            // Apply gain reduction
            var gainReduction = ones_like(envelope);
            var mask = envelope > thresholdLin;

            // Apply compression formula
            gainReduction.masked_scatter_(mask,
                pow(envelope.masked_select(mask) / thresholdLin, (1.0f / ratio) - 1.0f));

            // Apply makeup gain
            var makeupGainLin = exp(makeupGain / 20.0f * (float)Math.Log(10));

            // Apply gain reduction and makeup gain
            return input * gainReduction * makeupGainLin;
        }

        /// <summary>
        /// Applies a delay effect to an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="sampleRate">Sample rate of the audio.</param>
        /// <param name="delayTime">Delay time in seconds.</param>
        /// <param name="feedback">Feedback coefficient [0-1].</param>
        /// <param name="wetLevel">Wet level [0-1].</param>
        /// <param name="dryLevel">Dry level [0-1].</param>
        /// <returns>Processed audio tensor.</returns>
        public static Tensor ApplyDelay(
            Tensor audio,
            int sampleRate,
            float delayTime = 0.3f,
            float feedback = 0.3f,
            float wetLevel = 0.3f,
            float dryLevel = 0.7f)
        {
            int delaySamples = (int)(delayTime * sampleRate);
            var buffer = zeros(audio.size(0), audio.size(1), delaySamples, device: audio.device);
            var delayed = zeros_like(audio);

            // Process each sample
            for (int n = 0; n < audio.size(-1); n++)
            {
                var current_input = audio.index(TensorIndex.Ellipsis, n);
                var delayed_signal = buffer.index(TensorIndex.Ellipsis, -1);
                delayed.index_put_(delayed_signal, TensorIndex.Ellipsis, n);

                // Roll buffer (shift by 1)
                using var indices = arange(0, buffer.size(-1) - 1, device: buffer.device);
                buffer.index_copy_(-1, indices.add(1), buffer.index(TensorIndex.Ellipsis, indices));

                // Add current input with feedback to buffer
                buffer.index_put_(current_input + (delayed_signal * feedback), TensorIndex.Ellipsis, 0);
            }

            // Combine dry and wet signals
            return (dryLevel * audio) + (wetLevel * delayed);
        }

        /// <summary>
        /// Applies a distortion effect to an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="amount">Distortion amount [0-1].</param>
        /// <param name="wetLevel">Wet level [0-1].</param>
        /// <returns>Processed audio tensor.</returns>
        public static Tensor ApplyDistortion(
            Tensor audio,
            float amount = 0.5f,
            float wetLevel = 1.0f)
        {
            var input = audio;
            var processed = tanh(input * (1 + (amount * 10)));
            return (processed * wetLevel) + (input * (1 - wetLevel));
        }

        /// <summary>
        /// Applies a flanger effect to an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="sampleRate">Sample rate of the audio.</param>
        /// <param name="rate">Modulation rate in Hz.</param>
        /// <param name="depth">Modulation depth in seconds.</param>
        /// <param name="feedback">Feedback coefficient [0-1].</param>
        /// <param name="wetLevel">Wet level [0-1].</param>
        /// <returns>Processed audio tensor.</returns>
        public static Tensor ApplyFlanger(
            Tensor audio,
            int sampleRate,
            float rate = 0.5f,
            float depth = 0.002f,
            float feedback = 0.7f,
            float wetLevel = 0.7f)
        {
            var input = audio;
            var samples = input.size(-1);

            // Create time vector
            var time = linspace(0, (float)samples / sampleRate, samples, device: audio.device);

            // Calculate max delay in samples
            var maxDelaySamples = (int)(depth * sampleRate);

            // Generate LFO
            var lfo = maxDelaySamples * 0.5f * (1 + sin(2 * (float)Math.PI * rate * time));

            // Initialize output and buffer
            var processed = zeros_like(input);
            var buffer = zeros(input.size(0), input.size(1), maxDelaySamples + 1, device: input.device);

            for (int n = 0; n < input.size(-1); n++)
            {
                var currentInput = input.index(TensorIndex.Ellipsis, n);
                var delayLength = lfo[n];

                // Calculate interpolation indices and weights
                var delayFloor = floor(delayLength).to(ScalarType.Int64);
                var delayFrac = delayLength - delayFloor;

                // Get delayed samples (with linear interpolation)
                var delayed1 = buffer.index(TensorIndex.Ellipsis, delayFloor);
                var delayed2 = buffer.index(TensorIndex.Ellipsis, delayFloor + 1);
                var delayedSignal = (delayed1 * (1 - delayFrac)) + (delayed2 * delayFrac);

                // Roll buffer
                using var indices = arange(0, buffer.size(-1) - 1, device: buffer.device);
                buffer.index_copy_(-1, indices.add(1), buffer.index(TensorIndex.Ellipsis, indices));

                // Add current input with feedback
                buffer.index_put_(currentInput + (delayedSignal * feedback), TensorIndex.Ellipsis, 0);

                // Store output
                processed.index_put_(delayedSignal, TensorIndex.Ellipsis, n);
            }

            // Mix dry and wet signals
            return ((1 - wetLevel) * input) + (wetLevel * processed);
        }

        /// <summary>
        /// Applies a high-pass filter to an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="sampleRate">Sample rate of the audio.</param>
        /// <param name="cutoffFreq">Cutoff frequency in Hz.</param>
        /// <param name="filterOrder">Filter order (number of zeros).</param>
        /// <returns>Processed audio tensor.</returns>
        public static Tensor ApplyHighPassFilter(
            Tensor audio,
            int sampleRate,
            float cutoffFreq = 1000.0f,
            int filterOrder = 51)
        {
            // Ensure filter order is odd
            filterOrder = filterOrder % 2 == 0 ? filterOrder + 1 : filterOrder;

            // Normalize cutoff frequency
            var normCutoff = cutoffFreq / sampleRate;

            // Create filter
            var n = arange(-(filterOrder / 2), (filterOrder / 2) + 1, device: audio.device);

            // Sinc function
            var sinc = where(
                n == 0,
                ones(1, device: audio.device),
                sin(2 * (float)Math.PI * normCutoff * n) / (n * (float)Math.PI));

            // High-pass filter design
            var h = -sinc;
            h[filterOrder / 2] += 1.0f;  // Add impulse at center

            // Window the filter (Hamming)
            var window = 0.54f - (0.46f * cos(2 * (float)Math.PI * (n + (filterOrder / 2)) / filterOrder));
            h *= window;

            // Normalize
            h /= abs(h).sum();

            // Reshape for convolution
            h = h.reshape(1, 1, -1).to(audio.device);

            // Apply filter to each channel
            var filtered = zeros_like(audio);

            for (int b = 0; b < audio.size(0); b++)
            {
                for (int c = 0; c < audio.size(1); c++)
                {
                    var channel = audio[b, c].unsqueeze(0).unsqueeze(0);

                    // Apply convolution
                    var filteredChannel = nn.functional.conv1d(
                        channel,
                        h,
                        padding: filterOrder / 2
                    );

                    filtered[b, c] = filteredChannel.squeeze();
                }
            }

            return filtered;
        }

        /// <summary>
        /// Applies a low-pass filter to an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="sampleRate">Sample rate of the audio.</param>
        /// <param name="cutoffFreq">Cutoff frequency in Hz.</param>
        /// <param name="filterOrder">Filter order (number of zeros).</param>
        /// <returns>Processed audio tensor.</returns>
        public static Tensor ApplyLowPassFilter(
            Tensor audio,
            int sampleRate,
            float cutoffFreq = 1000.0f,
            int filterOrder = 51)
        {
            // Ensure filter order is odd
            filterOrder = filterOrder % 2 == 0 ? filterOrder + 1 : filterOrder;

            // Normalize cutoff frequency
            var normCutoff = cutoffFreq / sampleRate;

            // Create filter
            var n = arange(-(filterOrder / 2), (filterOrder / 2) + 1, device: audio.device);

            // Sinc function for low-pass filter
            var h = where(
                n == 0,
                tensor(2 * normCutoff, device: audio.device),
                sin(2 * (float)Math.PI * normCutoff * n) / (n * (float)Math.PI));

            // Window the filter (Hamming)
            var window = 0.54f - (0.46f * cos(2 * (float)Math.PI * (n + (filterOrder / 2)) / filterOrder));
            h *= window;

            // Normalize
            h /= h.sum();

            // Reshape for convolution
            h = h.reshape(1, 1, -1).to(audio.device);

            // Apply filter to each channel
            var filtered = zeros_like(audio);

            for (int b = 0; b < audio.size(0); b++)
            {
                for (int c = 0; c < audio.size(1); c++)
                {
                    var channel = audio[b, c].unsqueeze(0).unsqueeze(0);

                    // Apply convolution
                    var filteredChannel = nn.functional.conv1d(
                        channel,
                        h,
                        padding: filterOrder / 2
                    );

                    filtered[b, c] = filteredChannel.squeeze();
                }
            }

            return filtered;
        }

        /// <summary>
        /// Applies a reverb effect to an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="roomSize">Room size coefficient [0-1].</param>
        /// <param name="damping">Damping coefficient [0-1].</param>
        /// <param name="wetLevel">Wet level [0-1].</param>
        /// <param name="dryLevel">Dry level [0-1].</param>
        /// <returns>Processed audio tensor.</returns>
        public static Tensor ApplyReverb(
            Tensor audio,
            float roomSize = 0.8f,
            float damping = 0.5f,
            float wetLevel = 0.3f,
            float dryLevel = 0.7f)
        {
            roomSize = Math.Clamp(roomSize, 0f, 1f);
            damping = Math.Clamp(damping, 0f, 1f);
            wetLevel = Math.Clamp(wetLevel, 0f, 1f);
            dryLevel = Math.Clamp(dryLevel, 0f, 1f);

            var input = audio;

            // Comb filter delays (Schroeder reverberator)
            var combDelays = new[] { 1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116 };

            // Allpass filter delays
            var allpassDelays = new[] { 225, 556, 441, 341 };

            // Apply comb filters
            var combFiltered = zeros_like(input);

            foreach (var delay in combDelays)
            {
                var buffer = zeros(input.size(0), input.size(1), delay, device: input.device);
                var lastOutput = zeros_like(input[TensorIndex.Ellipsis, 0]);
                var tempBuffer = zeros_like(input);
                var feedback = roomSize * 0.84f;

                for (int n = 0; n < input.size(-1); n++)
                {
                    var currentInput = input.index(TensorIndex.Ellipsis, n);
                    var delayed = buffer.index(TensorIndex.Ellipsis, -1);

                    // Low-pass filter the feedback path
                    var filterOutput = delayed.mul(1 - damping).add(lastOutput.mul(damping));

                    // Roll buffer
                    using var indices = arange(0, buffer.size(-1) - 1, device: buffer.device);
                    buffer.index_copy_(-1, indices.add(1), buffer.index(TensorIndex.Ellipsis, indices));

                    // Add current input with feedback
                    buffer.index_put_(currentInput.add(filterOutput.mul(feedback)), TensorIndex.Ellipsis, 0);

                    // Store output
                    tempBuffer.index_put_(filterOutput, TensorIndex.Ellipsis, n);
                    lastOutput.copy_(filterOutput);
                }

                combFiltered.add_(tempBuffer);
            }

            // Apply allpass filters
            var allpassFiltered = combFiltered;

            foreach (var delay in allpassDelays)
            {
                var tempBuffer = zeros_like(allpassFiltered);
                var buffer = zeros(input.size(0), input.size(1), delay, device: input.device);
                var feedback = 0.5f;

                for (int n = 0; n < allpassFiltered.size(-1); n++)
                {
                    var currentInput = allpassFiltered.index(TensorIndex.Ellipsis, n);
                    var delayed = buffer.index(TensorIndex.Ellipsis, -1);

                    // Allpass structure
                    var filterOutput = currentInput.mul(-feedback).add(delayed).add(delayed.mul(feedback));

                    // Roll buffer
                    using var indices = arange(0, buffer.size(-1) - 1, device: buffer.device);
                    buffer.index_copy_(-1, indices.add(1), buffer.index(TensorIndex.Ellipsis, indices));

                    // Add to buffer
                    buffer.index_put_(currentInput.add(filterOutput.mul(feedback)), TensorIndex.Ellipsis, 0);

                    // Store output
                    tempBuffer.index_put_(filterOutput, TensorIndex.Ellipsis, n);
                }

                allpassFiltered = tempBuffer;
            }

            // Mix dry and wet signals
            return input.mul(dryLevel).add(allpassFiltered.mul(wetLevel));
        }

        /// <summary>
        /// Applies a tremolo effect to an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="sampleRate">Sample rate of the audio.</param>
        /// <param name="rate">Modulation rate in Hz.</param>
        /// <param name="depth">Modulation depth [0-1].</param>
        /// <returns>Processed audio tensor.</returns>
        public static Tensor ApplyTremolo(
            Tensor audio,
            int sampleRate,
            float rate = 5.0f,
            float depth = 0.5f)
        {
            var input = audio;
            var samples = input.size(-1);

            // Create time vector
            var time = linspace(0, (float)samples / sampleRate, samples, device: audio.device);

            // Generate LFO
            var lfo = 1 - depth + (depth * sin(2 * (float)Math.PI * rate * time));

            // Apply modulation (expand dimensions to match input)
            var modulation = lfo.unsqueeze(0).unsqueeze(0).expand_as(input);

            return input * modulation;
        }
    }
}