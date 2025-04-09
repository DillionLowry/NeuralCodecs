using NeuralCodecs.Torch.Utils;
using System.Numerics;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Provides a collection of audio processing effects for tensor and AudioSignal manipulation.
/// Implements common digital signal processing algorithms including reverb, delay, modulation,
/// and dynamic processing effects.
/// </summary>
public static partial class AudioEffects
{
    /// <summary>
    /// Applies a chorus effect using multiple modulated delay lines.
    /// </summary>
    /// <param name="signal">The input audio signal to process.</param>
    /// <param name="voices">Number of chorus voices to generate.</param>
    /// <param name="baseDelay">Base delay time in seconds.</param>
    /// <param name="rateSpread">Variation in modulation rate between voices.</param>
    /// <param name="depthSpread">Variation in modulation depth between voices.</param>
    /// <param name="wetLevel">Mix level between dry and processed signal. Range: 0.0 to 1.0.</param>
    /// <returns>A new AudioSignal instance containing the processed audio.</returns>
    public static AudioSignal ApplyChorus(
        this AudioSignal signal,
        int voices = 3,
        float baseDelay = 0.030f,
        float rateSpread = 0.2f,
        float depthSpread = 0.003f,
        float wetLevel = 0.5f)
    {
        var output = signal.Clone();
        var input = signal.AudioData;

        var processed = zeros_like(input);

        // Generate multiple modulated delay lines
        for (int voice = 0; voice < voices; voice++)
        {
            // Unique parameters for each voice
            var rate = 0.5f + (rateSpread * (voice - (voices / 2)) / voices);
            var depth = 0.002f + (depthSpread * voice / voices);
            var delay = baseDelay + (0.005f * voice);

            // Generate LFO
            var t = linspace(0, signal.SignalDuration, signal.SignalLength, device: signal.Device);
            var maxDelaySamples = (int)((delay + depth) * signal.SampleRate);
            var lfo = maxDelaySamples * 0.5f * (1 + sin((2 * (float)Math.PI * rate * t) + (voice * (float)Math.PI / voices)));

            var voiceBuffer = zeros(input.size(0), input.size(1), maxDelaySamples + 1).to(input.device);
            var voiceOutput = zeros_like(input);

            // Process voice
            for (int n = 0; n < input.size(-1); n++)
            {
                var currentInput = input[.., n];

                var delayLength = lfo[n];
                var delayFloor = floor(delayLength).to(int64);
                var delayFrac = delayLength - delayFloor;

                var delayed1 = voiceBuffer.index(TensorIndex.Ellipsis, delayFloor);

                var delayed2 = voiceBuffer.index(TensorIndex.Ellipsis, delayFloor + 1);
                var delayedSignal = (delayed1 * (1 - delayFrac)) + (delayed2 * delayFrac);

                voiceBuffer.RollInPlace(-1, -1);
                voiceBuffer[.., 0] = currentInput;

                voiceOutput[.., n] = delayedSignal;
            }

            processed += voiceOutput;
        }

        // Mix and normalize
        processed /= voices;
        output.AudioData = ((1 - wetLevel) * input) + (wetLevel * processed);
        return output;
    }

    public static AudioSignal ApplyChorus_ToTest(
        this AudioSignal signal,
        int voices = 3,
        float baseDelay = 0.030f,
        float rateSpread = 0.2f,
        float depthSpread = 0.003f,
        float wetLevel = 0.5f)
    {
        var output = signal.Clone();
        var input = signal.AudioData;

        // Preallocate tensors outside the loop
        var processed = zeros_like(input);
        var t = linspace(0, signal.SignalDuration, signal.SignalLength, device: signal.Device);

        // Calculate max buffer size once
        var maxDelayBase = baseDelay + (0.005f * (voices - 1));
        var maxDepth = 0.002f + (depthSpread * (voices - 1) / voices);
        var maxDelaySamples = (int)((maxDelayBase + maxDepth) * signal.SampleRate);

        // Reuse these tensors across voices
        var voiceBuffer = zeros(input.size(0), input.size(1), maxDelaySamples + 1).to(input.device);
        var voiceOutput = zeros_like(input);
        var delayLengths = zeros(signal.SignalLength, device: signal.Device);

        try
        {
            for (int voice = 0; voice < voices; voice++)
            {
                var rate = 0.5f + (rateSpread * (voice - (voices / 2)) / voices);
                var depth = 0.002f + (depthSpread * voice / voices);
                var delay = baseDelay + (0.005f * voice);
                var currentMaxDelay = (int)((delay + depth) * signal.SampleRate);

                // Calculate LFO in-place
                delayLengths.copy_(currentMaxDelay * 0.5f *
                    (1 + sin((2 * (float)Math.PI * rate * t) + (voice * (float)Math.PI / voices))));

                // Reset buffers
                voiceBuffer.zero_();
                voiceOutput.zero_();

                for (int n = 0; n < input.size(-1); n++)
                {
                    var currentInput = input[.., n];
                    var delayLength = delayLengths[n];
                    var delayFloor = floor(delayLength).to(int64);
                    var delayFrac = delayLength - delayFloor;

                    // Use indexing without creating new tensors
                    using (var delayed1 = voiceBuffer.index(TensorIndex.Ellipsis, delayFloor))
                    using (var delayed2 = voiceBuffer.index(TensorIndex.Ellipsis, delayFloor + 1))
                    {
                        voiceOutput[.., n] = (delayed1 * (1 - delayFrac)) + (delayed2 * delayFrac);
                    }

                    voiceBuffer.RollInPlace(-1, -1);
                    voiceBuffer[.., 0] = currentInput;
                }

                processed.add_(voiceOutput, alpha: 1.0f / voices);  // In-place addition and scaling
            }

            // Final mix in-place
            output.AudioData = input.mul(1 - wetLevel).add(processed.mul(wetLevel));
            return output;
        }
        finally
        {
            // Explicit cleanup
            voiceBuffer.Dispose();
            voiceOutput.Dispose();
            delayLengths.Dispose();
            processed.Dispose();
            t.Dispose();
        }
    }

    /// <summary>
    /// Applies dynamic range compression with attack, release, and makeup gain controls.
    /// </summary>
    /// <param name="signal">The input audio signal to process.</param>
    /// <param name="threshold">Level above which compression begins, in dB.</param>
    /// <param name="ratio">Amount of compression applied above threshold.</param>
    /// <param name="attackTime">Time taken for the compressor to respond to increases in level, in seconds.</param>
    /// <param name="releaseTime">Time taken for the compressor to respond to decreases in level, in seconds.</param>
    /// <param name="makeupGain">Additional gain applied after compression, in dB.</param>
    /// <returns>A new AudioSignal instance containing the processed audio.</returns>
    public static AudioSignal ApplyCompressor(
        this AudioSignal signal,
        float threshold = -20.0f,
        float ratio = 4.0f,
        float attackTime = 0.005f,
        float releaseTime = 0.050f,
        float makeupGain = 0.0f)
    {
        var output = signal.Clone();
        var input = signal.AudioData;

        // Convert threshold to linear scale
        var thresholdLin = exp(threshold / 20.0f * (float)Math.Log(10));

        // Calculate envelope
        var attackSamples = (int)(attackTime * signal.SampleRate);
        var releaseSamples = (int)(releaseTime * signal.SampleRate);

        var envelope = zeros_like(input);
        var currentLevel = zeros(input.size(0), input.size(1), device: input.device);

        // Envelope follower
        for (int n = 0; n < input.size(-1); n++)
        {
            var inputLevel = input[.., n].abs();
            var attackGain = 1.0f - exp(-1.0f / attackSamples);
            var releaseGain = 1.0f - exp(-1.0f / releaseSamples);

            var gain = (bool)(inputLevel > currentLevel) ? attackGain : releaseGain;
            currentLevel.add_(gain * (inputLevel - currentLevel));
            envelope[.., n] = currentLevel;
        }

        // Apply compression
        var gainReduction = ones_like(envelope);
        var mask = envelope > thresholdLin;
        gainReduction[mask] = pow(envelope[mask] / thresholdLin, (1.0f / ratio) - 1.0f);

        // Apply makeup gain
        var makeupGainLin = exp(makeupGain / 20.0f * (float)Math.Log(10));
        output.AudioData = input * gainReduction * makeupGainLin;

        return output;
    }

    /// <summary>
    /// Applies a delay effect with feedback for creating echo and repeat effects.
    /// </summary>
    /// <param name="signal">The input audio signal to process.</param>
    /// <param name="delayTime">Time in seconds between the original signal and its echo.</param>
    /// <param name="feedback">Amount of delayed signal fed back into the delay line. Range: 0.0 to 1.0.</param>
    /// <param name="wetLevel">Level of the delayed signal. Range: 0.0 to 1.0.</param>
    /// <param name="dryLevel">Level of the original signal. Range: 0.0 to 1.0.</param>
    /// <returns>A new AudioSignal instance containing the processed audio.</returns>
    public static AudioSignal ApplyDelay(
        this AudioSignal signal,
        float delayTime = 0.3f,
        float feedback = 0.3f,
        float wetLevel = 0.3f,
        float dryLevel = 0.7f)
    {
        var output = signal.Clone();
        var input = signal.AudioData;

        int delaySamples = (int)(delayTime * signal.SampleRate);
        var buffer = zeros(input.size(0), input.size(1), delaySamples).to(input.device);

        var delayed = zeros_like(input);

        for (int n = 0; n < input.size(-1); n++)
        {
            var current_input = input[.., n];
            var delayed_signal = buffer[.., -1];
            delayed[.., n] = delayed_signal;

            buffer.RollInPlace(-1, -1);
            buffer[.., 0] = current_input + (delayed_signal * feedback);
        }

        output.AudioData = (dryLevel * input) + (wetLevel * delayed);
        return output;
    }

    /// <summary>
    /// Applies waveshaping distortion using a hyperbolic tangent transfer function.
    /// </summary>
    /// <param name="signal">The input audio signal to process.</param>
    /// <param name="amount">Amount of distortion to apply. Range: 0.0 (clean) to 1.0 (heavily distorted).</param>
    /// <param name="wetLevel">Mix level between dry and processed signal. Range: 0.0 to 1.0.</param>
    /// <returns>A new AudioSignal instance containing the processed audio.</returns>
    public static AudioSignal ApplyDistortion(
        this AudioSignal signal,
        float amount = 0.5f,
        float wetLevel = 1.0f)
    {
        var output = signal.Clone();
        var input = signal.AudioData;

        // Apply waveshaping distortion
        var processed = tanh(input * (1 + (amount * 10)));

        output.AudioData = (processed * wetLevel) + (input * (1 - wetLevel));
        return output;
    }

    /// <summary>
    /// Applies a flanger effect using a modulated delay line with feedback.
    /// </summary>
    /// <param name="signal">The input audio signal to process.</param>
    /// <param name="rate">Speed of the modulation in Hz.</param>
    /// <param name="depth">Maximum delay time variation in seconds.</param>
    /// <param name="feedback">Amount of processed signal fed back into the effect. Range: 0.0 to 1.0.</param>
    /// <param name="wetLevel">Mix level between dry and processed signal. Range: 0.0 to 1.0.</param>
    /// <returns>A new AudioSignal instance containing the processed audio.</returns>
    public static AudioSignal ApplyFlanger(
        this AudioSignal signal,
        float rate = 0.5f,
        float depth = 0.002f,
        float feedback = 0.7f,
        float wetLevel = 0.7f)
    {
        var output = signal.Clone();
        var input = signal.AudioData;

        // Generate LFO for delay modulation
        var t = linspace(0, signal.SignalDuration, signal.SignalLength, device: signal.Device);
        var maxDelaySamples = (int)(depth * signal.SampleRate);
        var lfo = maxDelaySamples * 0.5f * (1 + sin(2 * (float)Math.PI * rate * t));

        var processed = zeros_like(input);
        var buffer = zeros(input.size(0), input.size(1), maxDelaySamples + 1).to(input.device);

        // Process samples
        for (int n = 0; n < input.size(-1); n++)
        {
            var currentInput = input[.., n];

            // Calculate fractional delay
            var delayLength = lfo[n];
            var delayFloor = floor(delayLength).to(int64);
            var delayFrac = delayLength - delayFloor;

            // Linear interpolation for fractional delay
            var delayed1 = buffer.index(TensorIndex.Ellipsis, delayFloor);
            var delayed2 = buffer.index(TensorIndex.Ellipsis, delayFloor + 1);
            var delayedSignal = (delayed1 * (1 - delayFrac)) + (delayed2 * delayFrac);

            // Update buffer
            buffer.RollInPlace(-1, -1);
            buffer[.., 0] = currentInput + (delayedSignal * feedback);

            processed[.., n] = delayedSignal;
        }

        output.AudioData = ((1 - wetLevel) * input) + (wetLevel * processed);
        return output;
    }

    /// <summary>
    /// Applies a phaser effect using multiple all-pass filters with modulated center frequencies.
    /// </summary>
    /// <param name="signal">The input audio signal to process.</param>
    /// <param name="rate">Speed of the modulation in Hz.</param>
    /// <param name="stages">Number of all-pass filter stages.</param>
    /// <param name="depth">Intensity of the frequency modulation. Range: 0.0 to 1.0.</param>
    /// <param name="feedback">Amount of processed signal fed back into the effect. Range: 0.0 to 1.0.</param>
    /// <param name="wetLevel">Mix level between dry and processed signal. Range: 0.0 to 1.0.</param>
    /// <returns>A new AudioSignal instance containing the processed audio.</returns>
    public static AudioSignal ApplyPhaser(
        this AudioSignal signal,
        float rate = 0.5f,
        int stages = 6,
        float depth = 0.8f,
        float feedback = 0.7f,
        float wetLevel = 0.7f)
    {
        var output = signal.Clone();
        var input = signal.AudioData;

        // All-pass filter coefficients
        var minFreq = 200.0f;
        var maxFreq = 1600.0f;

        // Generate LFO
        var t = linspace(0, signal.SignalDuration, signal.SignalLength, device: signal.Device);
        var lfo = depth * sin(2 * (float)Math.PI * rate * t);

        var processed = input.clone();
        var lastOutput = zeros_like(input[.., 0]);

        for (int stage = 0; stage < stages; stage++)
        {
            var freq = exp(log(tensor(maxFreq / minFreq)) * ((stage + 1.0f) / stages)) * minFreq;
            var processedStage = zeros_like(input);

            for (int n = 0; n < input.size(-1); n++)
            {
                var freqMod = freq * (1 + lfo[n]);
                var alpha = (tensor((float)Math.PI) * freqMod / signal.SampleRate).tan();
                var coefficient = (alpha - 1) / (alpha + 1);

                var currentInput = processed[.., n] + (lastOutput * feedback);
                processedStage[.., n] = (coefficient * currentInput) + lastOutput;
                lastOutput = processedStage[.., n];
            }

            processed = processedStage;
        }

        output.AudioData = ((1 - wetLevel) * input) + (wetLevel * processed);
        return output;
    }

    /// <summary>
    /// Applies pitch shifting using STFT (Short-Time Fourier Transform) processing.
    /// </summary>
    /// <param name="signal">The input audio signal to process.</param>
    /// <param name="semitones">Number of semitones to shift the pitch (positive or negative).</param>
    /// <param name="windowSize">Size of the STFT analysis window in samples.</param>
    /// <param name="wetLevel">Mix level between dry and processed signal. Range: 0.0 to 1.0.</param>
    /// <returns>A new AudioSignal instance containing the processed audio.</returns>
    public static AudioSignal ApplyPitchShift(
        this AudioSignal signal,
        float semitones = 12.0f,
        int windowSize = 2048,
        float wetLevel = 1.0f)
    {
        var output = signal.Clone();

        // Store original length for later
        var originalLength = signal.SignalLength;

        // Compute STFT
        output.STFT(windowSize);

        // Calculate frequency shift factor
        var shiftFactor = (float)Math.Pow(2, semitones / 12.0);

        // Shift frequencies
        var magnitudes = output.Magnitude;
        var phases = output.Phase;

        var freqBins = arange(magnitudes.size(2), device: signal.Device);
        var shiftedBins = (freqBins * shiftFactor).floor().to(int64);
        shiftedBins = shiftedBins.clamp(0, magnitudes.size(2) - 1);

        var newMagnitudes = zeros_like(magnitudes);
        var newPhases = zeros_like(phases);

        // Remap frequency bins
        for (int i = 0; i < magnitudes.size(2); i++)
        {
            var targetBin = shiftedBins[i];

            //newMagnitudes[.., targetBin, ..] += magnitudes[.., i, ..];
            newMagnitudes.index_add_(
                dim: 2,
                index: targetBin,
                source: magnitudes.index(TensorIndex.Ellipsis, i, TensorIndex.Ellipsis),
                alpha: 1
            );

            //newPhases[.., targetBin, ..] += phases[.., i, ..];
            newPhases.index_add_(
                dim: 2,
                index: targetBin,
                source: phases.index(TensorIndex.Ellipsis, i, TensorIndex.Ellipsis),
                alpha: 1
            );
        }

        // Reconstruct complex STFT
        output.StftData = newMagnitudes * exp(Complex.ImaginaryOne * newPhases);

        // Inverse STFT
        output.InverseSTFT(windowSize, length: originalLength);

        // Mix dry/wet
        if (wetLevel < 1.0f)
        {
            output.AudioData = ((1 - wetLevel) * signal.AudioData) + (wetLevel * output.AudioData);
        }

        return output;
    }

    /// <summary>
    /// Applies a Schroeder reverb effect to simulate room acoustics using parallel comb filters and series allpass filters.
    /// </summary>
    /// <param name="signal">The input audio signal to process.</param>
    /// <param name="roomSize">Controls the decay time of the reverb. Range: 0.0 (small room) to 1.0 (large hall).</param>
    /// <param name="damping">Controls high frequency absorption. Range: 0.0 (bright) to 1.0 (dark).</param>
    /// <param name="wetLevel">Level of the processed reverb signal. Range: 0.0 to 1.0.</param>
    /// <param name="dryLevel">Level of the original unprocessed signal. Range: 0.0 to 1.0.</param>
    /// <returns>A new AudioSignal instance containing the processed audio.</returns>
    public static AudioSignal ApplyReverb(
        this AudioSignal signal,
        float roomSize = 0.8f,
        float damping = 0.5f,
        float wetLevel = 0.3f,
        float dryLevel = 0.7f)
    {
        ArgumentNullException.ThrowIfNull(signal);

        // Clamp parameters to valid ranges
        roomSize = Math.Clamp(roomSize, 0f, 1f);
        damping = Math.Clamp(damping, 0f, 1f);
        wetLevel = Math.Clamp(wetLevel, 0f, 1f);
        dryLevel = Math.Clamp(dryLevel, 0f, 1f);

        // Schroeder reverberator implementation
        var output = signal.Clone();
        var input = signal.AudioData;

        // Create comb filters
        var combDelays = new[] { 1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116 };
        var allpassDelays = new[] { 225, 556, 441, 341 };

        // Preallocate reusable tensors
        var combFiltered = zeros_like(input);
        var tempBuffer = zeros_like(input);
        var lastOutput = zeros_like(input[.., 0]);

        try
        {
            // Apply parallel comb filters
            foreach (var delay in combDelays)
            {
                using var buffer = zeros(input.size(0), input.size(1), delay).to(input.device);
                var feedback = roomSize * 0.84f;

                tempBuffer.zero_(); // Reuse buffer by clearing
                lastOutput.zero_();

                for (int n = 0; n < input.size(-1); n++)
                {
                    var currentInput = input[.., n];
                    using var delayed = buffer[.., -1];
                    var filterOutput = delayed.mul(1 - damping).add_(lastOutput.mul(damping));

                    buffer.RollInPlace(-1, -1);
                    buffer[.., 0] = currentInput.add(filterOutput.mul(feedback));
                    tempBuffer[.., n] = filterOutput;
                    lastOutput.copy_(filterOutput);
                }

                combFiltered.add_(tempBuffer);
            }

            // Apply series allpass filters
            var allpassFiltered = combFiltered; // reuse tensor

            foreach (var delay in allpassDelays)
            {
                using var buffer = zeros(input.size(0), input.size(1), delay).to(input.device);
                tempBuffer.zero_();
                var feedback = 0.5f;

                for (int n = 0; n < input.size(-1); n++)
                {
                    var currentInput = allpassFiltered[.., n];
                    using var delayed = buffer[.., -1];
                    var filterOutput = currentInput.mul(-feedback).add(delayed).add(delayed.mul(feedback));

                    buffer.RollInPlace(-1, -1);
                    buffer[.., 0] = currentInput.add(filterOutput.mul(feedback));
                    tempBuffer[.., n] = filterOutput;
                }

                allpassFiltered = tempBuffer.clone();
            }

            // Final mix in-place
            output.AudioData = input.mul(dryLevel).add_(allpassFiltered.mul(wetLevel));
            return output;
        }
        finally
        {
            // Cleanup
            combFiltered.Dispose();
            tempBuffer.Dispose();
            lastOutput.Dispose();
        }
    }

    /// <summary>
    /// Applies amplitude modulation (tremolo) using a sine wave LFO.
    /// </summary>
    /// <param name="signal">The input audio signal to process.</param>
    /// <param name="rate">Frequency of the modulation in Hz.</param>
    /// <param name="depth">Intensity of the modulation effect. Range: 0.0 to 1.0.</param>
    /// <returns>A new AudioSignal instance containing the processed audio.</returns>
    public static AudioSignal ApplyTremolo(
           this AudioSignal signal,
           float rate = 5.0f,
           float depth = 0.5f)
    {
        var output = signal.Clone();
        var input = signal.AudioData;

        // Generate LFO
        var t = linspace(0, signal.SignalDuration, signal.SignalLength, device: signal.Device);
        var lfo = 1 - depth + (depth * sin(2 * (float)Math.PI * rate * t));

        // Apply amplitude modulation
        output.AudioData = input * lfo.unsqueeze(0).unsqueeze(0).expand_as(input);
        return output;
    }
}