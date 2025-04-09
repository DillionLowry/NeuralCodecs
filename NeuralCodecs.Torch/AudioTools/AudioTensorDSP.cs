using NeuralCodecs.Core.Utils;
using NeuralCodecs.Torch.Utils;
using System.Diagnostics;
using TorchSharp;
using static Tensorboard.Summary.Types;
using static TorchSharp.torch;
using Complex = System.Numerics.Complex;

namespace NeuralCodecs.Torch.AudioTools
{
    /// <summary>
    /// Provides digital signal processing operations for torch tensors.
    /// </summary>
    public partial class DSP
    {
        /// <summary>
        /// Collects overlapping windows from an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="windowLength">Length of each window in samples.</param>
        /// <param name="hopLength">Hop length between windows in samples.</param>
        /// <returns>Windowed audio tensor of shape (batch_size*num_windows, channels, window_length).</returns>
        public static Tensor CollectWindows(
            Tensor audio,
            int windowLength,
            int hopLength)
        {
            // Ensure 3D input
            if (audio.dim() < 3)
            {
                if (audio.dim() == 1)
                {
                    audio = audio.unsqueeze(0).unsqueeze(0);
                }
                else if (audio.dim() == 2)
                {
                    audio = audio.unsqueeze(1);
                }
            }

            var batchSize = audio.size(0);
            var channels = audio.size(1);
            var samples = audio.size(-1);

            // Calculate number of windows
            var numWindows = ((samples - windowLength) / hopLength) + 1;

            // Use unfold to create windows
            var unfolded = nn.functional.unfold(
                audio.reshape(batchSize * channels, 1, 1, samples),
                kernel_size: (1, windowLength),
                stride: (1, hopLength));

            // Reshape to (batch_size*num_windows, channels, window_length)
            unfolded = unfolded.permute(0, 2, 1).reshape(batchSize, channels, numWindows, windowLength);
            unfolded = unfolded.permute(0, 2, 1, 3).reshape(batchSize * numWindows, channels, windowLength);

            return unfolded;
        }

        /// <summary>
        /// Converts stereo audio to mono by averaging channels.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <returns>Mono audio tensor of shape (batch_size, 1, samples).</returns>
        public static Tensor ConvertToMono(Tensor audio)
        {
            // Check tensor dimensions and shape
            if (audio.dim() == 2)
            {
                // Case 1: [channels, time] format
                if (audio.size(0) == 2)
                {
                    return audio.mean([0], keepdim: true);
                }
                // Case 2: [time, channels] format
                else if (audio.size(1) == 2)
                {
                    return audio.mean([1], keepdim: true);
                }
            }
            // Case 3: [batch, channels, time] format
            else if (audio.dim() == 3 && audio.size(1) == 2)
            {
                return audio.mean([1], keepdim: true);
            }

            // Already mono or unrecognized format
            return audio;
        }

        /// <summary>
        /// Converts deinterleaved stereo audio (LL...RR...) to interleaved (LRLR...).
        /// </summary>
        /// <param name="deinterleavedAudio">Deinterleaved audio tensor of shape (batch_size, 2, samples).</param>
        /// <returns>Interleaved audio tensor of shape (batch_size, 2, samples).</returns>
        public static Tensor DeinterleaveToInterleave(Tensor deinterleavedAudio)
        {
            if (deinterleavedAudio.size(1) != 2)
            {
                throw new ArgumentException("Input must be stereo (2 channels)");
            }

            var batchSize = deinterleavedAudio.size(0);
            var samplesPerChannel = deinterleavedAudio.size(2) / 2;
            var result = zeros_like(deinterleavedAudio);

            for (int b = 0; b < batchSize; b++)
            {
                // Extract left and right channels
                var leftRange = arange(0, samplesPerChannel);
                var left = deinterleavedAudio[b, 0].index(leftRange);
                var right = deinterleavedAudio[b, 1].index(leftRange);

                // Store in interleaved format (alternating L and R samples)
                for (int i = 0; i < samplesPerChannel; i++)
                {
                    result[b, 0, i * 2] = left[i];
                    result[b, 0, (i * 2) + 1] = right[i];
                }
            }

            return result;
        }

        /// <summary>
        /// Gets an appropriate window function for STFT operations.
        /// </summary>
        /// <param name="windowType">Type of window (e.g., "hann", "hamming", "blackman")</param>
        /// <param name="windowLength">Length of the window in samples</param>
        /// <param name="device">Optional device to place the window tensor on</param>
        /// <returns>Window tensor</returns>
        public static Tensor GetWindow(string windowType, int windowLength, Device? device = null)
        {
            // Set default device to CPU if not provided
            device ??= CPU;

            Tensor window = windowType.ToLowerInvariant() switch
            {
                "hann" => hann_window(windowLength),
                "hamming" => hamming_window(windowLength),
                "blackman" => blackman_window(windowLength),
                "bartlett" => bartlett_window(windowLength),
                "sqrt_hann" => sqrt(hann_window(windowLength)),
                "average" => torch.ones(windowLength) / windowLength,
                "ones" => torch.ones(windowLength),
                _ => throw new ArgumentException($"Unsupported window type: {windowType}", nameof(windowType)),
            };

            return window.to(device).@float();
        }

        /// <summary>
        /// Converts interleaved stereo audio (LRLR...) to deinterleaved (LL...RR...).
        /// </summary>
        /// <param name="interleavedAudio">Interleaved audio tensor of shape (batch_size, 2, samples).</param>
        /// <returns>Deinterleaved audio tensor of shape (batch_size, 2, samples).</returns>
        public static Tensor InterleaveToDeinterleave(Tensor interleavedAudio)
        {

            return torch.tensor(AudioUtils.InterleaveToDeinterleave2d(interleavedAudio.cpu().detach().data<float>().ToArray()));
            //long batchSize = interleavedAudio.shape[0];
            //long samplesPerBatch = interleavedAudio.shape[2];

            //if (samplesPerBatch % 2 != 0)
            //    throw new ArgumentException("Each row must have an even number of samples", nameof(interleavedAudio));

            //long samplesPerChannel = samplesPerBatch / 2;
            //long totalSamplesPerChannel = batchSize * samplesPerChannel;

            //// Create the result tensor
            //var result = torch.zeros(new long[] { 2, totalSamplesPerChannel },
            //    dtype: interleavedAudio.dtype,
            //    device: interleavedAudio.device);

            //// Process each batch
            //for (long b = 0; b < batchSize; b++)
            //{
            //    // Get the current batch of interleaved audio
            //    var batch = interleavedAudio[b];

            //    // Even indices for left channel
            //    var evenIndices = torch.arange(0, samplesPerBatch, 2, dtype: torch.int64, device: interleavedAudio.device);
            //    var leftChannel = torch.index_select(batch, 0, evenIndices);

            //    // Odd indices for right channel
            //    var oddIndices = torch.arange(1, samplesPerBatch, 2, dtype: torch.int64, device: interleavedAudio.device);
            //    var rightChannel = torch.index_select(batch, 0, oddIndices);

            //    // Place in the result tensor
            //    var resultOffset = b * samplesPerChannel;
            //    result[0].narrow(0, resultOffset, samplesPerChannel).copy_(leftChannel);
            //    result[1].narrow(0, resultOffset, samplesPerChannel).copy_(rightChannel);
            //}

            //return result;
        }

        /// <summary>
        /// Computes the Inverse Short-Time Fourier Transform (ISTFT) of a complex spectrogram.
        /// </summary>
        /// <param name="stft">Complex STFT tensor of shape (batch_size, channels, freq_bins, time_frames).</param>
        /// <param name="params">STFT parameters.</param>
        /// <param name="length">Optional target length of the output signal.</param>
        /// <returns>Audio tensor of shape (batch_size, channels, samples).</returns>
        public static Tensor InverseSTFT(Tensor stft, STFTParams? @params = null, int? length = null)
        {
            @params ??= new STFTParams();

            var batchSize = stft.size(0);
            var channels = stft.size(1);
            var freqBins = stft.size(2);
            var timeFrames = stft.size(3);

            var window = GetWindow(@params.WindowType, @params.WindowLength, stft.device);

            // Reshape to [batch*channels, freq_bins, time_frames]
            var stftFlat = stft.reshape(batchSize * channels, freqBins, timeFrames);

            // Compute ISTFT
            var audioFlat = istft(
                stftFlat,
                n_fft: @params.WindowLength,
                hop_length: @params.HopLength,
                window: window,
                center: @params.Center,
                length: length ?? -1
            );

            // Reshape back to [batch_size, channels, samples]
            var samples = audioFlat.size(1);
            return audioFlat.reshape(batchSize, channels, samples);
        }

        /// <summary>
        /// Performs linear overlap-add reconstruction for segments of audio.
        /// Direct port of the PyTorch implementation to ensure compatibility.
        /// </summary>
        /// <param name="frames">List of audio frames to combine</param>
        /// <param name="stride">Stride between consecutive frames</param>
        /// <returns>Reconstructed audio tensor</returns>
        /// <exception cref="ArgumentException">Thrown when frames is empty</exception>
        public static Tensor LinearOverlapAdd(List<Tensor> frames, int stride)
        {
            if (frames.Count == 0)
            {
                throw new ArgumentException("At least one frame is required");
            }

            // Get device and dtype from first frame
            var device = frames[0].device;
            var dtype = frames[0].dtype;

            // Determine the shape excluding the last dimension (time)
            var shape = frames[0].shape.Take((int)(frames[0].dim() - 1)).ToArray();

            // Calculate total output size
            var totalSize = (stride * (frames.Count - 1)) + (int)frames[^1].shape[^1];

            // Create weight tensor for overlap-add (triangular window)
            var frameLength = (int)frames[0].shape[^1];

            // t = linspace(0, 1, frame_length + 2)[1:-1]
            // weight = 0.5 - (t - 0.5).abs()
            var t = linspace(0, 1, frameLength + 2, device: device, dtype: dtype)[1..^1];
            var weight = tensor(0.5f, device: device, dtype: dtype) - (t - tensor(0.5f, device: device, dtype: dtype)).abs();

            // Create output tensor and weight sum tensor
            var sumWeight = zeros(totalSize, device: device, dtype: dtype);
            var outputShape = shape.Concat(new long[] { totalSize }).ToArray();
            var output = zeros(outputShape, device: device, dtype: dtype);

            // Process frames with overlap-add
            int offset = 0;
            foreach (var frame in frames)
            {
                var currentFrameLength = (int)frame.shape[^1];

                // Get appropriate weight slice for current frame length
                var weightSlice = weight.narrow(0, 0, currentFrameLength);

                // Create a shape for broadcasting the weight to match frame dimensions
                var targetShape = new long[frame.dim()];
                for (int i = 0; i < frame.dim() - 1; i++)
                {
                    targetShape[i] = frame.shape[i];
                }
                targetShape[^1] = currentFrameLength;

                // Broadcast weight to match frame dimensions
                var broadcastedWeight = broadcast_to(weightSlice, targetShape);

                try
                {
                    // Apply weight to frame and add to output
                    using (var weightedFrame = frame.mul(broadcastedWeight))
                    {
                        output.narrow(-1, offset, currentFrameLength).add_(weightedFrame);
                    }

                    // Add weight to sum
                    sumWeight.narrow(0, offset, currentFrameLength).add_(weightSlice);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error in LinearOverlapAdd: {ex.Message}");
                    Debug.WriteLine($"Frame shape: {string.Join(", ", frame.shape)}");
                    Debug.WriteLine($"Weight shape: {string.Join(", ", broadcastedWeight.shape)}");
                    Debug.WriteLine($"Output shape: {string.Join(", ", output.shape)}");
                    throw;
                }

                offset += stride;
            }

            // Ensure no division by zero
            if (sumWeight.min().item<float>() <= 1e-10f)
            {
                // Add small epsilon to prevent division by zero
                sumWeight = sumWeight.add(1e-10f);
            }

            // Broadcast sumWeight for division
            var broadcastDims = new List<long>();
            for (int i = 0; i < output.dim() - 1; i++)
            {
                broadcastDims.Add(1);
            }
            broadcastDims.Add(totalSize);

            var reshapedSumWeight = sumWeight.reshape(broadcastDims.ToArray());
            var expandedShape = new List<long>();
            for (int i = 0; i < output.dim() - 1; i++)
            {
                expandedShape.Add(output.shape[i]);
            }
            expandedShape.Add(totalSize);

            var broadcastedSumWeight = reshapedSumWeight.expand(expandedShape.ToArray());

            // Divide output by sum of weights
            return output.div(broadcastedSumWeight);
        }

        /// <summary>
        /// Computes the logarithmic magnitude spectrogram.
        /// </summary>
        /// <param name="stft">Complex STFT tensor.</param>
        /// <param name="refValue">Reference value for logarithmic scaling.</param>
        /// <param name="amin">Minimum amplitude.</param>
        /// <param name="topDb">Maximum decibel value relative to peak.</param>
        /// <returns>Log magnitude spectrogram.</returns>
        public static Tensor LogMagnitude(Tensor stft, float refValue = 1.0f, float amin = 1e-5f, float? topDb = 80.0f)
        {
            var magnitude = Magnitude(stft);
            float aminSq = amin * amin;

            var logSpec = 10.0f * log10(magnitude.pow(2).clamp(min: aminSq));
            logSpec -= 10.0f * (float)Math.Log10(Math.Max(aminSq, refValue));

            if (topDb.HasValue)
            {
                var maxVal = logSpec.max();
                logSpec = maximum(logSpec, maxVal - topDb.Value);
            }

            return logSpec;
        }

        /// <summary>
        /// Computes the magnitude of a complex STFT tensor.
        /// </summary>
        /// <param name="stft">Complex STFT tensor.</param>
        /// <returns>Magnitude tensor.</returns>
        public static Tensor Magnitude(Tensor stft)
        {
            return abs(stft);
        }

        /// <summary>
        /// Masks specific frequency bands in a spectrogram.
        /// </summary>
        /// <param name="spectrogram">Complex STFT tensor.</param>
        /// <param name="sampleRate">Sample rate of the audio.</param>
        /// <param name="fminHz">Minimum frequency in Hz.</param>
        /// <param name="fmaxHz">Maximum frequency in Hz.</param>
        /// <param name="val">Value to fill masked regions with.</param>
        /// <returns>Masked spectrogram.</returns>
        public static Tensor MaskFrequencies(
            Tensor spectrogram,
            int sampleRate,
            Tensor fminHz,
            Tensor fmaxHz,
            float val = 0.0f)
        {
            // Get magnitude and phase
            var magnitude = DSP.Magnitude(spectrogram);
            var phase = DSP.Phase(spectrogram);

            // Expand frequency bounds to match spectrogram shape
            fminHz = fminHz.expand(magnitude.shape);
            fmaxHz = fmaxHz.expand(magnitude.shape);

            // Validate frequency bounds
            if (any(fminHz >= fmaxHz).item<bool>())
            {
                throw new ArgumentException("Minimum frequency must be less than maximum frequency");
            }

            // Create frequency bins
            int nbins = (int)magnitude.size(-2);
            var binsHz = linspace(0, sampleRate / 2, nbins, device: spectrogram.device);

            // Expand bins to match shape for broadcasting
            var binsExpanded = binsHz.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                .expand(spectrogram.size(0), 1, -1, spectrogram.size(-1));

            // Create mask
            var mask = fminHz <= binsExpanded & binsExpanded < fmaxHz;

            // Apply mask
            magnitude = magnitude.masked_fill(mask, val);
            phase = phase.masked_fill(mask, val);

            // Reconstruct complex spectrogram
            return magnitude * exp(Complex.ImaginaryOne * phase);
        }

        /// <summary>
        /// Masks specific time regions in a spectrogram.
        /// </summary>
        /// <param name="spectrogram">Complex STFT tensor.</param>
        /// <param name="signalDuration">Duration of the signal in seconds.</param>
        /// <param name="tminS">Minimum time in seconds.</param>
        /// <param name="tmaxS">Maximum time in seconds.</param>
        /// <param name="val">Value to fill masked regions with.</param>
        /// <returns>Masked spectrogram.</returns>
        public static Tensor MaskTimesteps(
            Tensor spectrogram,
            float signalDuration,
            Tensor tminS,
            Tensor tmaxS,
            float val = 0.0f)
        {
            // Get magnitude and phase
            var magnitude = DSP.Magnitude(spectrogram);
            var phase = DSP.Phase(spectrogram);

            // Expand time bounds to match spectrogram shape
            tminS = tminS.expand(magnitude.shape);
            tmaxS = tmaxS.expand(magnitude.shape);

            // Validate time bounds
            if (any(tminS >= tmaxS).item<bool>())
            {
                throw new ArgumentException("Minimum time must be less than maximum time");
            }

            // Create time bins
            int nt = (int)magnitude.size(-1);
            var binsT = linspace(0, signalDuration, nt, device: spectrogram.device);

            // Expand bins to match shape for broadcasting
            var binsExpanded = binsT.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                .expand(spectrogram.size(0), 1, magnitude.size(-2), -1);

            // Create mask
            var mask = tminS <= binsExpanded & binsExpanded < tmaxS;

            // Apply mask
            magnitude = magnitude.masked_fill(mask, val);
            phase = phase.masked_fill(mask, val);

            // Reconstruct complex spectrogram
            return magnitude * exp(Complex.ImaginaryOne * phase);
        }

        /// <summary>
        /// Computes Mel-Frequency Cepstral Coefficients (MFCCs) from an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="sampleRate">Sample rate of the audio.</param>
        /// <param name="nMfcc">Number of MFCC coefficients.</param>
        /// <param name="nMels">Number of mel bands.</param>
        /// <param name="stftParams">STFT parameters.</param>
        /// <param name="melFmin">Minimum frequency for mel filter.</param>
        /// <param name="melFmax">Maximum frequency for mel filter.</param>
        /// <param name="logOffset">Offset for log operation.</param>
        /// <returns>MFCC tensor of shape (batch_size, channels, mfcc_coeffs, time_frames).</returns>
        public static Tensor MFCC(
            Tensor audio,
            int sampleRate,
            int nMfcc = 40,
            int nMels = 80,
            STFTParams stftParams = null,
            float melFmin = 0.0f,
            float? melFmax = null,
            float logOffset = 1e-6f)
        {
            // Compute mel spectrogram
            var melSpectrogram = MelSpectrogram(
                audio,
                sampleRate,
                stftParams ?? new STFTParams(),
                nMels,
                melFmin,
                melFmax);

            // Apply log
            melSpectrogram = log(melSpectrogram + logOffset);

            // Apply DCT
            var dctMatrix = DCTMatrix(nMfcc, nMels, audio.device);

            // Transpose for matrix multiplication
            var transposed = melSpectrogram.transpose(-1, -2);

            // Apply DCT
            var mfcc = transposed.matmul(dctMatrix);

            // Transpose back
            return mfcc.transpose(-1, -2);
        }

        /// <summary>
        /// Performs overlap-and-add reconstruction from windowed audio.
        /// </summary>
        /// <param name="windows">Windowed audio tensor of shape (batch_size*num_windows, channels, window_length).</param>
        /// <param name="originalBatchSize">Original batch size.</param>
        /// <param name="totalLength">Total length of the output signal.</param>
        /// <param name="windowLength">Length of each window.</param>
        /// <param name="hopLength">Hop length between windows.</param>
        /// <returns>Reconstructed audio tensor of shape (batch_size, channels, total_length).</returns>
        public static Tensor OverlapAndAdd(
            Tensor windows,
            int originalBatchSize,
            long totalLength,
            long windowLength,
            long hopLength)
        {
            var numWindows = windows.size(0) / originalBatchSize;
            var channels = windows.size(1);

            // Reshape to (batch_size, num_windows, channels, window_length)
            var reshapedWindows = windows.reshape(originalBatchSize, numWindows, channels, windowLength);

            // Permute to (batch_size, channels, num_windows, window_length)
            reshapedWindows = reshapedWindows.permute(0, 2, 1, 3);

            // Reshape for fold operation
            var foldInput = reshapedWindows.reshape(originalBatchSize * channels, numWindows, windowLength)
                                          .permute(0, 2, 1);

            // Use fold to perform overlap-add
            var folded = nn.functional.fold(
                foldInput,
                output_size: (1, totalLength),
                kernel_size: (1, windowLength),
                stride: (1, hopLength));

            // Create normalization mask
            var ones = ones_like(foldInput);
            var normalization = nn.functional.fold(
                ones,
               output_size: (1, totalLength),
                kernel_size: (1, windowLength),
                stride: (1, hopLength));

            // Apply normalization (avoid division by zero)
            normalization.clamp_(min: 1e-9);
            folded /= normalization;

            // Reshape to original dimensions (batch_size, channels, total_length)
            folded = folded.reshape(originalBatchSize, channels, totalLength);

            return folded;
        }

        /// <summary>
        /// Computes the phase of a complex STFT tensor.
        /// </summary>
        /// <param name="stft">Complex STFT tensor.</param>
        /// <returns>Phase tensor.</returns>
        public static Tensor Phase(Tensor stft)
        {
            return angle(stft);
        }

        /// <summary>
        /// Applies preemphasis filter to an audio tensor.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="coefficient">Preemphasis coefficient.</param>
        /// <returns>Filtered audio tensor.</returns>
        public static Tensor Preemphasis(Tensor audio, float coefficient = 0.85f)
        {
            var kernel = tensor(new float[] { 1, -coefficient, 0 })
                .reshape(1, 1, -1)
                .to(audio.device);

            // Handle each batch and channel separately
            var result = zeros_like(audio);

            for (int b = 0; b < audio.size(0); b++)
            {
                for (int c = 0; c < audio.size(1); c++)
                {
                    var x = audio[b, c].unsqueeze(0).unsqueeze(0);
                    result[b, c] = nn.functional.conv1d(x, kernel, padding: 1).squeeze();
                }
            }

            return result;
        }

        /// <summary>
        /// Resamples audio using linear interpolation.
        /// </summary>
        /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
        /// <param name="sourceSampleRate">Source sample rate.</param>
        /// <param name="targetSampleRate">Target sample rate.</param>
        /// <returns>Resampled audio tensor.</returns>
        public static Tensor ResampleLinear(
            Tensor audio,
            int sourceSampleRate,
            int targetSampleRate)
        {
            if (sourceSampleRate == targetSampleRate)
            {
                return audio;
            }

            var ratio = (double)targetSampleRate / sourceSampleRate;
            var batchSize = audio.size(0);
            var channels = audio.size(1);
            var inputLength = audio.size(2);
            var outputLength = (int)(inputLength * ratio);

            // Use interpolate for each batch and channel
            var result = zeros(new long[] { batchSize, channels, outputLength }, device: audio.device);

            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    var channel = audio[b, c].unsqueeze(0).unsqueeze(0);
                    var resampledChannel = nn.functional.interpolate(
                        channel,
                        size: new long[] { outputLength },
                        mode: InterpolationMode.Linear,
                        align_corners: true);

                    result[b, c] = resampledChannel.squeeze();
                }
            }

            return result;
        }

        #region STFT and Mel Spectrogram Methods

        /// <summary>
        /// Computes a mel spectrogram from a raw audio tensor, providing frequency analysis
        /// with perceptually relevant mel scaling.
        /// </summary>
        /// <param name="audio">The input audio tensor. Can be 1D (samples), 2D (channels, samples),
        /// or 3D (batch, channels, samples)</param>
        /// <param name="sampleRate">The sampling rate of the audio in Hz</param>
        /// <param name="nMels">Number of mel bands to generate</param>
        /// <param name="melFmin">Lowest frequency in Hz</param>
        /// <param name="melFmax">Highest frequency in Hz. Defaults to sampleRate/2 if null</param>
        /// <param name="windowLength">Length of the FFT window in samples</param>
        /// <param name="hopLength">Number of samples between successive frames</param>
        /// <param name="windowType">Type of window function to apply (e.g., hann, hamming)</param>
        /// <param name="center">Whether to pad the audio on both sides</param>
        /// <returns>Mel spectrogram tensor of shape (batch, channels, mel_bands, time_frames)</returns>
        public static Tensor MelSpectrogram(
            Tensor audio,
            int sampleRate,
            int nMels = 80,
            float melFmin = 0.0f,
            float? melFmax = null,
            int windowLength = 2048,
            int hopLength = 512,
            string windowType = "hann",
            bool center = true)
        {
            using var scope = NewDisposeScope();

            // Validate parameters
            if (audio.IsInvalid)
            {
                throw new ArgumentNullException(nameof(audio));
            }

            if (sampleRate <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive");
            }

            if (nMels <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(nMels), "Number of mel bands must be positive");
            }

            if (melFmin < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(melFmin), "Minimum frequency must be non-negative");
            }

            if (melFmax.HasValue && melFmax.Value <= melFmin)
            {
                throw new ArgumentOutOfRangeException(nameof(melFmax), "Maximum frequency must be greater than minimum frequency");
            }

            melFmax ??= sampleRate / 2.0f;

            // Compute STFT
            var stft = STFT(
                audio,
                windowLength,
                hopLength,
                windowType,
                center
            );

            // Get magnitudes
            var magnitudes = abs(stft);

            // Create mel filterbank
            var (nFft, _) = FFTLengths(windowLength);
            var melBasis = MelFilterbank(
                sampleRate,
                nMels,
                nFft,
                melFmin,
                melFmax.Value
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
        /// Computes a mel spectrogram from a raw audio tensor using the provided STFT parameters.
        /// </summary>
        /// <param name="audio">The input audio tensor</param>
        /// <param name="sampleRate">The sampling rate of the audio in Hz</param>
        /// <param name="nMels">Number of mel bands to generate</param>
        /// <param name="stftParams">STFT parameters encapsulated in an STFTParams object</param>
        /// <param name="melFmin">Lowest frequency in Hz</param>
        /// <param name="melFmax">Highest frequency in Hz. Defaults to sampleRate/2 if null</param>
        /// <returns>Mel spectrogram tensor of shape (batch, channels, mel_bands, time_frames)</returns>
        public static Tensor MelSpectrogram(
            Tensor audio,
            int sampleRate,
            STFTParams stftParams,
            int nMels = 80,
            float melFmin = 0.0f,
            float? melFmax = null)
        {
            stftParams ??= new STFTParams();

            return MelSpectrogram(
                audio,
                sampleRate,
                nMels,
                melFmin,
                melFmax,
                stftParams.WindowLength,
                stftParams.HopLength,
                stftParams.WindowType,
                stftParams.Center
            );
        }

        /// <summary>
        /// Computes the Short-Time Fourier Transform (STFT) of an audio signal, converting
        /// the time-domain signal into a time-frequency representation.
        /// </summary>
        /// <param name="audio">The input audio tensor. Can be 1D (samples), 2D (batch, samples) or (channels, samples),
        /// or 3D (batch, channels, samples)</param>
        /// <param name="windowLength">Length of the FFT window in samples</param>
        /// <param name="hopLength">Number of samples between successive frames</param>
        /// <param name="windowType">Type of window function to apply (e.g., hann, hamming)</param>
        /// <param name="center">Whether to pad the audio on both sides</param>
        /// <param name="paddingMode">Mode used for padding when center is true</param>
        /// <returns>Complex STFT tensor of shape (batch, channels, freq_bins, time_frames)</returns>
        public static Tensor STFT(
            Tensor audio,
            int windowLength = 2048,
            int hopLength = 512,
            string windowType = "hann",
            bool center = true,
            PaddingModes paddingMode = PaddingModes.Reflect)
        {
            using var scope = NewDisposeScope();

            // Validate parameters
            if (audio.IsInvalid)
            {
                throw new ArgumentNullException(nameof(audio));
            }

            if (windowLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(windowLength), "Window length must be positive");
            }

            if (hopLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(hopLength), "Hop length must be positive");
            }

            if (string.IsNullOrEmpty(windowType))
            {
                throw new ArgumentException("Window type cannot be null or empty", nameof(windowType));
            }

            // Store original tensor dimensions
            int originalDim = (int)audio.dim();

            // Normalize shape to (batch, channels, time)
            if (originalDim == 1)
            {
                // [samples] -> [1, 1, samples]
                audio = audio.unsqueeze(0).unsqueeze(0);
            }
            else if (originalDim == 2)
            {
                // If second dim is small, assume [batch, channels]
                if (audio.size(1) <= 4)
                {
                    // [batch, channels] -> [batch, channels, 1]
                    audio = audio.unsqueeze(-1);
                }
                else
                {
                    // [batch, samples] -> [batch, 1, samples]
                    audio = audio.unsqueeze(1);
                }
            }

            // Get dimensions after normalization
            var batchSize = (int)audio.size(0);
            var channels = (int)audio.size(1);

            // Create window on the same device as audio
            var window = GetWindow(windowType, windowLength, audio.device);

            // Apply padding if center is true
            Tensor paddedAudio = audio;
            if (center)
            {
                var paddingSize = windowLength / 2;
                paddedAudio = nn.functional.pad(
                    audio,
                    new long[] { paddingSize, paddingSize },
                    mode: paddingMode);
            }

            // Reshape for stft
            var flattened = paddedAudio.reshape(-1, paddedAudio.size(-1));

            // Perform STFT
            var stftResult = torch.stft(
                flattened,
                n_fft: windowLength,
                hop_length: hopLength,
                window: window,
                center: false, // We've already applied padding if needed
                return_complex: true
            );

            // Calculate expected dimensions
            var freqBins = windowLength / 2 + 1;

            // Reshape back to (batch, channels, freq, time)
            var reshapedStft = stftResult.reshape(batchSize, channels, freqBins, -1);

            return reshapedStft.MoveToOuterDisposeScope();
        }

        /// <summary>
        /// Computes the Short-Time Fourier Transform (STFT) of an audio signal using the provided STFT parameters.
        /// </summary>
        /// <param name="audio">The input audio tensor. Can be 1D (samples), 2D (batch, samples) or (channels, samples),
        /// or 3D (batch, channels, samples)</param>
        /// <param name="params">STFT parameters encapsulated in an STFTParams object</param>
        /// <returns>Complex STFT tensor of shape (batch, channels, freq_bins, time_frames)</returns>
        public static Tensor STFT(Tensor audio, STFTParams @params)
        {
            if (@params == null)
            {
                throw new ArgumentNullException(nameof(@params));
            }

            return STFT(
                audio,
                @params.WindowLength,
                @params.HopLength,
                @params.WindowType,
                @params.Center,
                @params.PaddingMode
            );
        }

        /// <summary>
        /// Creates a mel filterbank matrix for converting linear frequency spectra to mel-scale.
        /// </summary>
        /// <param name="sampleRate">The sampling rate of the audio in Hz</param>
        /// <param name="nMels">Number of mel bands to generate</param>
        /// <param name="nFft">FFT size</param>
        /// <param name="fMin">Lowest frequency in Hz</param>
        /// <param name="fMax">Highest frequency in Hz. Defaults to sampleRate/2 if null</param>
        /// <returns>A tensor containing the mel filterbank matrix of shape (nMels, nFreqs)</returns>
        private static Tensor MelFilterbank(
            int sampleRate,
            int nMels,
            int nFft,
            float fMin,
            float? fMax = null)
        {
            fMax ??= sampleRate / 2.0f;

            // Create frequencies array
            var fftFreqs = linspace(0f, sampleRate / 2f, (nFft / 2) + 1);

            // Convert Hz to mel
            var mels = from_array(new[] { fMin, fMax.Value }).HertzToMel();
            var melPoints = linspace(mels[0].item<float>(), mels[1].item<float>(), nMels + 2);
            var hzPoints = melPoints.MelToHertz();

            // Create filterbank matrix
            var filterbank = zeros(nMels, (nFft / 2) + 1);

            // Populate filters
            for (int i = 0; i < nMels; i++)
            {
                var filter = new float[(nFft / 2) + 1];
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

        #endregion STFT and Mel Spectrogram Methods

        /// <summary>
        /// Gets a DCT matrix for MFCC computation.
        /// </summary>
        private static Tensor DCTMatrix(int numMfcc, int numMels, Device device)
        {
            using var scope = NewDisposeScope();
            var melIndices = arange(numMels, device: device);
            var mfccIndices = arange(numMfcc, device: device).unsqueeze(-1);

            var dctMatrix = cos(mfccIndices * ((2 * melIndices) + 1) * Math.PI / (2 * numMels));
            dctMatrix *= sqrt(tensor(2.0f / numMels, device: device));

            // Adjust the first coefficient
            return dctMatrix.select(0, 0).mul_(1.0f / sqrt(tensor(2.0f, device: device))).MoveToOuterDisposeScope();
        }

        /// <summary>
        /// Computes the FFT lengths needed for mel filterbank computation based on the window length.
        /// Ensures that the FFT length is at least 2048 samples for adequate frequency resolution.
        /// </summary>
        /// <param name="windowLength">Length of the FFT window in samples</param>
        /// <returns>A tuple containing (nFft, nFreqs) where nFft is the FFT size and nFreqs is the number of frequency bins</returns>
        private static (int nFft, int nFreqs) FFTLengths(int windowLength)
        {
            var nFft = Math.Max(windowLength, 2048);
            // Get number of unique FFT frequencies
            var nFreqs = (nFft / 2) + 1;
            return (nFft, nFreqs);
        }
    }
}