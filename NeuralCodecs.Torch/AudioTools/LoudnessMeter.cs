using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Provides loudness measurement functionality for audio tensors.
/// </summary>
public class LoudnessMeter : IDisposable
{
    /// <summary>
    /// Gain factor used in loudness calculations.
    /// </summary>
    public const float GAIN_FACTOR = 0.11512925464970229f; // ln(10) / 20

    private readonly float _blockSize;
    private readonly Dictionary<string, KFilter> _filters;
    private readonly Tensor _kWeights;
    private readonly int _sampleRate;
    private readonly bool _useFir;
    private readonly int _zeros;
    private bool _disposed = false;

    public LoudnessMeter(
        int sampleRate = 44100,
        string filterClass = "K-weighting",
        float blockSize = 0.400f,
        int zeros = 512,
        bool useFir = false)
    {
        _sampleRate = sampleRate;
        _blockSize = blockSize;
        _useFir = useFir;
        _zeros = zeros;

        _kWeights = tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.41f, 1.41f });

        // Initialize K-weighting filters
        _filters = new Dictionary<string, KFilter>
        {
            ["high_shelf"] = new KFilter
            {
                B = new float[] { 1.53512485958697f, -2.69169618940638f, 1.19839281085285f },
                A = new float[] { 1.0f, -1.69065929318241f, 0.73248077421585f },
                PassbandGain = 1.0f
            },
            ["high_pass"] = new KFilter
            {
                B = new float[] { 1.0f, -2.0f, 1.0f },
                A = new float[] { 1.0f, -1.99004745483398f, 0.99007225036621f },
                PassbandGain = 1.0f
            }
        };
    }

    /// <summary>
    /// Disposes resources used by the analyzer.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    public Tensor IntegratedLoudness(AudioSignal signal)
    {
        var audioData = signal.AudioData;
        var inputData = audioData.clone();

        // Apply frequency weighting filters
        inputData = ApplyFilter(inputData);

        var T_g = _blockSize;
        var overlap = 0.75f; // 75% overlap
        var step = 1.0f - overlap;

        int kernelSize = (int)(T_g * _sampleRate);
        int stride = (int)(T_g * _sampleRate * step);

        // Create overlapping windows
        var unfolded = Unfold(inputData.permute(0, 2, 1), kernelSize, stride);
        unfolded = unfolded.transpose(-1, -2);

        // Calculate power for each block
        var z = 1.0f / (T_g * _sampleRate) * unfolded.pow(2).sum(2);

        // Calculate loudness for each block
        var l = -0.691f + (10.0f * log10(
            (_kWeights[..(int)inputData.size(1)].unsqueeze(0).unsqueeze(-1) * z).sum(1, keepdim: true)));
        l = l.expand_as(z);

        // Gating block indices above absolute threshold
        var Gamma_a = -70.0f; // -70 LUFS absolute threshold
        var z_avg_gated = z.clone();
        z_avg_gated[l <= Gamma_a] = 0;
        var masked = l > Gamma_a;
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2).clamp(min: 1);

        // Calculate relative threshold
        var Gamma_r = -0.691f + (10.0f * log10(
            (z_avg_gated * _kWeights[..(int)inputData.size(1)]).sum(-1))) - 10.0f;
        Gamma_r = Gamma_r.unsqueeze(-1).unsqueeze(-1);
        Gamma_r = Gamma_r.expand(inputData.size(0), inputData.size(1), l.size(-1));

        // Apply both absolute and relative thresholds
        z_avg_gated = z.clone();
        z_avg_gated[l <= Gamma_a] = 0;
        z_avg_gated[l <= Gamma_r] = 0;
        masked = l > Gamma_a & l > Gamma_r;
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2).clamp(min: 1);

        // Handle NaN and Inf values
        z_avg_gated = nan_to_num(z_avg_gated, 0.0f);

        // Calculate LUFS
        var LUFS = -0.691f + (10.0f * log10(
            (_kWeights[..(int)inputData.size(1)] * z_avg_gated).sum(1)));

        return LUFS.@float();
    }

    /// <summary>
    /// Calculates the integrated loudness of an audio tensor according to ITU-R BS.1770-4.
    /// </summary>
    /// <param name="audio">Audio tensor of shape (batch_size, channels, samples).</param>
    /// <returns>Loudness in LUFS (Loudness Units relative to Full Scale).</returns>
    public Tensor IntegratedLoudness(Tensor audio)
    {
        var inputData = audio.clone();

        // Apply K-weighting filter
        inputData = ApplyFilter(inputData);

        // Gate block size and overlap
        var T_g = _blockSize;
        var overlap = 0.75f;
        var step = 1.0f - overlap;

        // Convert to samples
        int kernelSize = (int)(T_g * _sampleRate);
        int stride = (int)(T_g * _sampleRate * step);

        // Extract overlapping blocks
        var unfolded = Unfold(inputData.permute(0, 2, 1), kernelSize, stride);
        unfolded = unfolded.transpose(-1, -2);

        // Calculate mean square for each block
        var z = 1.0f / (T_g * _sampleRate) * unfolded.pow(2).sum(2);

        // Convert to loudness (l_k values)
        var l = -0.691f + (10.0f * torch.log10(
            (_kWeights[..(int)inputData.size(1)].unsqueeze(0).unsqueeze(-1) * z).sum(1, keepdim: true)));

        // Expand to match z shape
        l = l.expand_as(z);

        // Absolute gating (blocks below -70 LUFS are ignored)
        var Gamma_a = -70.0f;
        var z_avg_gated = z.clone();
        z_avg_gated = torch.where(l <= Gamma_a, torch.zeros_like(z_avg_gated), z_avg_gated);

        // Calculate mask for non-gated blocks
        var masked = l > Gamma_a;
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2).clamp(min: 1);

        // Calculate relative gating threshold
        var Gamma_r = -0.691f + (10.0f * torch.log10(
            (z_avg_gated * _kWeights[..(int)inputData.size(1)]).sum(-1))) - 10.0f;

        // Expand dimensions to match l
        Gamma_r = Gamma_r.unsqueeze(-1).unsqueeze(-1);
        Gamma_r = Gamma_r.expand(inputData.size(0), inputData.size(1), l.size(-1));

        // Apply relative gating
        z_avg_gated = z.clone();
        z_avg_gated = torch.where(l <= Gamma_a | l <= Gamma_r, torch.zeros_like(z_avg_gated), z_avg_gated);

        // Calculate mask for blocks above both thresholds
        masked = (l > Gamma_a) & (l > Gamma_r);
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2).clamp(min: 1);

        // Handle NaN values
        z_avg_gated = torch.nan_to_num(z_avg_gated, 0.0f);

        // Calculate final LUFS value
        var LUFS = -0.691f + (10.0f * torch.log10(
            (_kWeights[..(int)inputData.size(1)] * z_avg_gated).sum(1)));

        return LUFS.@float();
    }

    /// <summary>
    /// Normalizes audio to a target loudness level.
    /// </summary>
    /// <param name="audio">Audio tensor to normalize.</param>
    /// <param name="targetDb">Target loudness level in LUFS.</param>
    /// <returns>Normalized audio tensor.</returns>
    public Tensor NormalizeAudio(Tensor audio, float targetDb = -24.0f)
    {
        var db = torch.tensor(targetDb).to(audio.device);
        var refDb = IntegratedLoudness(audio);
        var gain = db - refDb;
        gain = torch.exp(gain * GAIN_FACTOR);

        // Apply gain
        return audio * gain.unsqueeze(-1).unsqueeze(-1);
    }

    /// <summary>
    /// Disposes resources used by the analyzer.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _kWeights?.Dispose();
            }
            _disposed = true;
        }
    }

    /// <summary>
    /// Applies K-weighting filter to audio.
    /// </summary>
    /// <param name="audio">Audio tensor to filter.</param>
    /// <returns>Filtered audio tensor.</returns>
    private Tensor ApplyFilter(Tensor audio)
    {
        if (audio.is_cuda || _useFir)
        {
            return ApplyFilterGPU(audio);
        }
        return ApplyFilterCPU(audio);
    }

    /// <summary>
    /// CPU implementation of K-weighting filter.
    /// </summary>
    private Tensor ApplyFilterCPU(Tensor audio)
    {
        var data = audio.clone();

        foreach (var filter in _filters.Values)
        {
            var a_coeffs = torch.tensor(filter.A).@float().to(data.device);
            var b_coeffs = torch.tensor(filter.B).@float().to(data.device);

            // Reshape for filter application
            data = data.permute(0, 2, 1);

            // Apply IIR filter
            data = LFilter(b_coeffs, a_coeffs, data);

            // Reshape back
            data = data.permute(0, 2, 1);

            // Apply passband gain
            data *= filter.PassbandGain;
        }

        return data;
    }

    /// <summary>
    /// GPU implementation of K-weighting filter using FIR approximation.
    /// </summary>
    private Tensor ApplyFilterGPU(Tensor audio)
    {
        var data = audio.clone();

        foreach (var filter in _filters.Values)
        {
            // Create impulse
            var impulse = torch.zeros(_zeros, device: audio.device);
            impulse[0] = 1.0f;

            // Create FIR approximation
            var a_coeffs = torch.tensor(filter.A).@float().to(audio.device);
            var b_coeffs = torch.tensor(filter.B).@float().to(audio.device);
            var fir = LFilter(b_coeffs, a_coeffs, impulse);

            // Apply FIR filter using convolution
            for (int b = 0; b < data.size(0); b++)
            {
                for (int c = 0; c < data.size(1); c++)
                {
                    var channel = data[b, c].unsqueeze(0).unsqueeze(0);

                    // Apply convolution
                    var filtered = torch.nn.functional.conv1d(
                        channel,
                        fir.flip(-1).reshape(1, 1, -1),
                        padding: _zeros / 2
                    );

                    data[b, c] = filtered.squeeze();
                }
            }

            // Apply passband gain
            data *= filter.PassbandGain;
        }

        return data;
    }

    /// <summary>
    /// Applies an IIR filter to a signal (like SciPy's lfilter).
    /// </summary>
    /// <param name="b">Numerator coefficients.</param>
    /// <param name="a">Denominator coefficients.</param>
    /// <param name="x">Input signal.</param>
    /// <returns>Filtered signal.</returns>
    private Tensor LFilter(Tensor b, Tensor a, Tensor x)
    {
        var y = torch.zeros_like(x);
        int n = (int)x.size(-1);

        // Normalize by a[0]
        b /= a[0];
        a /= a[0];

        // Apply filter
        for (int i = 0; i < n; i++)
        {
            // Direct terms (b coefficients)
            y[i] = b[0] * x[i];

            // Add delayed input terms
            for (int j = 1; j < b.size(0) && i - j >= 0; j++)
            {
                y[i] += b[j] * x[i - j];
            }

            // Subtract delayed output terms
            for (int j = 1; j < a.size(0) && i - j >= 0; j++)
            {
                y[i] -= a[j] * y[i - j];
            }
        }

        return y;
    }

    /// <summary>
    /// Unfolds a tensor into overlapping blocks.
    /// </summary>
    private Tensor Unfold(Tensor input, int kernelSize, int stride)
    {
        var batchSize = input.size(0);
        var channels = input.size(1);
        var length = input.size(2);

        var outputLength = ((length - kernelSize) / stride) + 1;
        var output = zeros(batchSize, channels, outputLength, kernelSize);

        for (int i = 0; i < outputLength; i++)
        {
            var start = i * stride;
            output[.., i, ..] = input[.., start..(start + kernelSize)];
        }

        return output;
    }

    /// <summary>
    /// K-weighting filter parameters.
    /// </summary>
    public class KFilter
    {
        /// <summary>
        /// Denominator coefficients.
        /// </summary>
        public float[] A { get; set; }

        /// <summary>
        /// Numerator coefficients.
        /// </summary>
        public float[] B { get; set; }

        /// <summary>
        /// Passband gain.
        /// </summary>
        public float PassbandGain { get; set; }
    }
}