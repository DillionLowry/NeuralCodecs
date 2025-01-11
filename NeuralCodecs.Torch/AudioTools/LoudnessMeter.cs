using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.AudioTools;

public class LoudnessMeter : IDisposable
{
    public const float GAIN_FACTOR = 0.11512925464970229f; // ln(10) / 20
    private readonly Tensor _K_weights;
    private readonly int _sampleRate;
    private readonly float _blockSize;
    private readonly bool _useFir;
    private readonly int _zeros;
    private readonly Dictionary<string, KFilter> _filters;

    public class KFilter
    {
        public float[] B { get; set; }
        public float[] A { get; set; }
        public float PassbandGain { get; set; }
    }

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

        _K_weights = torch.tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.41f, 1.41f });

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
        var z = (1.0f / (T_g * _sampleRate)) * unfolded.pow(2).sum(2);

        // Calculate loudness for each block
        var l = -0.691f + 10.0f * torch.log10(
            (_K_weights[..(int)inputData.size(1)].unsqueeze(0).unsqueeze(-1) * z).sum(1, keepdim: true));
        l = l.expand_as(z);

        // Gating block indices above absolute threshold
        var Gamma_a = -70.0f; // -70 LUFS absolute threshold
        var z_avg_gated = z.clone();
        z_avg_gated[l <= Gamma_a] = 0;
        var masked = l > Gamma_a;
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2).clamp(min: 1);

        // Calculate relative threshold
        var Gamma_r = -0.691f + (10.0f * torch.log10(
    (z_avg_gated * _K_weights[..(int)inputData.size(1)]).sum(-1))) - 10.0f;
        Gamma_r = Gamma_r.unsqueeze(-1).unsqueeze(-1);
        Gamma_r = Gamma_r.expand(inputData.size(0), inputData.size(1), l.size(-1));

        // Apply both absolute and relative thresholds
        z_avg_gated = z.clone();
        z_avg_gated[l <= Gamma_a] = 0;
        z_avg_gated[l <= Gamma_r] = 0;
        masked = (l > Gamma_a) & (l > Gamma_r);
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2).clamp(min: 1);

        // Handle NaN and Inf values
        z_avg_gated = torch.nan_to_num(z_avg_gated, 0.0f);

        // Calculate LUFS
        var LUFS = -0.691f + (10.0f * torch.log10(
            (_K_weights[..(int)inputData.size(1)] * z_avg_gated).sum(1)));

        return LUFS.@float();
    }

    private Tensor ApplyFilter(Tensor data)
    {
        if (data.is_cuda || _useFir)
        {
            return ApplyFilterGPU(data);
        }
        return ApplyFilterCPU(data);
    }

    private Tensor ApplyFilterCPU(Tensor data)
    {
        foreach (var filter in _filters.Values)
        {
            var a_coeffs = torch.tensor(filter.A).@float().to(data.device);
            var b_coeffs = torch.tensor(filter.B).@float().to(data.device);

            data = data.permute(0, 2, 1);
            data = LFilter(b_coeffs, a_coeffs, data);
            data = data.permute(0, 2, 1);
            data *= filter.PassbandGain;
        }
        return data;
    }

    private Tensor ApplyFilterGPU(Tensor data)
    {
        // Create impulse responses for each filter
        foreach (var filter in _filters.Values)
        {
            var impulse = torch.zeros(_zeros);
            impulse[0] = 1.0f;

            var fir = LFilter(
                torch.tensor(filter.B).@float(),
                torch.tensor(filter.A).@float(),
                impulse);

            // Apply filtering using convolution
            data = torch.nn.functional.conv1d(
                data.reshape(-1, 1, data.size(-1)),
                fir.flip(-1).reshape(1, 1, -1),
                padding: _zeros / 2
            );

            data *= filter.PassbandGain;
        }
        return data;
    }

    private Tensor LFilter(Tensor b, Tensor a, Tensor x)
    {
        var y = torch.zeros_like(x);
        int n = (int)x.size(-1);

        // Normalize filter coefficients by a[0]
        b /= a[0];
        a /= a[0];

        // Direct form II transposed implementation
        for (int i = 0; i < n; i++)
        {
            y[i] = b[0] * x[i];

            for (int j = 1; j < b.size(0) && i - j >= 0; j++)
            {
                y[i] += b[j] * x[i - j];
            }

            for (int j = 1; j < a.size(0) && i - j >= 0; j++)
            {
                y[i] -= a[j] * y[i - j];
            }
        }

        return y;
    }

    private Tensor Unfold(Tensor input, int kernelSize, int stride)
    {
        var batchSize = input.size(0);
        var channels = input.size(1);
        var length = input.size(2);

        var outputLength = (length - kernelSize) / stride + 1;
        var output = torch.zeros(batchSize, channels, outputLength, kernelSize);

        for (int i = 0; i < outputLength; i++)
        {
            var start = i * stride;
            output[.., i, ..] = input[.., start..(start + kernelSize)];
        }

        return output;
    }

    public void Dispose()
    {
        _K_weights?.Dispose();
    }
}