using NeuralCodecs.Torch.AudioTools;
using NeuralCodecs.Torch.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Mel-spectrogram loss supporting both AudioSignal and raw tensors
/// </summary>
public class MelSpectrogramLoss : AudioLossBase
{
    private readonly int[] _nMels;
    private readonly STFTParams[] _stftParams;
    private readonly Module<Tensor, Tensor, Tensor> _lossFn;
    private readonly float _clampEps;
    private readonly float _magWeight;
    private readonly float _logWeight;
    private readonly float _pow;
    private readonly float _weight;
    private readonly float[] _melFmin;
    private readonly float?[] _melFmax;
    private readonly bool _matchStride;
    private readonly string _windowType;

    /// <summary>
    /// Initializes a new instance of MelSpectrogramLoss
    /// </summary>
    public MelSpectrogramLoss(
        int[] nMels = null,
        int[] windowLengths = null,
        Module<Tensor, Tensor, Tensor> lossFn = null,
        float clampEps = 1e-5f,
        float magWeight = 1.0f,
        float logWeight = 1.0f,
        float pow = 2.0f,
        float weight = 1.0f,
        bool matchStride = false,
        float[] melFmin = null,
        float?[] melFmax = null,
        string windowType = "hann",
        int sampleRate = 44100) : base(nameof(MelSpectrogramLoss), sampleRate)
    {
        nMels ??= new[] { 150, 80 };
        windowLengths ??= new[] { 2048, 512 };
        melFmin ??= Enumerable.Repeat(0f, nMels.Length).ToArray();
        melFmax ??= Enumerable.Repeat((float?)null, nMels.Length).ToArray();
        lossFn ??= nn.L1Loss();

        // Validate input lengths match
        if (nMels.Length != windowLengths.Length ||
            nMels.Length != melFmin.Length ||
            nMels.Length != melFmax.Length)
        {
            throw new ArgumentException("All parameter arrays must have same length");
        }

        _nMels = nMels;
        _stftParams = windowLengths.Select(w => new STFTParams(
            windowLength: w,
            hopLength: w / 4,
            matchStride: matchStride,
            windowType: windowType
        )).ToArray();

        _lossFn = lossFn;
        _clampEps = clampEps;
        _magWeight = magWeight;
        _logWeight = logWeight;
        _pow = pow;
        _weight = weight;
        _melFmin = melFmin;
        _melFmax = melFmax;
        _matchStride = matchStride;
        _windowType = windowType;

        RegisterComponents();
    }

    /// <summary>
    /// Computes mel spectrogram for either AudioSignal or raw tensor
    /// </summary>
    private Tensor ComputeMelSpec(
        object input,
        int nMels,
        float melFmin,
        float? melFmax,
        STFTParams stftParams)
    {
        if (input is AudioSignal signal)
        {
            var kwargs = new Dictionary<string, object>
            {
                ["window_length"] = stftParams.WindowLength,
                ["hop_length"] = stftParams.HopLength,
                ["window_type"] = stftParams.WindowType
            };
            return signal.MelSpectrogram(nMels, melFmin, melFmax, kwargs);
        }
        else if (input is Tensor audio)
        {
            return AudioTensorOps.ComputeMelSpectrogram(
                audio,
                _sampleRate,
                nMels,
                melFmin,
                melFmax,
                stftParams.WindowLength,
                stftParams.HopLength,
                stftParams.WindowType
            );
        }
        throw new ArgumentException("Input must be AudioSignal or Tensor");
    }
    // device: x is AudioSignal signal ? signal.device :
    public override Tensor forward(Tensor x, Tensor y)
    {
        var loss = zeros(1, device: x.device);

        for (int i = 0; i < _stftParams.Length; i++)
        {
            var xMels = ComputeMelSpec(x, _nMels[i], _melFmin[i], _melFmax[i], _stftParams[i]);
            var yMels = ComputeMelSpec(y, _nMels[i], _melFmin[i], _melFmax[i], _stftParams[i]);

            // Log magnitude loss
            loss += _logWeight * _lossFn.forward(
                xMels.clamp(_clampEps).pow(_pow).log10(),
                yMels.clamp(_clampEps).pow(_pow).log10()
            );

            // Raw magnitude loss
            loss += _magWeight * _lossFn.forward(xMels, yMels);
        }

        return loss * _weight;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _lossFn?.Dispose();
        }
        base.Dispose(disposing);
    }
}