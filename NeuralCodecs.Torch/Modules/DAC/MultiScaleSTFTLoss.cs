using NeuralCodecs.Torch.AudioTools;
using NeuralCodecs.Torch.Utils;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Multi-scale STFT loss supporting both AudioSignal and raw tensors
/// </summary>
public class MultiScaleSTFTLoss : AudioLossBase
{
    private readonly STFTParams[] _stftParams;
    private readonly Module<Tensor, Tensor, Tensor> _lossFn;
    private readonly float _clampEps;
    private readonly float _magWeight;
    private readonly float _logWeight;
    private readonly float _pow;
    private readonly float _weight;

    public MultiScaleSTFTLoss(
        int[] windowLengths = null,
        Module<Tensor, Tensor, Tensor> lossFn = null,
        float clampEps = 1e-5f,
        float magWeight = 1.0f,
        float logWeight = 1.0f,
        float pow = 2.0f,
        float weight = 1.0f,
        bool matchStride = false,
        string windowType = null,
        int sampleRate = 44100) : base(nameof(MultiScaleSTFTLoss), sampleRate)
    {
        windowLengths ??= new[] { 2048, 512 };
        lossFn ??= nn.L1Loss();

        _stftParams = windowLengths.Select(w => new STFTParams(
            windowLength: w,
            hopLength: w / 4,
            windowType: windowType,
            matchStride: matchStride)).ToArray();

        _lossFn = lossFn;
        _clampEps = clampEps;
        _magWeight = magWeight;
        _logWeight = logWeight;
        _pow = pow;
        _weight = weight;

        RegisterComponents();
    }
    public override Tensor forward(Tensor x, Tensor y)
    {
        var loss = zeros(1, device: x.device);

        foreach (var stftParam in _stftParams)
        {
            Tensor xStft = AudioTensorOps.ComputeSTFT(
                x,
                stftParam.WindowLength,
                stftParam.HopLength,
                stftParam.WindowType
            );

            Tensor yStft = AudioTensorOps.ComputeSTFT(
                y,
                stftParam.WindowLength,
                stftParam.HopLength,
                stftParam.WindowType
            );

            var xMag = abs(xStft);
            var yMag = abs(yStft);

            // Log magnitude loss
            loss += _logWeight * _lossFn.forward(
                xMag.clamp(_clampEps).pow(_pow).log10(),
                yMag.clamp(_clampEps).pow(_pow).log10()
            );

            // Raw magnitude loss
            loss += _magWeight * _lossFn.forward(xMag, yMag);
        }

        return loss * _weight;
    }
    //public override Tensor forward(Tensor x, Tensor y)
    //{
    //    var (xAudio, isXSignal) = GetAudioTensor(x);
    //    var (yAudio, isYSignal) = GetAudioTensor(y);

    //    var loss = zeros(1, device: xAudio.device);

    //    foreach (var stftParam in _stftParams)
    //    {
    //        Tensor xStft, yStft;

    //        if (x is AudioSignal signal)
    //        {
    //            ((signal.stft(stftParam.window_length, stftParam.hop_length, stftParam.window_type);
    //            xStft = ((AudioSignal)x).stft_data;
    //        }
    //        else
    //        {
    //            xStft = AudioTensorOps.ComputeSTFT(
    //                xAudio,
    //                stftParam.window_length,
    //                stftParam.hop_length,
    //                stftParam.window_type
    //            );
    //        }

    //        if (isYSignal)
    //        {
    //            ((AudioSignal)y).stft(stftParam.window_length, stftParam.hop_length, stftParam.window_type);
    //            yStft = ((AudioSignal)y).stft_data;
    //        }
    //        else
    //        {
    //            yStft = AudioTensorOps.ComputeSTFT(
    //                yAudio,
    //                stftParam.window_length,
    //                stftParam.hop_length,
    //                stftParam.window_type
    //            );
    //        }

    //        var xMag = torch.abs(xStft);
    //        var yMag = torch.abs(yStft);

    //        // Log magnitude loss
    //        loss += _logWeight * _lossFn.forward(
    //            xMag.clamp(_clampEps).pow(_pow).log10(),
    //            yMag.clamp(_clampEps).pow(_pow).log10()
    //        );

    //        // Raw magnitude loss
    //        loss += _magWeight * _lossFn.forward(xMag, yMag);
    //    }

    //    return loss * _weight;
    //}
}
