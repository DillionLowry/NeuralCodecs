using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// L1 Loss between audio signals or raw tensors
/// </summary>
public class L1Loss : AudioLossBase
{
    private readonly string _attribute;
    private readonly float _weight;
    private readonly TorchSharp.Modules.L1Loss _l1Loss;

    public L1Loss(
        string attribute = "audio_data",
        float weight = 1.0f,
        int sampleRate = 44100) : base(nameof(L1Loss), sampleRate)
    {
        _attribute = attribute;
        _weight = weight;
        _l1Loss = nn.L1Loss();
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor y)
    {
        //var (xAudio, isXSignal) = GetAudioTensor(x);
        //var (yAudio, isYSignal) = GetAudioTensor(y);

        //if (isXSignal && isYSignal && _attribute != "audio_data")
        //{
        //    // Use reflection only if both are AudioSignal and different attribute requested
        //    xAudio = (Tensor)typeof(AudioSignal).GetProperty(_attribute).GetValue((AudioSignal)x);
        //    yAudio = (Tensor)typeof(AudioSignal).GetProperty(_attribute).GetValue((AudioSignal)y);
        //}

        return _l1Loss.forward(x, y) * _weight;
    }
}