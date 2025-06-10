using TorchSharp;
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
        using var scope = torch.NewDisposeScope();
        var loss = _l1Loss.forward(x, y);
        return loss.mul_(_weight).MoveToOuterDisposeScope();
    }
}