using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public class L1Loss : Module<AudioSignal, AudioSignal, Tensor>
{
    private readonly string attribute;
    private readonly float weight;
    private readonly Loss<Tensor, Tensor, Tensor> l1Loss;

    public L1Loss(string attribute = "audio_data", float weight = 1.0f) : base("L1Loss")
    {
        this.attribute = attribute;
        this.weight = weight;
        this.l1Loss = nn.L1Loss();
        RegisterComponents();
    }

    public override Tensor forward(AudioSignal x, AudioSignal y)
    {
        if (x is AudioSignal)
        {
            var xData = GetAttribute(x, attribute);
            var yData = GetAttribute(y, attribute);
            return l1Loss.forward(xData, yData);
        }
        return l1Loss.forward(x.audio_data, y.audio_data);
    }

    private Tensor GetAttribute(AudioSignal signal, string attr)
    {
        return attr switch
        {
            "audio_data" => signal.audio_data,
            _ => throw new ArgumentException($"Unknown attribute: {attr}")
        };
    }
}