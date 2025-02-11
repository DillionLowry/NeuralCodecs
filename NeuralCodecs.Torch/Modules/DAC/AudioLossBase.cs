using NeuralCodecs.Torch.AudioTools;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Base class for audio losses supporting both AudioSignal and raw tensors
/// </summary>
public abstract class AudioLossBase : Module<Tensor, Tensor, Tensor>
{
    protected readonly int _sampleRate;

    protected AudioLossBase(string name, int sampleRate = 44100) : base(name)
    {
        _sampleRate = sampleRate;
    }

    protected (Tensor audio, bool isAudioSignal) GetAudioTensor(object input)
    {
        if (input is AudioSignal signal)
        {
            return (signal.AudioData, true);
        }
        else if (input is Tensor tensor)
        {
            return (tensor, false);
        }
        throw new ArgumentException("Input must be either AudioSignal or Tensor");
    }
}