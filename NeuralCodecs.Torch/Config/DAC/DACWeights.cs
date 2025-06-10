using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Config.DAC;

public class DACWeights
{
    public Dictionary<string, Tensor> StateDict { get; }
    public Dictionary<string, object>? Metadata { get; }

    public DACWeights(
        Dictionary<string, Tensor> stateDict,
        Dictionary<string, object>? metadata = null)
    {
        StateDict = stateDict ?? throw new ArgumentNullException(nameof(stateDict));
        Metadata = metadata;
    }
}