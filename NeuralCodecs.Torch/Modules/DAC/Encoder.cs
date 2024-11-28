using NeuralCodecs.Torch.Modules;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

public class Encoder : Module<Tensor, Tensor>
{
    private readonly Sequential block; //TODO NAME

    public Encoder(
        int dModel = 64,
        int[] strides = null,
        int dLatent = 64) : base("Encoder")
    {
        strides ??= [2, 4, 8, 8];

        var layers = new List<Module<Tensor, Tensor>>
        {
            // First convolution layer
            new WNConv1d(1, dModel, kernelSize: 7, padding: 3)
        };

        // Add encoder blocks that double channels and downsample
        foreach (var stride in strides)
        {
            dModel *= 2;
            layers.Add(new EncoderBlock(dModel, stride: stride));
        }

        // Add final layers
        layers.AddRange(new Module<Tensor, Tensor>[]
        {
            new Snake1d(dModel),
            new WNConv1d(dModel, dLatent, kernelSize: 3, padding: 1)
        });

        block = Sequential(layers);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => block.forward(x);
}