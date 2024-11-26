using NeuralCodecs.Torch.Modules;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public class Decoder : Module<Tensor, Tensor>
{
    private readonly Sequential model;

    public Decoder(
        int inputChannel,
        int channels,
        int[] rates,
        int dOut = 1) : base("Decoder")
    {
        // First convolution layer
        var layers = new List<Module<Tensor, Tensor>>
        {
            new WNConv1d(inputChannel, channels, kernelSize: 7, padding: 3)
        };

        // Add upsampling + residual blocks
        int outputDim = 1;
        for (int i = 0; i < rates.Length; i++)
        {
            int inputDim = channels / (1 << i);
            outputDim = channels / (1 << (i + 1));
            layers.Add(new DecoderBlock(inputDim, outputDim, rates[i]));
        }

        // Add final layers
        layers.AddRange(new Module<Tensor, Tensor>[]
        {
            new Snake1d(outputDim),
            new WNConv1d(outputDim, dOut, kernelSize: 7, padding: 3),
            nn.Tanh()
        });

        model = nn.Sequential(layers);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => model.forward(x);
}