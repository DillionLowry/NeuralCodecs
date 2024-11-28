using NeuralCodecs.Torch.Modules;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

public class EncoderBlock : Module<Tensor, Tensor>
{
    private readonly Sequential block;

    public EncoderBlock(int dim = 16, int stride = 1) : base($"EncoderBlock_{dim}")
    {
        block = Sequential(
            new ResidualUnit(dim / 2, dilation: 1),
            new ResidualUnit(dim / 2, dilation: 3),
            new ResidualUnit(dim / 2, dilation: 9),
            new Snake1d(dim / 2),
            new WNConv1d(
                dim / 2,
                dim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: (int)Math.Ceiling(stride / 2.0)
            )
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => block.forward(x);
}