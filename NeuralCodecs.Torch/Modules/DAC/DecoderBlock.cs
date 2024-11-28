using NeuralCodecs.Torch.Modules;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

public class DecoderBlock : Module<Tensor, Tensor>
{
    private readonly Sequential block;

    public DecoderBlock(int inputDim = 16, int outputDim = 8, int stride = 1)
        : base($"DecoderBlock_{inputDim}_{outputDim}")
    {
        block = Sequential(
            new Snake1d(inputDim),
            new WNConvTranspose1d(
                inputDim,
                outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: (int)Math.Ceiling(stride / 2.0)
            ),
            new ResidualUnit(outputDim, dilation: 1),
            new ResidualUnit(outputDim, dilation: 3),
            new ResidualUnit(outputDim, dilation: 9)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => block.forward(x);
}