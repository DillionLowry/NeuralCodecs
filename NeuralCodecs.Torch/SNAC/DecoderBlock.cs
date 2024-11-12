using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.SNAC;

public partial class SNAC
{
    /// <summary>
    /// Implements a decoder block for audio signal processing that progressively upsamples
    /// and refines the signal through transposed convolutions and residual units.
    /// </summary>
    public class DecoderBlock : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Sequential container for all decoder operations
        /// </summary>
        private readonly Sequential block;

        /// <summary>
        /// Initializes a new instance of the DecoderBlock
        /// </summary>
        /// <param name="inputDim">Input channel dimension</param>
        /// <param name="outputDim">Output channel dimension</param>
        /// <param name="stride">Stride factor for upsampling</param>
        /// <param name="noise">Whether to include noise injection</param>
        /// <param name="groups">Number of groups for grouped convolutions</param>
        public DecoderBlock(
            int inputDim = 16,
            int outputDim = 8,
            int stride = 1,
            bool noise = false,
            int groups = 1) : base($"DecoderBlock_{inputDim}_{outputDim}")
        {
            var layers = new List<Module<Tensor, Tensor>> {
            new Snake1d(inputDim),
            new WNConvTranspose1d(
                inputDim,
                outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: (int)Math.Ceiling(stride / 2.0),
                outputPadding: stride % 2
            )
        };

            if (noise)
            {
                layers.Add(new NoiseBlock(outputDim));
            }

            layers.AddRange(new Module<Tensor, Tensor>[] {
            new ResidualUnit(outputDim, dilation: 1, groups: groups),
            new ResidualUnit(outputDim, dilation: 3, groups: groups),
            new ResidualUnit(outputDim, dilation: 9, groups: groups)
        });

            block = Sequential(layers);
            RegisterComponents();
        }

        /// <summary>
        /// Performs forward pass through the decoder block
        /// </summary>
        /// <param name="x">Input tensor of shape (batch, channels, time)</param>
        /// <returns>
        /// Upsampled and processed tensor
        /// </returns>
        public override Tensor forward(Tensor x) => block.forward(x);
    }
}