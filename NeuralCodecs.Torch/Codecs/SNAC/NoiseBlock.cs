using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Codecs.SNAC;

public partial class SNAC
{
    /// <summary>
    /// Implements a noise injection block for audio processing networks.
    /// Adds learned-scale random noise to the input signal while preserving the input dimensions.
    /// </summary>
    public class NoiseBlock : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Weight-normalized 1x1 convolution used to learn noise scaling factors
        /// </summary>
        private readonly WNConv1d linear;

        /// <summary>
        /// Initializes a new instance of the NoiseBlock
        /// </summary>
        /// <param name="dim">Number of input/output channels</param>
        /// <remarks>
        /// Uses a 1x1 convolution without bias to learn channel-wise noise scaling
        /// </remarks>
        public NoiseBlock(int dim) : base($"NoiseBlock_{dim}")
        {
            linear = new WNConv1d(dim, dim, kernelSize: 1, useBias: false);
            RegisterComponents();
        }

        /// <summary>
        /// Performs forward pass of the noise block
        /// </summary>
        /// <param name="x">Input tensor of shape (batch, channels, time)</param>
        /// <returns>
        /// Output tensor with same shape as input:
        /// output = input + (noise * learned_scale)
        /// </returns>
        public override Tensor forward(Tensor x)
        {
            var (B, C, T) = (x.size(0), x.size(1), x.size(2));
            var noise = randn(new long[] { B, 1, T }, device: x.device, dtype: x.dtype);
            var h = linear.forward(x);
            var n = noise * h;
            return x + n;
        }
    }
}