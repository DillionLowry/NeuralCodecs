using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.SNAC;

public partial class SNAC
{
    /// <summary>
    /// Implements an audio encoder that processes input signals through multiple encoder blocks
    /// and a final weight-normalized convolution layer.
    /// </summary>
    public class Encoder : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Sequential container for all encoder operations
        /// </summary>
        private readonly Sequential block;

        /// <summary>
        /// Initializes a new instance of the Encoder
        /// </summary>
        /// <param name="dModel">Model dimension/number of channels</param>
        /// <param name="strides">List of stride values for each encoder block</param>
        /// <param name="depthwise">Whether to use depthwise separable convolutions</param>
        /// <param name="attnWindowSize">Size of local attention window (optional)</param>
        public Encoder(
            int dModel = 64,
            int[] strides = null,
            bool depthwise = false,
            int? attnWindowSize = 32) : base("Encoder")
        {
            strides ??= new[] { 3, 3, 7, 7 };
            var layers = new List<Module<Tensor, Tensor>>
            {
                new WNConv1d(1, dModel, kernelSize: 7, padding: 3)
            };

            // Create EncoderBlocks that double channels as they downsample by `stride`
            foreach (var stride in strides)
            {
                dModel *= 2;
                int groups = depthwise ? dModel / 2 : 1;

                layers.Add(new EncoderBlock(
                    outputDim: dModel,
                    stride: stride,
                    groups: groups));
            }

            if (attnWindowSize.HasValue)
            {
                layers.Add(new LocalMHA(dim: dModel, windowSize: attnWindowSize.Value, useRotaryPosEmb: true));
            }

            int finalGroups = depthwise ? dModel : 1;

            layers.Add(new WNConv1d(
                dModel, dModel,
                kernelSize: 7,
                padding: 3,
                groups: finalGroups));

            block = Sequential(layers);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x) => block.forward(x);
    }
}