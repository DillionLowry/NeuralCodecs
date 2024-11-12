using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.SNAC;

public partial class SNAC
{
    /// <summary>
    /// Implements a Weight Normalized 1D Convolution layer.
    /// Weight Normalization is a reparameterization of the weights that decouples the magnitude
    /// of those weights from their direction, simulating PyTorch's torch.nn.utils.parametrizations.weight_norm().
    /// </summary>
    public class WNConv1d : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Optional bias parameter
        /// </summary>
        private readonly Parameter bias;

        private readonly ParameterDict parametrizations = [];

        /// <summary>
        /// Stride for the convolution operation
        /// </summary>
        private readonly long stride;

        /// <summary>
        /// Padding applied to input before convolution
        /// </summary>
        private readonly long padding;

        /// <summary>
        /// Dilation factor for the convolution
        /// </summary>
        private readonly long dilation;

        /// <summary>
        /// Number of groups for grouped convolution
        /// </summary>
        private readonly long groups;

        /// <summary>
        /// Initializes a new instance of the WNConv1d class.
        /// </summary>
        /// <param name="inChannels">Number of input channels.</param>
        /// <param name="outChannels">Number of output channels.</param>
        /// <param name="kernelSize">Size of the convolution kernel.</param>
        /// <param name="stride">Stride of the convolution. Default is 1.</param>
        /// <param name="padding">Zero-padding added to both sides of the input. Default is 0.</param>
        /// <param name="dilation">Spacing between kernel elements. Default is 1.</param>
        /// <param name="groups">Number of blocked connections from input channels to output channels. Default is 1.</param>
        /// <param name="useBias">If true, adds a learnable bias to the output. Default is true.</param>
        public WNConv1d(long inChannels, long outChannels, long kernelSize,
            long stride = 1, long padding = 0, long dilation = 1, long groups = 1, bool useBias = true)
            : base($"WNConv1d")
        {
            this.stride = stride;
            this.padding = padding;
            this.dilation = dilation;
            this.groups = groups;

            parametrizations.Add("weight.original0", new Parameter(
                empty(new long[] { 1, outChannels / groups, 1 }, dtype: float32)));

            parametrizations.Add("weight.original1", new Parameter(
                empty(new long[] { outChannels, inChannels / groups, kernelSize }, dtype: float32)));

            if (useBias)
            {
                bias = new Parameter(empty(outChannels, dtype: float32));
            }

            RegisterComponents();
            ResetParameters(useBias);
        }

        /// <summary>
        /// Resets the parameters of the layer, initializing the weights using Kaiming uniform initialization and setting the bias.
        /// </summary>
        /// <param name="useBias">If true, initializes the bias parameter.</param>
        private void ResetParameters(bool useBias)
        {
            using (no_grad())
            {
                var weight = empty_like(parametrizations["weight.original1"]);
                init.kaiming_uniform_(weight, Math.Sqrt(5));

                // Compute norm along dims [1,2] (in_channels and kernel_size) with keepdim
                var norm = weight.contiguous().pow(2).sum(new long[] { 1, 2 }, keepdim: true, ScalarType.Float32).sqrt();

                parametrizations["weight.original0"].set_(norm);
                parametrizations["weight.original1"].set_(weight.div(norm.sub(1e-7f)));

                if (useBias)
                {
                    var fan_in = parametrizations["weight.original1"].size(1) * parametrizations["weight.original1"].size(2);
                    var bound = fan_in > 0 ?
                                1.0 / Math.Sqrt(fan_in) :
                                0.0;
                    init.uniform_(bias, -bound, bound);
                }
            }
        }

        /// <summary>
        /// Performs forward pass applying weight normalized convolution
        /// </summary>
        /// <remarks>
        /// The computed L2 norm of v is almost exactly the same as PyTorch's value, but there is a potential floating
        /// point difference in the least significant digit causing a compounding error of +/-(2^31 - 1)
        /// </remarks>
        /// <param name="input">Input tensor of shape (batch, in_channels, length)</param>
        /// <returns>
        /// Convolved tensor with shape (batch, out_channels, new_length)
        /// </returns>
        public override Tensor forward(Tensor input)
        {
            using var scope = NewDisposeScope();
            var weight_v = parametrizations["weight.original1"];
            var weight_g = parametrizations["weight.original0"];

            // Compute L2 norm of v along [1,2] dimensions
            // The floating point difference is here, this is the closest I could get to the final memory layout and
            // operations of the native Torch calls. The operations are fused in the native code and the order of operations
            // is important. The max cumulative error over 10000 iterations should be within +/- 1e-4
            var v_norm = weight_v.contiguous().pow(2).sum(new long[] { 1, 2 }, keepdim: true, ScalarType.Float32).sqrt();
            var weight = mul(weight_v.div(v_norm), weight_g.sub(1e-7f)).contiguous();

            return functional.conv1d(
                input,
                weight,
                bias,
                stride: stride,
                padding: padding,
                dilation: dilation,
                groups: groups).MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                parametrizations["weight.original0"]?.Dispose();
                parametrizations["weight.original1"]?.Dispose();
                bias?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}