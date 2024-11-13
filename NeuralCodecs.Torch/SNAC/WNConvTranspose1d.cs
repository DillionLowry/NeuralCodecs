using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public partial class SNAC
{
    /// <summary>
    /// Implements a Weight Normalized 1D Transposed Convolution layer.
    /// Weight Normalization is a reparameterization of the weights that decouples the magnitude
    /// of those weights from their direction, simulating PyTorch's torch.nn.utils.parametrizations.weight_norm().
    /// </summary>
    public class WNConvTranspose1d : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Optional bias parameter
        /// </summary>
        private readonly Parameter bias;

        /// <summary>
        /// Weight Normalized parameters for convolution
        /// </summary>
        private readonly ParameterDict parametrizations = [];

        /// <summary>
        /// Stride for the convolution operation
        /// </summary>
        private readonly long stride;

        /// <summary>
        /// Padding applied to input before convolution
        /// </summary>
        private readonly long padding;

        private readonly long outputPadding;

        /// <summary>
        /// Dilation factor for the convolution
        /// </summary>
        private readonly long dilation;

        /// <summary>
        /// Number of groups for grouped convolution
        /// </summary>
        private readonly long groups;

        /// <summary>
        /// Initializes a new instance of the WNConvTranspose1d module.
        /// </summary>
        /// <param name="inChannels">Number of channels in the input tensor</param>
        /// <param name="outChannels">Number of channels produced by the convolution</param>
        /// <param name="kernelSize">Size of the convolving kernel</param>
        /// <param name="stride">Stride of the convolution (default: 1)</param>
        /// <param name="padding">Padding added to input (default: 0)</param>
        /// <param name="outputPadding">Additional size added to output shape (default: 0)</param>
        /// <param name="dilation">Spacing between kernel elements (default: 1)</param>
        /// <param name="groups">Number of blocked connections from input to output channels (default: 1)</param>
        /// <param name="useBias">If true, adds a learnable bias to the output (default: true)</param>
        public WNConvTranspose1d(long inChannels, long outChannels, long kernelSize,
            long stride = 1, long padding = 0, long outputPadding = 0,
            long dilation = 1, long groups = 1, bool useBias = true)
            : base($"WNConvTranspose1d")
        {
            this.padding = padding;
            this.groups = groups;

            parametrizations.Add("weight.original0", new Parameter(
                empty(new long[] { 1, outChannels / groups, 1 }, dtype: float32)));

            parametrizations.Add("weight.original1", new Parameter(
                empty(new long[] { inChannels, outChannels / groups, kernelSize }, dtype: float32)));

            if (useBias)
            {
                bias = Parameter(empty(outChannels, dtype: float32));
            }

            this.stride = stride;
            this.outputPadding = outputPadding;
            this.dilation = dilation;
            RegisterComponents();
            ResetParameters(useBias);
        }

        /// <summary>
        /// Resets the parameters of the layer using Kaiming initialization for weights
        /// and uniform initialization for bias if enabled.
        /// </summary>
        /// <param name="useBias">Whether to initialize bias parameter</param>
        public void ResetParameters(bool useBias)
        {
            using (no_grad())
            {
                // Initialize weight_v using Kaiming initialization
                var weight = empty_like(parametrizations["weight.original1"]);
                init.kaiming_uniform_(weight, Math.Sqrt(5));

                // Compute norm along dims [1,2] (in_channels and kernel_size) with keepdim
                var norm = sqrt(weight.pow(2).sum(new long[] { 1, 2 }, keepdim: true));

                // Set weight_g and weight_v
                parametrizations["weight.original0"].set_(norm);
                parametrizations["weight.original1"].set_(weight.div(norm.add(1e-7f)));

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
        /// Performs forward pass applying weight normalized transposed convolution
        /// </summary>
        /// <param name="input">Input tensor of shape (batch, in_channels, length)</param>
        /// <returns>
        /// Upsampled tensor with shape (batch, out_channels, new_length)
        /// where new_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        /// </returns>
        public override Tensor forward(Tensor input)
        {
            var v = parametrizations["weight.original1"];
            var g = parametrizations["weight.original0"];

            var v_norm = v.contiguous().pow(2).sum(new long[] { 1, 2 }, keepdim: true, ScalarType.Float32).sqrt();
            var weight = mul(v.div(v_norm), g.sub(1e-7f)).contiguous();

            return functional.conv_transpose1d(
                input,
                weight,
                bias,
                stride: stride,
                padding: padding,
                output_padding: outputPadding,
                groups: groups,
                dilation: dilation);
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