using NeuralCodecs.Torch.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.SNAC;

public partial class SNAC
{
    /// <summary>
    /// Implements the Snake activation function for 1D inputs.
    /// Snake is a learned periodic activation function that generalizes both ReLU and Sine.
    /// </summary>
    /// <remarks>
    /// The transform applies the function:
    /// x + sin²(αx)/(α + ε)
    /// where α is a learnable parameter and ε is a small constant for numerical stability
    /// </remarks>
    public class Snake1d : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Learnable parameter that controls the frequency of the periodic component
        /// </summary>
        private readonly Parameter alpha;

        /// <summary>
        /// Small constant to prevent numerical instability
        /// </summary>
        private const float EPSILON = 1e-9f;

        /// <summary>
        /// Flag indicating GPU availability for optimized computation paths
        /// </summary>
        private static readonly bool UseGPU = cuda.is_available();

        /// <summary>
        /// Initializes Snake activation with learnable parameters
        /// </summary>
        /// <param name="channels">Number of input channels</param>
        public Snake1d(long channels) : base("Snake1d")
        {
            alpha = Parameter(ones(1, channels, 1, dtype: float32));
            RegisterComponents();
        }

        /// <summary>
        /// Performs an optimized sine calculation with precision and synchronization controls
        /// based on whether GPU or CPU computation is being used.
        /// </summary>
        /// <param name="x">Input tensor to compute sine of</param>
        /// <returns>
        /// Sine of input tensor with controlled precision and synchronization
        /// </returns>
        private static Tensor OptimizedSin(Tensor x)
        {
            if (UseGPU)
            {
                // On GPU, force stream synchronization for consistent results
                cuda.synchronize();

                // Ensure computation happens in a single kernel if possible
                //using var _ = torch.cuda.
                return sin(x).to(float32, non_blocking: false);
            }
            else
            {
                // On CPU, try to match PyTorch's precision pattern
                var result = sin(x);

                // Force exact float32 precision
                result = result.to(float32);

                // Optional: Add memory fence to ensure operation order
                cuda.synchronize();

                return result;
            }
        }

        /// <summary>
        /// Performs forward pass of Snake activation: x + (1/α) * sin²(αx)
        /// </summary>
        /// <param name="x">Input tensor of shape (batch, channels, time)</param>
        /// <returns>Activated tensor of same shape</returns>
        public override Tensor forward(Tensor x)
        {
            using var scope = NewDisposeScope();

            var shape = x.shape;
            var reshaped = x.reshape(shape[0], shape[1], -1);

            // Follow exact torch graph operation order
            var alpha_mul = alpha.mul(reshaped);
            var sin_result = OptimizedSin(alpha_mul);
            var powered = sin_result.pow(2);

            var alpha_eps = alpha.add(EPSILON);
            var reciprocal = alpha_eps.reciprocal();
            var mul_result = reciprocal.mul(powered);

            var added = reshaped.add(mul_result, alpha: 1.0f);

            return added.reshape(shape).MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                alpha?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}