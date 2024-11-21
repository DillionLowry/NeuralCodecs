using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public partial class SNAC
{
    /// <summary>
    /// Implements Residual Vector Quantization (RVQ) for neural compression.
    /// RVQ uses multiple codebooks to progressively quantize input vectors,
    /// with each stage operating on the residual from previous stages.
    /// </summary>
    public class ResidualVectorQuantizer : Module<Tensor, (Tensor zQ, List<Tensor> codes)>
    {
        /// <summary>
        /// Number of codebooks used in the quantization process
        /// </summary>
        private readonly int nCodebooks;

        /// <summary>
        /// Dimension of each codebook entry
        /// </summary>
        private readonly int codebookDim;

        /// <summary>
        /// Size of each codebook (number of entries)
        /// </summary>
        private readonly int codebookSize;

        /// <summary>
        /// List of vector quantizers, one for each codebook
        /// </summary>
        private readonly ModuleList<VectorQuantizer> quantizers;

        /// <summary>
        /// Initializes a new instance of ResidualVectorQuantize
        /// </summary>
        /// <param name="inputDim">Dimension of input vectors</param>
        /// <param name="codebookSize">Number of entries in each codebook</param>
        /// <param name="codebookDim">Dimension of codebook entries</param>
        /// <param name="vqStrides">Strides for each vector quantizer</param>
        public ResidualVectorQuantizer(
            int inputDim = 512,
            int codebookSize = 1024,
            int codebookDim = 8,
            int[] vqStrides = null) : base("RVQ")
        {
            vqStrides ??= new[] { 1, 1, 1, 1 };

            nCodebooks = vqStrides.Length;
            this.codebookDim = codebookDim;
            this.codebookSize = codebookSize;

            // Python: self.quantizers = nn.ModuleList([
            //     VectorQuantize(input_dim, codebook_size, codebook_dim, stride)
            //     for stride in vq_strides
            // ])
            VectorQuantizer[] quantizerList = vqStrides.Select(stride =>
                new VectorQuantizer(inputDim, codebookSize, codebookDim, stride)).ToArray();
            quantizers = ModuleList(quantizerList);
            RegisterComponents();
        }

        /// <summary>
        /// Performs forward pass of RVQ encoding
        /// </summary>
        /// <param name="z">Input tensor to quantize</param>
        /// <returns>
        /// Tuple containing:
        /// - zQ: Quantized tensor
        /// - codes: List of codebook indices for each quantization stage
        /// </returns>
        public override (Tensor zQ, List<Tensor> codes) forward(Tensor z)
        {
            using var scope = NewDisposeScope();
            z = z.to(float32).contiguous();
            var zQ = zeros_like(z, dtype: z.dtype, device: z.device);
            var residual = z.clone();
            var codes = new List<Tensor>();

            // Apply each quantizer to the residual
            for (int i = 0; i < quantizers.Count; i++)
            {
                var (zQi, indicesI) = quantizers[i].forward(residual);

                // Add quantized values and update residual
                zQ = add(zQ, zQi, alpha: 1.0f);
                residual = sub(residual, zQi);

                codes.Add(indicesI.clone().MoveToOuterDisposeScope());
            }

            return (zQ.MoveToOuterDisposeScope(), codes);
        }

        /// <summary>
        /// Reconstructs tensor from list of codebook indices
        /// </summary>
        /// <param name="codes">List of codebook indices from encoding</param>
        /// <returns>Reconstructed tensor</returns>
        /// <exception cref="ArgumentException">
        /// Thrown when number of provided codes doesn't match number of codebooks
        /// </exception>
        public Tensor FromCodes(List<Tensor> codes)
        {
            using var scope = NewDisposeScope();
            if (codes.Count != nCodebooks)
                throw new ArgumentException($"Expected {nCodebooks} codebooks but got {codes.Count}");

            var zQ = zeros(1, device: codes[0].device, dtype: codes[0].dtype);
            bool first = true;

            for (int i = 0; i < nCodebooks; i++)
            {
                // Decode codes to embeddings
                var zPi = quantizers[i].DecodeCode(codes[i]);

                // Project to output space
                var zQi = quantizers[i].OutProj(zPi);

                // Apply stride if needed
                if (quantizers[i].stride > 1)
                {
                    zQi = zQi.repeat_interleave(quantizers[i].stride, dim: -1);
                }

                if (first)
                {
                    zQ = zQi;
                    first = false;
                }
                else
                {
                    zQ = add(zQ, zQi, alpha: 1.0f);
                }
            }

            return zQ.MoveToOuterDisposeScope();
        }
    }
}