using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Codecs.SNAC;

public partial class SNAC
{
    /// <summary>
    /// Implements Vector Quantization (VQ) for neural audio coding.
    /// Maps continuous latent vectors to discrete codes from a learned codebook.
    /// </summary>
    public class VectorQuantizer : Module<Tensor, (Tensor output, Tensor indices)>
    {
        /// <summary>
        /// Number of entries in the codebook
        /// </summary>
        private readonly int codebookSize;

        /// <summary>
        /// Dimension of each codebook entry
        /// </summary>
        private readonly int codebookDim;

        /// <summary>
        /// Stride factor for temporal downsampling/upsampling
        /// </summary>
        public readonly int stride;

        /// <summary>
        /// Projects input to codebook dimension space
        /// </summary>
        private readonly WNConv1d in_proj;

        /// <summary>
        /// Projects quantized vectors back to input dimension space
        /// </summary>
        private readonly WNConv1d out_proj;

        /// <summary>
        /// Learnable codebook of embedding vectors
        /// </summary>
        private readonly Embedding codebook;

        /// <summary>
        /// Initializes a new Vector Quantization layer.
        /// </summary>
        /// <param name="inputDim">Dimension of input features</param>
        /// <param name="codebookSize">Number of discrete codes in the codebook (N)</param>
        /// <param name="codebookDim">Dimension of each codebook vector (D)</param>
        /// <param name="stride">Temporal stride for optional downsampling/upsampling</param>
        public VectorQuantizer(int inputDim, int codebookSize, int codebookDim, int stride = 1)
            : base($"VQ_{codebookSize}_{codebookDim}")
        {
            this.codebookSize = codebookSize;
            this.codebookDim = codebookDim;
            this.stride = stride;

            in_proj = new WNConv1d(inputDim, codebookDim, kernelSize: 1);
            out_proj = new WNConv1d(codebookDim, inputDim, kernelSize: 1);
            codebook = Embedding(codebookSize, codebookDim, dtype: float32);
            RegisterComponents();
        }

        /// <summary>
        /// Projects input tensor through input projection layer
        /// </summary>
        public Tensor InProj(Tensor x) => in_proj.forward(x);

        /// <summary>
        /// Projects tensor through output projection layer
        /// </summary>
        public Tensor OutProj(Tensor x) => out_proj.forward(x);

        /// <summary>
        /// Performs forward pass of vector quantization
        /// </summary>
        /// <param name="z">Input tensor of shape (batch, channels, time)</param>
        /// <returns>
        /// Tuple containing:
        /// - Quantized tensor with straight-through gradient
        /// - Indices of selected codebook entries
        /// </returns>
        public override (Tensor output, Tensor indices) forward(Tensor z)
        {
            using var scope = NewDisposeScope();

            if (stride > 1) // Downsample if stride > 1
            {
                z = functional.avg_pool1d(z, kernel_size: stride, stride: stride).to(float32);
            }

            var zE = in_proj.forward(z).to(float32);

            var (zQ, indices) = DecodeLatents(zE);

            zQ = zE + (zQ - zE).detach(); // Straight-through gradient estimator
            zQ = out_proj.forward(zQ);
            if (stride > 1) // Upsample if needed
            {
                zQ = zQ.repeat_interleave(stride, dim: -1);
            }
            return (zQ.MoveToOuterDisposeScope(), indices.MoveToOuterDisposeScope());
        }

        /// <summary>
        /// Maps continuous latent vectors to their nearest codebook entries.
        /// Uses efficient L2 distance computation via matrix multiplication.
        /// </summary>
        /// <param name="latents">Input continuous vectors to be quantized</param>
        /// <returns>
        /// Tuple containing:
        /// - Quantized vectors from the codebook
        /// - Indices of the selected codebook entries
        /// </returns>
        private (Tensor zQ, Tensor indices) DecodeLatents(Tensor latents)
        {
            using var scope = NewDisposeScope();

            // Reshape to (batch * time, dim)
            var shape = latents.shape;
            var encodings = latents.transpose(1, 2).reshape(-1, codebookDim).to(float32).contiguous();

            var codebookWeight = codebook.weight?.to(float32).contiguous() ?? throw new InvalidOperationException("Codebook weight is null");

            // L2 normalize both encodings and codebook
            var encodingsSquared = encodings.pow(2).sum(1, keepdim: true);
            var codebookSquared = codebookWeight.pow(2).sum(1, keepdim: true);

            // Compute cross terms with einsum
            var crossTerms = einsum("bd,nd->bn", encodings, codebookWeight);
            crossTerms = crossTerms.mul_(2.0f);

            // Compute distances
            var dist = encodingsSquared + codebookSquared.t() - crossTerms;

            // Get nearest codebook entries
            var indices = dist.argmin(1).reshape(shape[0], shape[2]);

            var zQ = DecodeCode(indices);
            return (zQ.MoveToOuterDisposeScope(), indices.MoveToOuterDisposeScope());
        }

        /// <summary>
        /// Converts codebook indices back into continuous vectors using the learned codebook embeddings.
        /// </summary>
        /// <param name="indices">Tensor of codebook indices with shape (batch, time)</param>
        /// <returns>
        /// Tensor of continuous vectors with shape (batch, codebook_dim, time)
        /// representing the corresponding codebook entries
        /// </returns
        public Tensor DecodeCode(Tensor indices)
        {
            using var scope = NewDisposeScope();
            // Look up embeddings and reshape
            var embeddings = codebook.forward(indices).to(float32).contiguous();
            return embeddings.transpose(1, 2).contiguous().MoveToOuterDisposeScope();
        }
    }
}