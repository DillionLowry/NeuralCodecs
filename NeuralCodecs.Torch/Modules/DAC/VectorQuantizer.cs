using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace NeuralCodecs.Torch.Modules.DAC;

public class VectorQuantizer : Module<Tensor, (Tensor quantized, Tensor commitmentLoss, Tensor codebookLoss, Tensor indices, Tensor latents)>
{
    /// <summary>
    /// Number of entries in the codebook
    /// </summary>
    private readonly int _codebookSize;

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

    private bool _disposed;

    /// <summary>
    /// Gets the dimension of the codebook.
    /// </summary>
    public int CodebookDim { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="VectorQuantizer"/> class.
    /// </summary>
    /// <param name="inputDim">The input dimension.</param>
    /// <param name="codebookSize">The number of entries in the codebook.</param>
    /// <param name="codebookDim">The dimension of the codebook.</param>
    public VectorQuantizer(int inputDim, int codebookSize, int codebookDim)
        : base($"VQ_{codebookSize}_{codebookDim}")
    {
        _codebookSize = codebookSize;
        CodebookDim = codebookDim;

        in_proj = new WNConv1d(inputDim, codebookDim, kernelSize: 1);
        out_proj = new WNConv1d(codebookDim, inputDim, kernelSize: 1);
        codebook = Embedding(codebookSize, codebookDim);

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

    public override (Tensor quantized, Tensor commitmentLoss, Tensor codebookLoss, Tensor indices, Tensor latents)
      forward(Tensor z)
    {
        using var scope = NewDisposeScope();

        // Project input into low-dimensional space
        var zE = in_proj.forward(z).to(float32);

        // Decode latents to get quantized vectors and indices
        var (zQ, indices) = DecodeLatents(zE);

        // Calculate losses
        var commitmentLoss = F.mse_loss(zE, zQ.detach(), reduction: Reduction.None)
            .mean([1, 2]);
        var codebookLoss = F.mse_loss(zQ, zE.detach(), reduction: Reduction.None)
            .mean([1, 2]);

        zQ = zE + (zQ - zE).detach(); // Straight-through gradient estimator
        zQ = out_proj.forward(zQ);

        return (
            zQ.MoveToOuterDisposeScope(),
            commitmentLoss.MoveToOuterDisposeScope(),
            codebookLoss.MoveToOuterDisposeScope(),
            indices.MoveToOuterDisposeScope(),
            zE.MoveToOuterDisposeScope()
        );
    }

    /// <summary>
    /// Maps continuous latent vectors to their nearest codebook entries.
    /// Uses efficient L2 distance computation via matrix multiplication.
    /// </summary>
    /// <param name="latents">Input continuous vectors to be quantized</param>
    /// <returns>Tuple of quantized vectors and codebook indices</returns>
    public (Tensor zQ, Tensor indices) DecodeLatents(Tensor latents)
    {
        using var scope = NewDisposeScope();

        // Reshape to (batch * time, dim)
        var shape = latents.shape;
        var encodings = latents.transpose(1, 2).reshape(-1, CodebookDim).to(float32).contiguous();

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
        var indices = dist.argmin(1).reshape(shape[0], shape[^1]).to(int64);

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
    /// </returns>
    public Tensor DecodeCode(Tensor indices)
    {
        using var scope = NewDisposeScope();
        // Look up embeddings and reshape
        var embeddings = codebook.forward(indices).contiguous();
        return embeddings.transpose(-2, -1).contiguous().MoveToOuterDisposeScope();
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                in_proj?.Dispose();
                out_proj?.Dispose();
                codebook?.Dispose();
            }
            base.Dispose(disposing);
            _disposed = true;
        }
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}