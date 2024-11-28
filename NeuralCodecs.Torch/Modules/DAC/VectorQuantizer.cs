using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace NeuralCodecs.Torch.Modules.DAC;

public class VectorQuantizer : Module<Tensor, (Tensor quantized, Tensor commitmentLoss, Tensor codebookLoss, Tensor indices, Tensor latents)>
{
    private readonly int codebookSize;
    private readonly int codebookDim;

    //private readonly bool quantizerDropout;
    private readonly Linear _projection;

    private readonly Embedding codebook;
    private readonly float _beta = 0.25f;
    private readonly WNConv1d in_proj;
    private readonly WNConv1d out_proj;
    // TODO: OPTIONAL LOSS

    public VectorQuantizer(int inputDim, int codebookSize, int codebookDim)
        : base($"VQ_{codebookSize}_{codebookDim}")
    {
        codebookSize = codebookSize;
        codebookDim = codebookDim;

        in_proj = new WNConv1d(inputDim, codebookDim, kernelSize: 1);
        out_proj = new WNConv1d(codebookDim, inputDim, kernelSize: 1);
        _projection = Linear(inputDim, codebookDim);
        codebook = Embedding(codebookSize, codebookDim);

        RegisterComponents();
    }

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
            .mean(new long[] { 1, 2 });
        var codebookLoss = F.mse_loss(zQ, zE.detach(), reduction: Reduction.None)
            .mean(new long[] { 1, 2 });

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

        // TODO: test euclidean distance
        //var encodingsSquared = encodings.pow(2).sum(1, keepdim: true);
        //var codebookSquared = codebookWeight.pow(2).sum(1, keepdim: true);
        //var dist = encodingsSquared - 2 * torch.mm(encodings, codebookWeight.t())
        //          + codebookSquared.t();

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

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            in_proj?.Dispose();
            out_proj?.Dispose();
            codebook?.Dispose();
        }
        base.Dispose(disposing);
    }
    //public override (Tensor quantized, Tensor codes, Tensor latents, Tensor commitmentLoss, Tensor codebookLoss)
    //    forward(Tensor z)
    //{
    //    using var scope = NewDisposeScope();

    //    // Project input to codebook dimension
    //    var shape = z.shape;
    //    var flatZ = z.transpose(-2, -1).reshape(-1, shape[1]);
    //    var latents = projection.forward(flatZ);

    //    // Calculate distances to codebook entries
    //    var flatLatents = latents.unsqueeze(1);
    //    var codebookSquared = codebook.weight.pow(2).sum(-1, keepdim: true);
    //    var latentsSquared = flatLatents.pow(2).sum(-1, keepdim: true);

    //    var distances = latentsSquared + codebookSquared.transpose(-2, -1)
    //        - 2 * torch.matmul(flatLatents, codebook.weight.transpose(-2, -1));

    //    // Get nearest codebook entries
    //    var indices = distances.argmin(-1);
    //    var quantized = codebook.forward(indices);

    //    // Reshape back to input shape
    //    quantized = quantized.view(shape[0], -1, shape[1]).transpose(-2, -1);

    //    // Calculate losses
    //    var commitmentLoss = torch.nn.functional.mse_loss(quantized, z.detach());
    //    var codebookLoss = torch.nn.functional.mse_loss(quantized.detach(), z);

    //    // Straight-through estimator
    //    quantized = z + (quantized - z).detach();

    //    if (quantizerDropout && this.training)
    //    {
    //        var mask = torch.bernoulli(torch.full_like(quantized, 0.1f));
    //        quantized = quantized * (1 - mask);
    //    }

    //    return (
    //        quantized.MoveToOuterDisposeScope(),
    //        indices.MoveToOuterDisposeScope(),
    //        latents.MoveToOuterDisposeScope(),
    //        commitmentLoss.MoveToOuterDisposeScope(),
    //        codebookLoss.MoveToOuterDisposeScope()
    //    );
    //}
}