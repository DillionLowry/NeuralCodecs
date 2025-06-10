using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Vector quantizer implementation with Euclidean distance metric.
/// </summary>
public class VectorQuantizer : Module<Tensor, (Tensor quantized, Tensor codes, Tensor loss)>
{
    private readonly float _commitmentWeight;
    private readonly float _epsilon;
    private readonly EuclideanCodebook codebook;
    private readonly Linear projectIn;
    private readonly Linear projectOut;

    public VectorQuantizer(
        int dim,
        int codebookSize,
        int? codebookDim = null,
        float decay = 0.99f,
        float epsilon = 1e-5f,
        bool kmeansInit = true,
        int kmeansIters = 50,
        int thresholdEmaDeadCode = 2,
        float commitmentWeight = 1.0f) : base("VQ")
    {
        // Use input dim as codebook dim if not specified
        var actualCodebookDim = codebookDim ?? dim;
        CodebookSize = codebookSize;

        // Create projection layers if dimensions differ
        var requiresProjection = actualCodebookDim != dim;
        projectIn = requiresProjection ? Linear(dim, actualCodebookDim) : null;
        projectOut = requiresProjection ? Linear(actualCodebookDim, dim) : null;

        _epsilon = epsilon;
        _commitmentWeight = commitmentWeight;

        codebook = new EuclideanCodebook(
            actualCodebookDim,
            codebookSize,
            kmeansInit: kmeansInit,
            kmeansIters: kmeansIters,
            decay: decay,
            epsilon: epsilon,
            thresholdEmaDeadCode: thresholdEmaDeadCode);

        RegisterComponents();
    }

    public int CodebookSize { get; }

    public int Stride { get; }

    public Tensor Decode(Tensor embedInd)
    {
        using var scope = NewDisposeScope();
        var quantize = codebook.Decode(embedInd);
        quantize = projectOut?.forward(quantize) ?? quantize;
        quantize = quantize.transpose(1, 2);  // B N D -> B D N
        return quantize.MoveToOuterDisposeScope();
    }

    public Tensor Encode(Tensor x)
    {
        using var scope = NewDisposeScope();
        x = x.transpose(1, 2); // B D N -> B N D
        x = projectIn?.forward(x) ?? x;
        var embedIn = codebook.Encode(x);
        return embedIn.MoveToOuterDisposeScope();
    }

    public override (Tensor quantized, Tensor codes, Tensor loss) forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var device = x.device;

        // Rearrange to BND
        x = x.transpose(1, 2);
        if (projectIn is not null)
        {
            x = projectIn.forward(x);
        }

        // Get quantized values and codes
        var (quantize, embedInd) = codebook.forward(x);

        if (training)
        {
            // Straight-through gradient estimation
            quantize = x + (quantize - x).detach();
        }

        // Initialize loss tensors
        var loss = tensor(0.0f, device: device, requires_grad: training);

        if (training && _commitmentWeight > 0)
        {
            var commitLoss = mse_loss(quantize.detach(), x);
            loss += commitLoss * _commitmentWeight;
        }

        // Project back to original dimension and rearrange
        quantize = projectOut?.forward(quantize) ?? quantize;
        quantize = quantize.transpose(1, 2);  // BND -> BDN

        return (
            quantize.MoveToOuterDisposeScope(),
            embedInd.MoveToOuterDisposeScope(),
            loss.MoveToOuterDisposeScope()
        );
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            projectIn?.Dispose();
            projectOut?.Dispose();
            codebook?.Dispose();
        }
        base.Dispose(disposing);
    }

    private static void ValidateParameters(
                        int inputDim, int codebookSize, int codebookDim, int stride)
    {
        if (inputDim <= 0)
        {
            throw new ArgumentException($"Input dimension must be positive, got {inputDim}");
        }

        if (codebookSize <= 0)
        {
            throw new ArgumentException($"Codebook size must be positive, got {codebookSize}");
        }

        if (codebookDim <= 0)
        {
            throw new ArgumentException($"Codebook dimension must be positive, got {codebookDim}");
        }

        if (stride <= 0)
        {
            throw new ArgumentException($"Stride must be positive, got {stride}");
        }
    }

    private void ValidateIndices(Tensor indices)
    {
        if (indices.dim() != 2)
        {
            throw new ArgumentException(
                $"Expected 2D indices tensor [B, T], got shape [{string.Join(", ", indices.shape)}]");
        }

        if (indices.max().item<long>() >= CodebookSize)
        {
            throw new ArgumentException(
                $"Indices contain values >= codebook size {CodebookSize}");
        }

        if (indices.min().item<long>() < 0)
        {
            throw new ArgumentException("Indices contain negative values");
        }
    }

    private void ValidateInputShape(Tensor x)
    {
        if (x.dim() != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input tensor [B, C, T], got shape [{string.Join(", ", x.shape)}]");
        }
    }
}