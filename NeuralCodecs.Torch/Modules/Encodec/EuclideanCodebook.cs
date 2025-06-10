using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Euclidean Codebook for Vector Quantization.
/// Implements a codebook with Euclidean distance metric and supports k-means initialization.
/// </summary>
public class EuclideanCodebook : Module<Tensor, (Tensor quantized, Tensor indices)>
{
    private readonly int _codebookSize;
    private readonly float _decay;
    private readonly int _dimension;
    private readonly float _epsilon;
    private readonly bool _kmeansInit;
    private readonly int _kmeansIters;
    private readonly int _thresholdEmaDeadCode;
    private readonly Parameter cluster_size;
    private readonly Parameter embed;
    private readonly Parameter embed_avg;
    private readonly Parameter inited;

    /// <summary>
    /// Initialize Euclidean Codebook
    /// </summary>
    /// <param name="dimension">Dimension of codebook vectors</param>
    /// <param name="codebookSize">Number of codebook entries</param>
    /// <param name="kmeansInit">Whether to use k-means for initialization</param>
    /// <param name="kmeansIters">Number of k-means iterations if using k-means init</param>
    /// <param name="decay">EMA decay for codebook updates</param>
    /// <param name="epsilon">Small value for numerical stability</param>
    /// <param name="thresholdEmaDeadCode">Threshold for dead code replacement</param>
    public EuclideanCodebook(
        int dimension,
        int codebookSize,
        bool kmeansInit = false,
        int kmeansIters = 10,
        float decay = 0.99f,
        float epsilon = 1e-5f,
        int thresholdEmaDeadCode = 2) : base("EuclideanCodebook")
    {
        ValidateParameters(dimension, codebookSize, kmeansIters, decay, epsilon, thresholdEmaDeadCode);

        _dimension = dimension;
        _codebookSize = codebookSize;
        _kmeansInit = kmeansInit;
        _kmeansIters = kmeansIters;
        _decay = decay;
        _epsilon = epsilon;
        _thresholdEmaDeadCode = thresholdEmaDeadCode;

        var initFn = !kmeansInit ?
            (Func<long[], Tensor>)VQUtils.UniformInit :
            (shape) => zeros(shape);

        embed = Parameter(initFn(new long[] { codebookSize, dimension }));
        cluster_size = Parameter(zeros(codebookSize));
        using (no_grad())
        {
            embed_avg = Parameter(embed.detach().clone());
        }
        inited = Parameter(tensor(new[] { !kmeansInit }, dtype: ScalarType.Float32));

        RegisterComponents();
    }

    /// <summary>
    /// Decode indices to vectors
    /// </summary>
    public Tensor Decode(Tensor embedInd)
    {
        return Dequantize(embedInd);
    }

    /// <summary>
    /// Convert indices to quantized vectors
    /// </summary>
    public Tensor Dequantize(Tensor embedInd)
    {
        return embed.index(embedInd);
    }

    /// <summary>
    /// Encode input to indices
    /// </summary>
    public Tensor Encode(Tensor x)
    {
        using var scope = NewDisposeScope();
        var shape = x.shape;
        x = Preprocess(x);
        var embedInd = Quantize(x);
        return PostprocessEmb(embedInd, shape).MoveToOuterDisposeScope();
    }

    public override (Tensor quantized, Tensor indices) forward(Tensor x)
    {
        using var scope = NewDisposeScope();

        var shape = x.shape;
        var dtype = x.dtype;

        // Flatten input for processing
        var flatX = Preprocess(x);

        // Initialize codebook if needed
        InitEmbed(flatX);

        // Quantize input to get codebook indices
        var embedInd = Quantize(flatX);

        // Create one-hot representation for codebook updates
        var embedOnehot = one_hot(embedInd, _codebookSize).to(dtype);

        // Reshape indices to match input shape
        var reshapedIndices = PostprocessEmb(embedInd, shape);

        // Dequantize indices to get quantized vectors
        var quantize = Dequantize(reshapedIndices);

        // Update codebook during training
        if (training)
        {
            UpdateCodebook(flatX, embedOnehot);
        }

        return (
            quantize.MoveToOuterDisposeScope(),
            reshapedIndices.MoveToOuterDisposeScope()
        );
    }

    /// <summary>
    /// Process embedding indices into the appropriate shape
    /// </summary>
    public Tensor PostprocessEmb(Tensor embedInd, long[] shape)
    {
        return embedInd.view(shape[..^1]);
    }

    /// <summary>
    /// Preprocess input tensor for quantization
    /// </summary>
    public Tensor Preprocess(Tensor x)
    {
        return x.reshape(-1, x.size(-1));
    }

    /// <summary>
    /// Optimized quantization method using efficient matrix operations
    /// </summary>
    public Tensor Quantize(Tensor x)
    {
        using var scope = NewDisposeScope();

        // Reshape for flat computation if necessary
        if (x.dim() == 3)
        {
            x = x.reshape(-1, x.size(-1));
        }

        // Compute squared L2 distances:
        // dist = ||x - e||^2 = ||x||^2 + ||e||^2 - 2<x,e>

        // Compute ||x||^2 term - shape: [N, 1]
        var xNormSquared = x.pow(2).sum(1, keepdim: true);

        // Compute ||e||^2 term - shape: [1, K]
        var embedNormSquared = embed.pow(2).sum(1, keepdim: true).t();

        // Compute -2<x,e> term - shape: [N, K]
        var negDotProduct = -2 * x.matmul(embed.t());

        // Combine terms - shape: [N, K]
        var distances = xNormSquared.add(embedNormSquared).add(negDotProduct);

        // Get indices of minimum distances - shape: [N]
        return distances.argmin(dim: -1).MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            embed?.Dispose();
            cluster_size?.Dispose();
            embed_avg?.Dispose();
            inited?.Dispose();
        }
        base.Dispose(disposing);
    }

    /// <summary>
    /// Apply exponential moving average in-place
    /// </summary>
    private static void EmaInplace(Tensor movingAvg, Tensor newValue, float decay)
    {
        // moving_avg = decay * moving_avg + (1 - decay) * new_value
        movingAvg.mul_(decay).add_(newValue, alpha: 1 - decay);
    }

    /// <summary>
    /// Apply Laplace smoothing to prevent division by zero
    /// </summary>
    private static Tensor LaplaceSmoothing(Tensor x, int nCategories, float epsilon)
    {
        // (x + eps) / (x.sum() + n_categories * eps)
        return x.add(epsilon).div(x.sum() + (nCategories * epsilon));
    }

    private static void ValidateParameters(
                                                int dimension, int codebookSize, int kmeansIters,
        float decay, float epsilon, int thresholdEmaDeadCode)
    {
        if (dimension <= 0)
        {
            throw new ArgumentException($"Dimension must be positive, got {dimension}");
        }

        if (codebookSize <= 0)
        {
            throw new ArgumentException($"Codebook size must be positive, got {codebookSize}");
        }

        if (kmeansIters <= 0)
        {
            throw new ArgumentException($"K-means iterations must be positive, got {kmeansIters}");
        }

        if (decay is <= 0 or >= 1)
        {
            throw new ArgumentException($"Decay must be between 0 and 1, got {decay}");
        }

        if (epsilon <= 0)
        {
            throw new ArgumentException($"Epsilon must be positive, got {epsilon}");
        }

        if (thresholdEmaDeadCode < 0)
        {
            throw new ArgumentException($"Threshold for dead code must be non-negative, got {thresholdEmaDeadCode}");
        }
    }

    /// <summary>
    /// Handle dead codes by replacing them with new vectors
    /// </summary>
    private void ExpireCodes(Tensor batchSamples)
    {
        if (_thresholdEmaDeadCode == 0)
        {
            return;
        }

        using var scope = NewDisposeScope();
        var expiredCodes = cluster_size.lt(_thresholdEmaDeadCode);

        if (!expiredCodes.any().item<bool>())
        {
            return;
        }

        var samples = batchSamples.reshape(-1, batchSamples.size(-1));
        ReplaceCodebook(samples, expiredCodes);
    }

    /// <summary>
    /// Initialize the codebook using k-means if specified
    /// </summary>
    /// <param name="data">Input data for k-means initialization</param>
    private void InitEmbed(Tensor data)
    {
        if (inited.ToBoolean())
        {
            return;
        }

        using var scope = NewDisposeScope();

        var (cEmbed, clusterSize) = VQUtils.KMeans(data, _codebookSize, _kmeansIters);

        embed.copy_(cEmbed);
        embed_avg.copy_(cEmbed.clone());
        cluster_size.copy_(clusterSize);
        inited.copy_(tensor(true));

        // Synchronization would happen here for distributed training
    }

    /// <summary>
    /// Replace codebook entries based on a mask
    /// </summary>
    private void ReplaceCodebook(Tensor samples, Tensor mask)
    {
        using var scope = NewDisposeScope();
        var sampledVectors = VQUtils.SampleVectors(samples, _codebookSize);
        var modifiedCodebook = where(mask.unsqueeze(-1), sampledVectors, embed);
        embed.copy_(modifiedCodebook);
    }

    /// <summary>
    /// Update the codebook using exponential moving averages
    /// </summary>
    private void UpdateCodebook(Tensor flatX, Tensor embedOnehot)
    {
        using var scope = NewDisposeScope();

        // Handle expired/dead codes
        ExpireCodes(flatX);

        // Update cluster sizes - EMA of cluster occupancy
        EmaInplace(cluster_size, embedOnehot.sum(0), _decay);

        // Update embedding averages - EMA of sum(x * one_hot(idx))
        using var embedSum = matmul(embedOnehot.t(), flatX);
        EmaInplace(embed_avg, embedSum, _decay);

        // Normalize embeddings with Laplace smoothing
        var n = cluster_size.sum();
        using var clusterSize = LaplaceSmoothing(cluster_size, _codebookSize, _epsilon).mul(n);
        using var embedNormalized = embed_avg.div(clusterSize.unsqueeze(1));

        // Update codebook
        embed.copy_(embedNormalized);
    }
}