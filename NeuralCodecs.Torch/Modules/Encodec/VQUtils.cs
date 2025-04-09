using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

public static class VQUtils
{
    /// <summary>
    /// Calculate the exponential moving average of a tensor.
    /// </summary>
    /// <param name="movingAvg"></param>
    /// <param name="newValue"></param>
    /// <param name="decay"></param>
    public static void EmaInplace(Tensor movingAvg, Tensor newValue, float decay)
    {
        movingAvg.mul_(decay).add_(newValue, alpha: 1 - decay);
    }

    /// <summary>
    /// Perform k-means clustering on samples.
    /// </summary>
    /// <param name="samples">Tensor of shape [num_samples, feature_dim]</param>
    /// <param name="numClusters">Number of clusters (k)</param>
    /// <param name="numIters">Number of iterations</param>
    /// <returns>Tuple of (centroids, cluster_sizes)</returns>
    public static (Tensor centroids, Tensor clusterSizes) KMeans(
        Tensor samples, int numClusters, int numIters = 10)
    {
        using var scope = NewDisposeScope();

        // Initialize centroids by sampling from the data
        var means = SampleVectors(samples, numClusters);
        var dtype = samples.dtype;
        var device = samples.device;

        for (int iter = 0; iter < numIters; iter++)
        {
            using var iterScope = NewDisposeScope();

            // Calculate distances between samples and centroids
            using var diffs = samples.unsqueeze(1).sub(means.unsqueeze(0));
            using var squaredDists = diffs.pow(2).sum(-1);
            using var distances = squaredDists.neg();

            // Assign samples to nearest centroid
            using var buckets = distances.argmax(dim: -1);

            // Count assignments per cluster
            using var bins = bincount(buckets, minlength: numClusters);
            using var zeroMask = bins.eq(0);
            using var binsMinClamped = bins.masked_fill(zeroMask, 1);

            // Calculate new centroids using vectorized operations
            using var newMeans = zeros(numClusters, samples.size(-1), dtype: dtype, device: device);
            using var expandedBuckets = buckets.unsqueeze(-1).expand(-1, samples.size(-1));

            // Efficiently sum all samples for each cluster
            newMeans.scatter_add_(0, expandedBuckets, samples);

            // Divide by count to get mean
            newMeans.div_(binsMinClamped.unsqueeze(-1));

            // Update centroids, keeping old ones for empty clusters
            means = where(zeroMask.unsqueeze(-1), means, newMeans);
        }

        // Calculate final assignments and cluster sizes
        using var diffsFinal = samples.unsqueeze(1).sub(means.unsqueeze(0));
        using var distsFinal = diffsFinal.pow(2).sum(-1).neg();
        var bucketsFinal = distsFinal.argmax(dim: -1);
        var binsFinal = bincount(bucketsFinal, minlength: numClusters);

        return (means.MoveToOuterDisposeScope(), binsFinal.MoveToOuterDisposeScope());
    }

    /// <summary>
    /// Calculate the exponential moving average of a tensor.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="nCategories"></param>
    /// <param name="epsilon"></param>
    /// <returns></returns>
    public static Tensor LaplaceSmoothing(Tensor x, int nCategories, float epsilon = 1e-5f)
    {
        return (x + epsilon) / (x.sum() + (nCategories * epsilon));
    }

    /// <summary>
    /// Sample vectors from a tensor of samples.
    /// </summary>
    /// <param name="samples">Tensor of shape [num_samples, feature_dim]</param>
    /// <param name="num">Number of vectors to sample</param>
    /// <returns>Tensor of shape [num, feature_dim]</returns>
    public static Tensor SampleVectors(Tensor samples, int num)
    {
        using var scope = NewDisposeScope();
        var numSamples = samples.size(0);
        var device = samples.device;

        if (numSamples >= num)
        {
            // Sample without replacement if we have enough samples
            var indices = randperm(numSamples, device: device)[..num];
            return samples.index_select(0, indices).MoveToOuterDisposeScope();
        }
        else
        {
            // Sample with replacement if we don't have enough
            var indices = randint(0, numSamples, num, device: device);
            return samples.index_select(0, indices).MoveToOuterDisposeScope();
        }
    }

    /// <summary>
    /// Initialize weights with uniform distribution
    /// </summary>
    public static Tensor UniformInit(long[] shape)
    {
        var t = empty(shape);
        init.kaiming_uniform_(t);
        return t;
    }
}