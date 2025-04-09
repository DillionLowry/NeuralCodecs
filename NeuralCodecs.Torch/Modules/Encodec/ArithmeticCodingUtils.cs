using static TorchSharp.torch;

/// <summary>
/// Utilities for arithmetic coding
/// </summary>
public static class ArithmeticCodingUtils
{
    /// <summary>
    /// Build a stable quantized CDF for arithmetic coding
    /// </summary>
    /// <param name="pdf">Probability distribution function</param>
    /// <param name="totalRangeBits">Total range bits for arithmetic coder</param>
    /// <param name="roundoff">Rounding value for probabilities (default: 1e-8)</param>
    /// <param name="minRange">Minimum range width (default: 2)</param>
    /// <param name="check">Whether to check for invalid conditions (default: true)</param>
    /// <returns>Quantized CDF tensor</returns>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
    public static Tensor BuildStableQuantizedCdf(
        Tensor pdf,
        int totalRangeBits,
        float roundoff = 1e-8f,
        int minRange = 2,
        bool check = true)
    {
        // Validate parameters
        if (pdf is null || pdf.IsInvalid)
        {
            throw new ArgumentNullException(nameof(pdf), "PDF tensor cannot be null or invalid");
        }

        if (totalRangeBits <= 0)
        {
            throw new ArgumentException("Total range bits must be positive", nameof(totalRangeBits));
        }

        if (minRange < 2)
        {
            throw new ArgumentException("Minimum range must be at least 2 for numerical stability",
                                        nameof(minRange));
        }

        // Detach PDF from computation graph and apply roundoff
        using var scope = NewDisposeScope();
        using var pdfDetached = pdf.detach();
        var workingPdf = pdfDetached;

        // Apply roundoff if specified
        if (roundoff > 0)
        {
            workingPdf = workingPdf.div(roundoff).floor().mul(roundoff);
        }

        // Calculate parameters for CDF construction
        var totalRange = 1L << totalRangeBits;
        var cardinality = workingPdf.shape[0];
        var alpha = (float)(minRange * cardinality / (double)totalRange);

        // Validate parameters
        if (alpha > 1)
        {
            throw new ArgumentException(
                $"Alpha ({alpha}) > 1. Reduce minRange or increase totalRangeBits",
                nameof(minRange));
        }

        // Calculate ranges for each symbol
        using var scaledPdf = workingPdf.mul((1.0f - alpha) * totalRange);
        var ranges = scaledPdf.floor().to(ScalarType.Int64);
        ranges += minRange;

        // Create CDF by cumulative sum
        var quantizedCdf = ranges.cumsum(0);

        // Validate the CDF if checking is enabled
        if (check)
        {
            if (quantizedCdf[-1].item<long>() > totalRange)
            {
                throw new ArgumentException(
                    $"CDF total ({quantizedCdf[-1].item<long>()}) exceeds range ({totalRange})");
            }

            // Check that each range has at least the minimum width
            bool hasSmallRange = false;
            if (quantizedCdf.shape[0] > 1)
            {
                using var diffs = quantizedCdf[1..].sub(quantizedCdf[..^1]);
                hasSmallRange = diffs.lt(minRange).any().item<bool>();
            }

            bool firstTooSmall = quantizedCdf[0].item<long>() < minRange;

            if (hasSmallRange || firstTooSmall)
            {
                throw new ArgumentException(
                    "Ranges too small. Increase totalRangeBits or decrease minRange");
            }
        }

        return quantizedCdf.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Apply exponential moving average in-place: moving_avg = decay * moving_avg + (1 - decay) * new_value
    /// </summary>
    /// <param name="movingAvg">Moving average tensor to update in-place</param>
    /// <param name="newValue">New value to incorporate</param>
    /// <param name="decay">Decay rate (between 0 and 1)</param>
    public static void EmaInplace(Tensor movingAvg, Tensor newValue, float decay)
    {
        if (movingAvg is null || movingAvg.IsInvalid)
        {
            throw new ArgumentNullException(nameof(movingAvg), "Moving average tensor cannot be null");
        }

        if (newValue is null || newValue.IsInvalid)
        {
            throw new ArgumentNullException(nameof(newValue), "New value tensor cannot be null");
        }

        if (decay is <= 0 or >= 1)
        {
            throw new ArgumentException($"Decay must be between 0 and 1, got {decay}", nameof(decay));
        }

        // Compute in-place: moving_avg = decay * moving_avg + (1 - decay) * new_value
        movingAvg.mul_(decay).add_(newValue, alpha: 1 - decay);
    }

    /// <summary>
    /// Apply Laplace smoothing to a distribution
    /// </summary>
    /// <param name="x">Input distribution</param>
    /// <param name="nCategories">Number of categories</param>
    /// <param name="epsilon">Smoothing parameter</param>
    /// <returns>Smoothed distribution</returns>
    public static Tensor LaplaceSmoothing(Tensor x, int nCategories, float epsilon = 1e-5f)
    {
        if (x is null || x.IsInvalid)
        {
            throw new ArgumentNullException(nameof(x), "Input tensor cannot be null");
        }

        if (nCategories <= 0)
        {
            throw new ArgumentException("Number of categories must be positive", nameof(nCategories));
        }

        if (epsilon <= 0)
        {
            throw new ArgumentException("Epsilon must be positive", nameof(epsilon));
        }

        return x.add(epsilon).div(x.sum() + (nCategories * epsilon));
    }
}