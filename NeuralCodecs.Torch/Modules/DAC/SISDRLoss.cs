using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Computes Scale-Invariant Source-to-Distortion Ratio for both AudioSignal and raw tensors.
/// Handles batched inputs and supports different reduction modes.
/// </summary>
public class SISDRLoss : AudioLossBase
{
    private readonly bool _scaling;
    private readonly string _reduction;
    private readonly bool _zeroMean;
    private readonly float? _clipMin;
    private readonly float _weight;
    private const float EPS = 1e-8f;

    /// <summary>
    /// Initializes a new instance of SISDRLoss
    /// </summary>
    /// <param name="scaling">Whether to use scale-invariant (true) or signal-to-noise ratio (false)</param>
    /// <param name="reduction">How to reduce across batch ('mean', 'sum', or 'none')</param>
    /// <param name="zeroMean">Zero mean the references and estimates before computing loss</param>
    /// <param name="clipMin">Minimum possible loss value</param>
    /// <param name="weight">Weight of this loss</param>
    /// <param name="sampleRate">Sample rate of input audio (only used for raw tensors)</param>
    public SISDRLoss(
        bool scaling = true,
        string reduction = "mean",
        bool zeroMean = true,
        float? clipMin = null,
        float weight = 1.0f,
        int sampleRate = 44100) : base(nameof(SISDRLoss), sampleRate)
    {
        _scaling = scaling;
        _reduction = reduction;
        _zeroMean = zeroMean;
        _clipMin = clipMin;
        _weight = weight;
        RegisterComponents();
    }

    /// <summary>
    /// Normalizes input shape to (batch, channels, time)
    /// </summary>
    private Tensor NormalizeShape(Tensor x)
    {
        if (x.dim() == 1)
        {
            // Single channel, no batch
            return x.unsqueeze(0).unsqueeze(0);
        }
        else if (x.dim() == 2)
        {
            if (x.size(0) == 1 || x.size(0) == 2)
            {
                // Likely (channels, time)
                return x.unsqueeze(0);
            }
            else
            {
                // Likely (batch, time)
                return x.unsqueeze(1);
            }
        }
        else if (x.dim() == 3)
        {
            // Already (batch, channels, time)
            return x;
        }
        throw new ArgumentException($"Unsupported tensor shape: {string.Join(",", x.shape)}");
    }

    /// <summary>
    /// Reshapes tensors for SI-SDR computation and ensures matching shapes
    /// </summary>
    private (Tensor references, Tensor estimates) PrepareInputs(Tensor references, Tensor estimates)
    {
        references = NormalizeShape(references);
        estimates = NormalizeShape(estimates);

        // Verify matching dimensions
        if (!references.shape.SequenceEqual(estimates.shape))
        {
            throw new ArgumentException(
                $"Shape mismatch: references {string.Join(",", references.shape)} " +
                $"vs estimates {string.Join(",", estimates.shape)}"
            );
        }

        // Reshape to (batch * channels, 1, time) for easier processing
        var (batchSize, channels, timeSteps) = (references.size(0), references.size(1), references.size(2));
        references = references.reshape(batchSize * channels, 1, timeSteps);
        estimates = estimates.reshape(batchSize * channels, 1, timeSteps);

        return (references, estimates);
    }

    /// <summary>
    /// Computes SI-SDR loss between estimate and reference signals
    /// </summary>
    public override Tensor forward(Tensor x, Tensor y)
    {
        var (xAudio, _) = GetAudioTensor(x);
        var (yAudio, _) = GetAudioTensor(y);

        var (references, estimates) = PrepareInputs(yAudio, xAudio); // Note reversed order - y is reference

        // Get original shape for later
        var originalShape = references.shape;
        var batchChannels = references.size(0);

        // Permute to put time dimension second for easier operations
        references = references.permute(0, 2, 1); // (batch*channels, time, 1)
        estimates = estimates.permute(0, 2, 1);

        // Zero-mean if requested
        if (_zeroMean)
        {
            var referenceMean = references.mean([1], keepdim: true);
            var estimateMean = estimates.mean([1], keepdim: true);
            references = references - referenceMean;
            estimates = estimates - estimateMean;
        }

        // Compute scaling factor if requested
        var scale = ones_like(references);
        if (_scaling)
        {
            var referencesPower = (references * references).sum(-2) + EPS;
            var crossCorrelation = (estimates * references).sum(-2) + EPS;
            scale = (crossCorrelation / referencesPower).unsqueeze(1);
        }

        // Compute SI-SDR components
        var targetSignal = scale * references;
        var errorSignal = estimates - targetSignal;

        var targetPower = (targetSignal * targetSignal).sum(1);
        var errorPower = (errorSignal * errorSignal).sum(1);

        // Compute loss in dB
        var loss = -10 * log10(targetPower / (errorPower + EPS) + EPS);

        // Apply clipping if requested
        if (_clipMin.HasValue)
        {
            loss = clamp(loss, min: _clipMin.Value);
        }

        // Reshape back to match batch/channel structure
        loss = loss.reshape(originalShape[0], 1);

        // Apply reduction
        if (_reduction == "mean")
        {
            loss = loss.mean();
        }
        else if (_reduction == "sum")
        {
            loss = loss.sum();
        }
        // else "none" - return as is

        return loss * _weight;
    }

    /// <summary>
    /// Disposes of unmanaged resources
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // No tensors to dispose since we don't keep any persistent state
        }
        base.Dispose(disposing);
    }
}