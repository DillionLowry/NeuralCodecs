using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public class SISDRLoss : Module<AudioSignal, AudioSignal, Tensor>
{
    private readonly bool scaling;
    private readonly string reduction;
    private readonly bool zeroMean;
    private readonly float? clipMin;
    private readonly float weight;

    public SISDRLoss(
        bool scaling = true,
        string reduction = "mean",
        bool zeroMean = true,
        float? clipMin = null,
        float weight = 1.0f) : base("SISDRLoss")
    {
        this.scaling = scaling;
        this.reduction = reduction;
        this.zeroMean = zeroMean;
        this.clipMin = clipMin;
        this.weight = weight;
        RegisterComponents();
    }

    public override Tensor forward(AudioSignal x, AudioSignal y)
    {
        const float eps = 1e-8f;

        var references = x is AudioSignal ? x.audio_data : x as Tensor;
        var estimates = y is AudioSignal ? y.audio_data : y as Tensor;

        // Reshape to (batch, 1, time)
        var nb = references.shape[0];
        references = references.reshape(nb, 1, -1).permute(0, 2, 1);
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1);

        // Zero mean if needed
        Tensor meanReference = 0, meanEstimate = 0;
        if (zeroMean)
        {
            meanReference = references.mean(dim: 1, keepDim: true);
            meanEstimate = estimates.mean(dim: 1, keepDim: true);
        }

        var _references = references - meanReference;
        var _estimates = estimates - meanEstimate;

        // Calculate projections
        var referencesProjection = _references.pow(2).sum(dim: -2) + eps;
        var referencesOnEstimates = (_estimates * _references).sum(dim: -2) + eps;

        // Calculate scale
        var scale = scaling ?
            (referencesOnEstimates / referencesProjection).unsqueeze(1) :
            torch.ones_like(referencesProjection).unsqueeze(1);

        // True signal and residual
        var eTrue = scale * _references;
        var eRes = _estimates - eTrue;

        var signal = eTrue.pow(2).sum(dim: 1);
        var noise = eRes.pow(2).sum(dim: 1);

        var sdr = -10 * torch.log10(signal / noise + eps);

        if (clipMin.HasValue)
        {
            sdr = torch.clamp(sdr, min: clipMin.Value);
        }

        // Reduction
        return reduction switch
        {
            "mean" => sdr.mean(),
            "sum" => sdr.sum(),
            _ => sdr,
        };
    }
}