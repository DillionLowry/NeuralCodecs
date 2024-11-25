using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public class MultiScaleSTFTLoss : Module<AudioSignal, AudioSignal, Tensor>
{
    private readonly List<STFTParams> stftParams;
    private readonly Loss<Tensor, Tensor, Tensor> lossFn;
    private readonly float clampEps;
    private readonly float magWeight;
    private readonly float logWeight;
    private readonly float pow;
    private readonly float weight;

    public MultiScaleSTFTLoss(
        int[] windowLengths = null,
        Loss<Tensor, Tensor, Tensor> lossFn = null,
        float clampEps = 1e-5f,
        float magWeight = 1.0f,
        float logWeight = 1.0f,
        float pow = 2.0f,
        float weight = 1.0f,
        bool matchStride = false,
        string windowType = null) : base("MultiScaleSTFTLoss")
    {
        windowLengths ??= new[] { 2048, 512 };
        this.lossFn = lossFn ?? nn.L1Loss();

        stftParams = windowLengths.Select(w => new STFTParams(
            windowLength: w,
            hopLength: w / 4,
            matchStride: matchStride,
            windowType: windowType
        )).ToList();

        this.clampEps = clampEps;
        this.magWeight = magWeight;
        this.logWeight = logWeight;
        this.pow = pow;
        this.weight = weight;

        RegisterComponents();
    }

    public override Tensor forward(AudioSignal x, AudioSignal y)
    {
        var loss = torch.zeros(1, dtype: torch.float32);

        foreach (var s in stftParams)
        {
            x.stft(s.windowLength, s.hopLength, s.windowType);
            y.stft(s.windowLength, s.hopLength, s.windowType);

            // Log magnitude loss
            loss += logWeight * lossFn.forward(
                x.magnitude.clamp(clampEps).pow(pow).log10(),
                y.magnitude.clamp(clampEps).pow(pow).log10()
            );

            // Raw magnitude loss
            loss += magWeight * lossFn.forward(x.magnitude, y.magnitude);
        }

        return loss;
    }
}