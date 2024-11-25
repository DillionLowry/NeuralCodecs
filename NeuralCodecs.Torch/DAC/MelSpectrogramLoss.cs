using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public class MelSpectrogramLoss : Module<AudioSignal, AudioSignal, Tensor>
{
    private readonly List<STFTParams> stftParams;
    private readonly int[] nMels;
    private readonly Loss<Tensor, Tensor, Tensor> lossFn;
    private readonly float clampEps;
    private readonly float magWeight;
    private readonly float logWeight;
    private readonly float pow;
    private readonly float weight;
    private readonly float[] melFmin;
    private readonly float?[] melFmax;

    public MelSpectrogramLoss(
        int[] nMels = null,
        int[] windowLengths = null,
        Loss<Tensor, Tensor, Tensor> lossFn = null,
        float clampEps = 1e-5f,
        float magWeight = 1.0f,
        float logWeight = 1.0f,
        float pow = 2.0f,
        float weight = 1.0f,
        bool matchStride = false,
        float[] melFmin = null,
        float?[] melFmax = null,
        string windowType = null) : base("MelSpectrogramLoss")
    {
        // Default values
        nMels ??= new[] { 150, 80 };
        windowLengths ??= new[] { 2048, 512 };
        melFmin ??= new[] { 0.0f, 0.0f };
        melFmax ??= new float?[] { null, null };
        this.lossFn = lossFn ?? nn.L1Loss();

        this.nMels = nMels;
        this.clampEps = clampEps;
        this.magWeight = magWeight;
        this.logWeight = logWeight;
        this.pow = pow;
        this.weight = weight;
        this.melFmin = melFmin;
        this.melFmax = melFmax;

        // Create STFT parameters for each window length
        stftParams = windowLengths.Select(w => new STFTParams(
            windowLength: w,
            hopLength: w / 4,
            matchStride: matchStride,
            windowType: windowType
        )).ToList();

        RegisterComponents();
    }

    public override Tensor forward(AudioSignal x, AudioSignal y)
    {
        var loss = torch.zeros(1, dtype: torch.float32);

        for (int i = 0; i < stftParams.Count; i++)
        {
            var s = stftParams[i];
            var fmin = melFmin[i];
            var fmax = melFmax[i];
            var mel_n = nMels[i];

            var kwargs = new Dictionary<string, object> {
                    { "windowLength", s.windowLength },
                    { "hopLength", s.hopLength },
                    { "windowType", s.windowType }
                };

            // Get mel spectrograms
            var xMels = x.mel_spectrogram(mel_n, fmin, fmax,
                s.windowLength, s.hopLength, s.windowType);
            var yMels = y.mel_spectrogram(mel_n, fmin, fmax,
                s.windowLength, s.hopLength, s.windowType);

            // Log magnitude loss
            loss += logWeight * lossFn.forward(
                xMels.clamp(clampEps).pow(pow).log10(),
                yMels.clamp(clampEps).pow(pow).log10()
            );

            // Raw magnitude loss
            loss += magWeight * lossFn.forward(xMels, yMels);
        }

        return loss;
    }
}