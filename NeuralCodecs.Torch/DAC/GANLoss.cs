using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public class GANLoss : Module<AudioSignal, AudioSignal, (Tensor[] dFake, Tensor[] dReal)>
{
    private readonly Module<Tensor, Tensor[]> discriminator;

    public GANLoss(Module<Tensor, Tensor[]> discriminator) : base("GANLoss")
    {
        this.discriminator = discriminator;
        RegisterComponents();
    }

    public override (Tensor[] dFake, Tensor[] dReal) forward(AudioSignal fake, AudioSignal real)
    {
        var dFake = discriminator.forward(fake.audio_data);
        var dReal = discriminator.forward(real.audio_data);
        return (dFake, dReal);
    }

    public Tensor DiscriminatorLoss(AudioSignal fake, AudioSignal real)
    {
        // Detach fake samples since we don't want to update generator
        var detachedFake = fake.audio_data.detach();
        var (dFake, dReal) = forward(new AudioSignal(detachedFake), real);

        var lossD = torch.zeros(1, device: fake.audio_data.device);

        for (int i = 0; i < dFake.Length; i++)
        {
            // Last element contains the final discriminator output
            var fakePred = dFake[i][^1];
            var realPred = dReal[i][^1];

            lossD += torch.mean(fakePred.pow(2));
            lossD += torch.mean((torch.ones_like(realPred) - realPred).pow(2));
        }

        return lossD;
    }

    public (Tensor lossG, Tensor lossFeature) GeneratorLoss(AudioSignal fake, AudioSignal real)
    {
        var (dFake, dReal) = forward(fake, real);

        var lossG = torch.zeros(1, device: fake.audio_data.device);

        // Generator tries to fool discriminator
        foreach (var fakePreds in dFake)
        {
            lossG += torch.mean((torch.ones_like(fakePreds[^1]) - fakePreds[^1]).pow(2));
        }

        // Feature matching loss
        var lossFeature = torch.zeros(1, device: fake.audio_data.device);

        for (int i = 0; i < dFake.Length; i++)
        {
            // Compare all intermediate features except the last one
            for (int j = 0; j < dFake[i].Length - 1; j++)
            {
                lossFeature += nn.functional.l1_loss(
                    dFake[i][j],
                    dReal[i][j].detach()
                );
            }
        }

        return (lossG, lossFeature);
    }
}