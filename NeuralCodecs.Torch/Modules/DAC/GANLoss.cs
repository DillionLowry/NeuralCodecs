using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// GAN losses for audio generation, supporting both AudioSignal and raw tensors
/// </summary>
public class GANLoss : AudioLossBase
{
    private readonly Module<Tensor, Tensor[]> discriminator;

    /// <summary>
    /// Initializes a new instance of GANLoss
    /// </summary>
    /// <param name="discriminator">The discriminator network that returns both final output and intermediate features</param>
    /// <param name="sampleRate">Sample rate for raw audio input</param>
    public GANLoss(
        Module<Tensor, Tensor[]> discriminator,
        int sampleRate = 44100) : base(nameof(GANLoss), sampleRate)
    {
        this.discriminator = discriminator;
        RegisterComponents();
    }

    /// <summary>
    /// Gets the discriminator outputs including intermediate features
    /// </summary>
    private (Tensor[] fake, Tensor[] real) GetDiscriminatorOutputs(Tensor fake, Tensor real)
    {
        return (this.discriminator.forward(fake), this.discriminator.forward(real));
    }

    /// <summary>
    /// Computes the discriminator loss
    /// </summary>
    public Tensor DiscriminatorLoss(object fake, object real)
    {
        var (fakeAudio, _) = GetAudioTensor(fake);
        var (realAudio, _) = GetAudioTensor(real);

        // Detach fake data to avoid training generator
        fakeAudio = fakeAudio.detach();

        var (fakePreds, realPreds) = GetDiscriminatorOutputs(fakeAudio, realAudio);

        // Get final discriminator outputs (last in array)
        var dFake = fakePreds[^1];
        var dReal = realPreds[^1];

        // Compute adversarial loss
        var lossFake = mean(dFake.pow(2));
        var lossReal = mean((ones_like(dReal) - dReal).pow(2));

        return (lossFake + lossReal) * 0.5f;
    }

    /// <summary>
    /// Computes the generator loss and feature matching loss
    /// </summary>
    public (Tensor lossG, Tensor lossFeature) GeneratorLoss(object fake, object real)
    {
        var (fakeAudio, _) = GetAudioTensor(fake);
        var (realAudio, _) = GetAudioTensor(real);

        var (fakePreds, realPreds) = GetDiscriminatorOutputs(fakeAudio, realAudio);

        // Generator tries to fool discriminator
        var lossG = mean((ones_like(fakePreds[^1]) - fakePreds[^1]).pow(2));

        // Feature matching loss across all intermediate layers
        var lossFeature = zeros(1, device: fakeAudio.device);
        for (int i = 0; i < fakePreds.Length - 1; i++)
        {
            lossFeature += functional.l1_loss(
                fakePreds[i],
                realPreds[i].detach()
            );
        }

        return (lossG, lossFeature);
    }

    /// <summary>
    /// Forward pass returning both discriminator outputs
    /// </summary>
    public override Tensor forward(Tensor fake, Tensor real)
    {
        var (fakeAudio, _) = GetAudioTensor(fake);
        var (realAudio, _) = GetAudioTensor(real);
        var (fakePreds, realPreds) = GetDiscriminatorOutputs(fakeAudio, realAudio);
        return fakePreds[^1];  // Return final discriminator output
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.discriminator?.Dispose();
        }
        base.Dispose(disposing);
    }
}