using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

//using static TqdmSharp.Tqdm;

namespace NeuralCodecs.Torch;

/// <summary>
/// Implements the Differentiable Audio Codec (DAC) neural network model.
/// This model consists of an encoder, vector quantizer, and decoder for audio compression.
/// </summary>
public class DAC : Module<Tensor, Dictionary<string, Tensor>>
{
    private readonly int encoderDim;
    private readonly int[] encoderRates;
    private readonly int decoderDim;
    private readonly int[] decoderRates;
    private readonly int latentDim;
    private readonly int nCodebooks;
    private readonly int codebookSize;
    private readonly int codebookDim; // original python uses a union here, but never uses the list option

    private readonly Encoder encoder;
    private readonly ResidualVectorQuantize quantizer;
    private readonly Decoder decoder; // use null attention

    public int SampleRate { get; }
    public int HopLength { get; }
    //public Device Device => this.device;
    //public Module<Tensor, Tensor> Quantizer => quantizer;

    /// <summary>
    /// Initializes a new instance of the DAC model.
    /// </summary>
    /// <param name="encoderDim">Dimension of the encoder's hidden layers.</param>
    /// <param name="encoderRates">Stride rates for each encoder layer.</param>
    /// <param name="latentDim">Dimension of the latent space. If null, calculated from encoderDim.</param>
    /// <param name="decoderDim">Dimension of the decoder's hidden layers.</param>
    /// <param name="decoderRates">Stride rates for each decoder layer.</param>
    /// <param name="nCodebooks">Number of codebooks in the vector quantizer.</param>
    /// <param name="codebookSize">Size of each codebook.</param>
    /// <param name="codebookDim">Dimension of each codebook entry.</param>
    /// <param name="quantizerDropout">Dropout rate to use in the quantizer training.</param>
    /// <param name="sampleRate">Audio sample rate the model operates on.</param>
    /// <remarks>
    /// In the original Python implementation, the quantizerDropout is a bool parameter which is passed to 
    /// the quantizer as a float, effectively setting dropout to either 1.0 or 0, which the quantizer uses 
    /// to determine what fraction of the batch will have randomized quantizer counts. This contradicts the
    /// original research paper, which shows the optimal value to be 0.5.
    /// </remarks>
    public DAC(
        int encoderDim = 64,
        int[] encoderRates = null,
        int? latentDim = null,
        int decoderDim = 1536,
        int[] decoderRates = null,
        int nCodebooks = 9,
        int codebookSize = 1024,
        int codebookDim = 8,
        float quantizerDropout = 0.0f,
        int sampleRate = 44100) : base("DAC")
    {
        this.encoderDim = encoderDim;
        this.encoderRates = encoderRates ?? [2, 4, 8, 8];
        this.decoderDim = decoderDim;
        this.decoderRates = decoderRates ?? [8, 8, 4, 2];
        this.SampleRate = sampleRate;

        // Calculate latent dimension if not provided
        this.latentDim = latentDim ?? encoderDim * (1 << this.encoderRates.Length);

        // Calculate total stride (hop length)as product of encoder rates
        this.HopLength = this.encoderRates.Aggregate((a, b) => a * b);

        // Create encoder
        encoder = new Encoder(
            dModel: this.encoderDim,
            strides: this.encoderRates,
            dLatent: this.latentDim);

        this.nCodebooks = nCodebooks;
        this.codebookSize = codebookSize;
        this.codebookDim = codebookDim;

        // Create quantizer
        quantizer = new ResidualVectorQuantize(
            inputDim: this.latentDim,
            nCodebooks: this.nCodebooks,
            codebookSize: this.codebookSize,
            codebookDim: this.codebookDim,
            quantizerDropout: quantizerDropout);

        // Create decoder
        decoder = new Decoder(
            inputChannel: this.latentDim,
            channels: this.decoderDim,
            rates: this.decoderRates);

        RegisterComponents();
        InitializeWeights();

        // todo: This would come from CodecMixin
        // this.delay = this.GetDelay();
    }

    /// <summary>
    /// Initializes the model's weights using Kaiming normal initialization.
    /// </summary>
    private void InitializeWeights()
    {
        // Implement weight initialization
        // Similar to init_weights in Python
        foreach (var module in this.modules())
        {
            if (module is Conv1d conv)
            {
                init.kaiming_normal_(conv.weight);
                if (conv.bias is not null)
                    init.zeros_(conv.bias);
            }
            else if (module is ConvTranspose1d convT)
            {
                init.kaiming_normal_(convT.weight);
                if (convT.bias is not null)
                    init.zeros_(convT.bias);
            }
        }
    }

    /// <summary>
    /// Preprocesses the input audio data by padding it to match the model's hop length.
    /// </summary>
    /// <param name="audioData">Input audio tensor.</param>
    /// <param name="sampleRate">Sample rate of the input audio.</param>
    /// <returns>Padded audio tensor.</returns>
    /// <exception cref="ArgumentException">Thrown when the sample rate doesn't match the model's expected rate.</exception>
    private Tensor Preprocess(Tensor audioData, int? sampleRate = null)
    {
        sampleRate ??= this.SampleRate;
        if (sampleRate != this.SampleRate)
            throw new ArgumentException($"Expected sample rate {this.SampleRate}, got {sampleRate}");

        var length = audioData.size(-1);
        var rightPad = (int)Math.Ceiling(length / (double)HopLength) * HopLength - length;

        var padded = nn.functional.pad(audioData, new[] { 0L, rightPad });

        return padded;
    }

    /// <summary>
    /// Encodes the input audio data through the encoder and quantizer.
    /// </summary>
    /// <param name="audioData">Input audio tensor.</param>
    /// <param name="nQuantizers">Number of quantizers to use (optional).</param>
    /// <returns>Tuple containing the quantized output, codes, latents, and losses.</returns>
    public (Tensor z, Tensor codes, Tensor latents, Tensor commitmentLoss, Tensor codebookLoss)
        Encode(Tensor audioData, int? nQuantizers = null)
    {
        using var scope = NewDisposeScope();

        var z = encoder.forward(audioData);
        var (zQ, codes, latents, commitmentLoss, codebookLoss) =
            quantizer.forward(z, nQuantizers);

        return (
            zQ.MoveToOuterDisposeScope(),
            codes.MoveToOuterDisposeScope(),
            latents.MoveToOuterDisposeScope(),
            commitmentLoss.MoveToOuterDisposeScope(),
            codebookLoss.MoveToOuterDisposeScope()
        );
    }

    /// <summary>
    /// Decodes the latent representation back to audio.
    /// </summary>
    /// <param name="z">Latent representation to decode.</param>
    /// <returns>Reconstructed audio tensor.</returns>
    public Tensor Decode(Tensor z)
    {
        return decoder.forward(z);
    }

    /// <summary>
    /// Performs a forward pass through the model with additional parameters.
    /// </summary>
    /// <param name="audioData">Input audio tensor.</param>
    /// <param name="sampleRate">Sample rate of the input audio.</param>
    /// <param name="nQuantizers">Number of quantizers to use.</param>
    /// <returns>Dictionary containing the model's outputs including reconstructed audio and losses.</returns>
    public Dictionary<string, Tensor> forward(
        Tensor audioData,
        int? sampleRate,
        int? nQuantizers)
    {
        using var scope = NewDisposeScope();

        var length = audioData.shape[^1];
        audioData = Preprocess(audioData, sampleRate);

        var (z, codes, latents, commitmentLoss, codebookLoss) = Encode(audioData, nQuantizers);

        var audio = Decode(z);
        // Trim to original length
        audio = audio.narrow(-1, 0, length);

        return new Dictionary<string, Tensor>
        {
            ["audio"] = audio.MoveToOuterDisposeScope(),
            ["z"] = z.MoveToOuterDisposeScope(),
            ["codes"] = codes.MoveToOuterDisposeScope(),
            ["latents"] = latents.MoveToOuterDisposeScope(),
            ["vq/commitment_loss"] = commitmentLoss.MoveToOuterDisposeScope(),
            ["vq/codebook_loss"] = codebookLoss.MoveToOuterDisposeScope(),
        };
    }

    /// <summary>
    /// Performs a forward pass through the model.
    /// </summary>
    /// <param name="audioData">Input audio tensor.</param>
    /// <returns>Dictionary containing the model's outputs including reconstructed audio and losses.</returns>
    public override Dictionary<string, Tensor> forward(Tensor audioData)
    {
        using var scope = NewDisposeScope();

        var length = audioData.shape[^1];

        var preprocessed = Preprocess(audioData);

        var (z, codes, latents, commitmentLoss, codebookLoss) = Encode(preprocessed);
        var audio = Decode(z);

        // Trim to original length
        audio = audio.narrow(-1, 0, length);

        return new Dictionary<string, Tensor>
        {
            ["audio"] = audio.MoveToOuterDisposeScope(),
            ["z"] = z.MoveToOuterDisposeScope(),
            ["codes"] = codes.MoveToOuterDisposeScope(),
            ["latents"] = latents.MoveToOuterDisposeScope(),
            ["vq/commitment_loss"] = commitmentLoss.MoveToOuterDisposeScope(),
            ["vq/codebook_loss"] = codebookLoss.MoveToOuterDisposeScope(),
        };
    }

    /// <summary>
    /// Disposes the model's resources.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            encoder?.Dispose();
            quantizer?.Dispose();
            decoder?.Dispose();
        }
        base.Dispose(disposing);
    }


}