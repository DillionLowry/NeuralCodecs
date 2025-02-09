using NeuralCodecs.Core;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Torch.Config.DAC;
using NeuralCodecs.Torch.Utils;
using NeuralCodecs.Torch.Modules.DAC;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Utils;


namespace NeuralCodecs.Torch.Models;

/// <summary>
/// Implements the Differentiable Audio Codec (DAC) neural network model.
/// This model consists of an encoder, vector quantizer, and decoder for audio compression.
/// </summary>
public class DAC : Module<Tensor, Dictionary<string, Tensor>>, INeuralCodec
{
    private readonly int _encoderDim;
    private readonly int[] _encoderRates;
    private readonly int _decoderDim;
    private readonly int[] _decoderRates;
    private readonly int _latentDim;
    private readonly int _nCodebooks;
    private readonly int _codebookSize;
    private readonly int _codebookDim; // original python uses a union here, but never uses the list option
    private readonly int _sampleRate;
    private readonly int _hopLength;
    private torch.Device _device => TorchUtils.GetDevice(_config.Device);

    private readonly Encoder encoder;
    private readonly ResidualVectorQuantizer quantizer;
    private readonly Decoder decoder; // use null attention

    private DACConfig _config;
    public IModelConfig Config => _config;

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
        _encoderDim = encoderDim;
        _encoderRates = encoderRates ?? [2, 4, 8, 8];
        _decoderDim = decoderDim;
        _decoderRates = decoderRates ?? [8, 8, 4, 2];
        _sampleRate = sampleRate;

        // Calculate latent dimension if not provided
        _latentDim = latentDim ?? encoderDim * (1 << _encoderRates.Length);

        // Calculate total stride (hop length)as product of encoder rates
        _hopLength = _encoderRates.Aggregate((a, b) => a * b);

        // Create encoder
        encoder = new Encoder(
            dModel: _encoderDim,
            strides: _encoderRates,
            dLatent: _latentDim);

        _nCodebooks = nCodebooks;
        _codebookSize = codebookSize;
        _codebookDim = codebookDim;

        // Create quantizer
        quantizer = new ResidualVectorQuantizer(
            inputDim: _latentDim,
            nCodebooks: _nCodebooks,
            codebookSize: _codebookSize,
            codebookDim: _codebookDim,
            quantizerDropout: quantizerDropout);

        // Create decoder
        decoder = new Decoder(
            inputChannel: _latentDim,
            channels: _decoderDim,
            rates: _decoderRates);

        RegisterComponents();
        InitializeWeights();
    }

    /// <summary>
    /// Initializes the model's weights using Kaiming normal initialization.
    /// </summary>
    private void InitializeWeights()
    {
        // Implement weight initialization
        // Similar to init_weights in Python
        foreach (var module in modules())
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
    public DAC(DACConfig config) : this(
        encoderDim: config.EncoderDim,
        encoderRates: config.EncoderRates,
        latentDim: config.LatentDim,
        decoderDim: config.DecoderDim,
        decoderRates: config.DecoderRates,
        nCodebooks: config.NumCodebooks,
        codebookSize: config.CodebookSize,
        codebookDim: config.CodebookDim,
        quantizerDropout: config.QuantizerDropout,
        sampleRate: config.SamplingRate)
    {
        _config = config;
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
        // TODO: Implement sample rate check
        sampleRate ??= _sampleRate;

        var length = audioData.size(-1);
        var rightPad = (long)(Math.Ceiling((double)length / _hopLength) * _hopLength) - length;
        return functional.pad(audioData, [0L, rightPad]);
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

        audioData = Preprocess(audioData);
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
    /// Encodes audio data into a list of quantized codes.
    /// </summary>
    /// <param name="audioData">Input audio samples</param>
    /// <returns>Quantized audio array</returns>
    public float[] Encode(float[] audioData)
    {
        ArgumentNullException.ThrowIfNull(audioData);

        using var scope = torch.NewDisposeScope();

        // Convert input audio data to tensor
        var inputTensor = torch.tensor(audioData, dtype: torch.float32)
                              .reshape(1, 1, -1)
                              .to(_device);

        // Preprocess and encode
        inputTensor = Preprocess(inputTensor);
        var z = encoder.forward(inputTensor);
        var (quantized, _, _, _, _) = quantizer.forward(z);

        // Convert quantized to float array
        return quantized.cpu().detach().to(torch.float32).data<float>().ToArray();
    }
    /// <summary>
    /// Decodes the latent representation back to audio.
    /// </summary>
    /// <param name="qAudio">Latent representation to decode.</param>
    /// <returns>Reconstructed audio tensor.</returns>
    public Tensor Decode(Tensor qAudio)
    {
        return decoder.forward(qAudio);
    }
    public float[] Decode(float[] qAudio)
    {
        ArgumentNullException.ThrowIfNull(qAudio);
        using var scope = torch.NewDisposeScope();
        // Convert input latent data to tensor
        var inputTensor = torch.tensor(qAudio, dtype: torch.float32)
                              .reshape(1, 1, -1)
                              .to(_device);
        // Decode
        var audio = decoder.forward(inputTensor);
        // Convert audio to float array
        return audio.cpu().detach().to(torch.float32).data<float>().ToArray();
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
        var (z, codes, latents, commitmentLoss, codebookLoss) = Encode(audioData, nQuantizers);
        var audio = Decode(z);

        return new Dictionary<string, Tensor> //TODO NAME
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
        var (z, codes, latents, commitmentLoss, codebookLoss) = Encode(audioData);
        var audio = Decode(z);

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

    private static string GetRelativePath(string fullPath, string basePath)
    {
        var fullUri = new Uri(fullPath);
        var baseUri = new Uri(basePath);
        var relativeUri = baseUri.MakeRelativeUri(fullUri);
        return Uri.UnescapeDataString(relativeUri.ToString());
    }

    public void LoadWeights(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"DAC weights not found at {path}");

        try
        {
            switch (FileUtil.DetectFileType(path))
            {
                case ModelFileType.Checkpoint:
                    this.load_checkpoint(path);
                    break;
                default:
                    var (modelDict, config) = DACUnpickler.LoadWithConfig(path);
                    this.load_state_dict(modelDict.StateDict);
                    _config ??= config;
                    break;
            }
        }
        catch (Exception ex) when (ex is not (FileNotFoundException or InvalidOperationException))
        {
            throw new InvalidOperationException($"Failed to load DAC weights from {path} {ex.Message}", ex);
        }
    }
}