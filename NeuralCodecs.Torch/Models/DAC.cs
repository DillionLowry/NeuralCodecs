using NeuralCodecs.Core;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Utils;
using NeuralCodecs.Torch.Config.DAC;
using NeuralCodecs.Torch.Modules.DAC;
using NeuralCodecs.Torch.Utils;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

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
    private readonly Decoder decoder;

    private DACConfig _config;
    public IModelConfig Config => _config;
    public ResidualVectorQuantizer Quantizer => quantizer;

    /// <summary>
    /// Initializes a new instance of the DAC model.
    /// </summary>
    /// <param name="config">Configuration parameters for the codec</param>
    /// <remarks>
    /// In the original Python implementation, the quantizerDropout is a bool parameter which is passed to
    /// the quantizer as a float, effectively setting dropout to either 1.0 or 0, which the quantizer uses
    /// to determine what fraction of the batch will have randomized quantizer counts. This contradicts the
    /// original research paper, which shows the optimal value to be 0.5.
    /// </remarks>
    public DAC(DACConfig config) : base("DAC")
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;

        _encoderDim = config.EncoderDim;
        _encoderRates = config.EncoderRates ?? [2, 4, 8, 8];
        _decoderDim = config.DecoderDim;
        _decoderRates = config.DecoderRates ?? [8, 8, 4, 2];
        _sampleRate = config.SampleRate;

        // Calculate latent dimension if not provided
        _latentDim = config.LatentDim ?? _encoderDim * (1 << _encoderRates.Length);

        // Calculate total stride (hop length) as product of encoder rates
        _hopLength = _encoderRates.Aggregate((a, b) => a * b);

        encoder = new Encoder(
            dModel: _encoderDim,
            strides: _encoderRates,
            dLatent: _latentDim);

        _nCodebooks = config.NumCodebooks;
        _codebookSize = config.CodebookSize;
        _codebookDim = config.CodebookDim;

        quantizer = new ResidualVectorQuantizer(
            inputDim: _latentDim,
            nCodebooks: _nCodebooks,
            codebookSize: _codebookSize,
            codebookDim: _codebookDim,
            quantizerDropout: config.QuantizerDropout);

        // Create decoder
        decoder = new Decoder(
            inputChannel: _latentDim,
            channels: _decoderDim,
            rates: _decoderRates);

        RegisterComponents();
        this.to(_device);
        InitializeWeights();
    }

    /// <summary>
    /// Decodes quantized codes back into audio waveform.
    /// This method reconstructs audio from previously encoded discrete codes using the quantizer's decoder.
    /// </summary>
    /// <param name="codes">Tensor containing the quantized codes to decode. Shape should be compatible with the quantizer's expected input format.</param>
    /// <returns>Reconstructed audio tensor in the original audio domain.</returns>
    public Tensor FromCodes(Tensor codes)
    {
        using var inference = torch.inference_mode();
        var (audio, _, _) = quantizer.FromCodes(codes);
        return audio.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Initializes the model's weights using Kaiming normal initialization.
    /// </summary>
    private void InitializeWeights()
    {
        foreach (var module in modules())
        {
            if (module is Conv1d conv)
            {
                init.kaiming_normal_(conv.weight);
                if (conv.bias is not null)
                {
                    init.zeros_(conv.bias);
                }
            }
            else if (module is ConvTranspose1d convT)
            {
                init.kaiming_normal_(convT.weight);
                if (convT.bias is not null)
                {
                    init.zeros_(convT.bias);
                }
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
        sampleRate ??= _sampleRate;
        if (sampleRate != _sampleRate)
        {
            throw new ArgumentException(
                $"Input audio sample rate {sampleRate}Hz does not match model sample rate {_sampleRate}Hz",
                nameof(sampleRate));
        }

        var length = audioData.size(-1);
        var rightPad = (long)(Math.Ceiling((double)length / _hopLength) * _hopLength) - length;
        return functional.pad(audioData, [0L, rightPad]);
    }

    /// <summary>
    /// Encodes the input audio data through the encoder and quantizer.
    /// </summary>
    /// <param name="audioData">Input audio tensor.</param>
    /// <param name="nQuantizers">Number of quantizers to use (optional).</param>
    /// <param name="sampleRate">Sample rate of the input audio (optional).</param>
    /// <returns>Tuple containing the quantized output, codes, latents, and losses.</returns>
    public (Tensor z, Tensor codes, Tensor latents, Tensor commitmentLoss, Tensor codebookLoss)
        Encode(Tensor audioData, int? nQuantizers = null, int? sampleRate = null)
    {
        using var scope = NewDisposeScope();
        using var inferenceScope = torch.inference_mode();

        var processed = Preprocess(audioData, sampleRate);
        var z = encoder.forward(processed);
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
    /// Encodes the input audio data through the encoder and quantizer.
    /// </summary>
    /// <param name="audioData">Input audio tensor.</param>
    /// <returns>Quantized audio tensor.</returns>
    public Tensor EncodeAudio(Tensor audioData)
    {
        using var scope = NewDisposeScope();
        using var inferenceScope = torch.inference_mode();

        var processed = Preprocess(audioData, _sampleRate);
        var z = encoder.forward(processed);
        var (zQ, _, _, _, _) = quantizer.forward(z, null);

        return zQ.MoveToOuterDisposeScope();
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
        using var inferenceScope = torch.inference_mode();

        // Convert input audio data to tensor
        var inputTensor = torch.tensor(audioData, dtype: torch.float32, _device)
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

    /// <summary>
    /// Decodes the latent representation back to audio.
    /// </summary>
    /// <param name="qAudio">Latent representation to decode.</param>
    /// <returns>Reconstructed audio array.</returns>
    public float[] Decode(float[] qAudio)
    {
        ArgumentNullException.ThrowIfNull(qAudio);
        using var scope = torch.NewDisposeScope();
        // Convert input latent data to tensor
        var inputTensor = torch.tensor(qAudio, dtype: torch.float32, _device)
                              .reshape(1, _latentDim, -1)
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

        var (z, codes, latents, commitmentLoss, codebookLoss) = Encode(audioData, nQuantizers, sampleRate);
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
    /// Performs a forward pass through the model.
    /// </summary>
    /// <param name="audioData">Input audio tensor.</param>
    /// <returns>Reconstructed audio tensor.</returns>
    public float[] forward(float[] audioData)
    {
        ArgumentNullException.ThrowIfNull(audioData);
        using var scope = torch.NewDisposeScope();
        // Convert input audio data to tensor
        var inputTensor = torch.tensor(audioData, dtype: torch.float32, _device)
                              .reshape(1, 1, -1)
                              .to(_device);
        // Preprocess and forward
        var outputs = forward(inputTensor);
        // Convert audio to float array
        return outputs["audio"].cpu().detach().to(torch.float32).data<float>().ToArray();
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

    /// <summary>
    /// Loads the model weights from the specified file path.
    /// </summary>
    /// <param name="path">The file path to load the weights from.</param>
    /// <exception cref="FileNotFoundException">Thrown when the specified file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown when there is an error loading the weights.</exception>
    public void LoadWeights(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"DAC weights not found at {path}");
        }

        try
        {
            set_default_device(CPU);
            if (_device != CPU)
            {
                using (no_grad())
                {
                    this.to(CPU);
                }
            }

            switch (FileUtils.DetectFileType(path))
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

            if (_device != CPU)
            {
                using (no_grad())
                {
                    this.to(_device);
                    set_default_device(_device);
                }
            }
        }
        catch (Exception ex) when (ex is not (FileNotFoundException or InvalidOperationException))
        {
            throw new InvalidOperationException($"Failed to load DAC weights from {path} {ex.Message}", ex);
        }
    }
}