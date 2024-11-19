using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Torch.Utils;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

/// <summary>
/// Implements a neural audio codec that combines encoding, quantization, and decoding
/// for efficient audio compression and reconstruction.
/// </summary>
public partial class SNAC : Module<Tensor, (Tensor audio, List<Tensor> codes)>, INeuralCodec
{
    private readonly int _hopLength;

    private readonly Encoder encoder;
    private readonly ResidualVectorQuantizer quantizer;
    private readonly Decoder decoder;

    private SNACConfig _config;
    public IModelConfig Config => _config;


    public SNAC(SNACConfig config, torch.Device device) : base("SNAC")
    {
        _config = config;
        _config.LatentDim = config.LatentDim ?? config.EncoderDim * (1 << config.EncoderRates.Length);
        _hopLength = config.EncoderRates.Aggregate((a, b) => a * b);

        encoder = new Encoder(
            _config.EncoderDim,
            _config.EncoderRates,
            _config.Depthwise,
            _config.AttnWindowSize);

        quantizer = new ResidualVectorQuantizer(
            _config.LatentDim.Value,
            _config.CodebookSize,
            _config.CodebookDim,
            _config.VQStrides);

        decoder = new Decoder(
            _config.LatentDim.Value,
            _config.DecoderDim,
            _config.DecoderRates,
            _config.Noise,
            _config.Depthwise,
            _config.AttnWindowSize);

        RegisterComponents();

        if (device != null)
        {
            this.to(device);
        }
    }

    /// <summary>
    /// Preprocessing method for input audio
    /// </summary>
    /// <param name="audioData">Raw audio tensor</param>
    /// <returns>Padded and normalized audio tensor</returns>
    private Tensor Preprocess(Tensor audioData)
    {
        var length = audioData.size(-1);

        long lcm = MathUtils.LCM(_config.VQStrides[0], _config.AttnWindowSize ?? 1);
        long padTo = _hopLength * lcm;

        long rightPad = (long)(Math.Ceiling((double)length / padTo) * padTo) - length;

        audioData = nn.functional.pad(audioData, (0, rightPad));

        return audioData;
    }

    /// <summary>
    /// Performs forward pass through the complete codec pipeline
    /// </summary>
    /// <param name="audioData">Input audio tensor</param>
    /// <returns>
    /// Tuple containing:
    /// - Reconstructed audio tensor
    /// - List of quantization codes
    /// </returns>
    public override (Tensor audio, List<Tensor> codes) forward(Tensor audioData)
    {
        using var scope = NewDisposeScope();
        var length = audioData.shape[-1];
        audioData = Preprocess(audioData);

        var z = encoder.forward(audioData);
        var (zQ, codes) = quantizer.forward(z);
        var audioHat = decoder.forward(zQ);

        // Trim to original length
        audioHat = audioHat.narrow(-1, 0, length);

        return (audioHat.MoveToOuterDisposeScope(), codes);
    }

    public List<Tensor> Encode(Tensor audioData)
    {
        audioData = Preprocess(audioData);
        var z = encoder.forward(audioData);
        var (_, codes) = quantizer.forward(z);

        return codes;
    }

    public Tensor Decode(List<Tensor> codes)
    {
        using var scope = torch.NewDisposeScope();

        var zQ = quantizer.FromCodes(codes);
        var audio = decoder.forward(zQ);

        return audio.MoveToOuterDisposeScope();
    }

    public void LoadWeights(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"PyTorch weights not found at {path}");

        try
        {
            // TODO: torchsharp load
            this.load_py(path);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load PyTorch weights from {path}", ex);
        }
    }
}