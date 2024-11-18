using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Loading;
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
    private readonly int samplingRate;
    private readonly int encoderDim;
    private readonly int[] encoderRates;

    private readonly int decoderDim;
    private readonly int[] decoderRates;
    private readonly int latentDim;
    private readonly int? attnWindowSize;
    private readonly int numCodebooks;
    private readonly int codebookSize;
    private readonly int codebookDim;
    private readonly int[] vqStrides;
    private readonly int hopLength;

    private readonly Encoder encoder;
    private readonly ResidualVectorQuantizer quantizer;
    private readonly Decoder decoder;

    public ModelConfig Config { get; set; }

    public string Architecture => "Torch";

    public Core.Models.Device Device { get; set; }

    public SNAC(
        int samplingRate = 44100,
        int encoderDim = 64,
        int[] encoderRates = null,
        int? latentDim = null,
        int decoderDim = 1536,
        int[] decoderRates = null,
        int? attnWindowSize = 32,
        int codebookSize = 4096,
        int codebookDim = 8,
        int[] vqStrides = null,
        bool noise = true,
        bool depthwise = true) : base("SNAC")
    {
        this.samplingRate = samplingRate;
        this.encoderDim = encoderDim;
        this.encoderRates = encoderRates ?? [3, 3, 7, 7];
        this.decoderDim = decoderDim;
        this.decoderRates = decoderRates ?? [7, 7, 3, 3];
        this.latentDim = latentDim ?? encoderDim * (1 << this.encoderRates.Length);
        this.hopLength = this.encoderRates.Aggregate((a, b) => a * b);
        encoder = new Encoder(
            encoderDim,
            this.encoderRates,
            depthwise,
            attnWindowSize);

        this.vqStrides = vqStrides ?? [8, 4, 2, 1];
        this.numCodebooks = vqStrides?.Length ?? 4;
        this.codebookSize = codebookSize;
        this.codebookDim = codebookDim;

        this.attnWindowSize = attnWindowSize;

        quantizer = new ResidualVectorQuantizer(
            this.latentDim,
            codebookSize,
            codebookDim,
            this.vqStrides);

        decoder = new Decoder(
            this.latentDim,
            decoderDim,
            this.decoderRates,
            noise,
            depthwise,
            attnWindowSize);
        RegisterComponents();
    }

    public SNAC(SNACConfig config, torch.Device? device = null) : base("SNAC")
    {
        this.samplingRate = config.SamplingRate;
        this.encoderDim = config.EncoderDim;
        this.encoderRates = config.EncoderRates;
        this.decoderDim = config.DecoderDim;
        this.decoderRates = config.DecoderRates;
        this.latentDim = config.LatentDim ?? config.EncoderDim * (1 << config.EncoderRates.Length);
        this.hopLength = config.EncoderRates.Aggregate((a, b) => a * b);

        encoder = new Encoder(
            config.EncoderDim,
            config.EncoderRates,
            config.Depthwise,
            config.AttnWindowSize);

        this.vqStrides = config.VQStrides;
        this.numCodebooks = config.VQStrides.Length;
        this.codebookSize = config.CodebookSize;
        this.codebookDim = config.CodebookDim;
        this.attnWindowSize = config.AttnWindowSize;

        quantizer = new ResidualVectorQuantizer(
            this.latentDim,
            config.CodebookSize,
            config.CodebookDim,
            config.VQStrides);

        decoder = new Decoder(
            this.latentDim,
            config.DecoderDim,
            config.DecoderRates,
            config.Noise,
            config.Depthwise,
            config.AttnWindowSize);

        RegisterComponents();

        //if (config.Device != null)
        //{
        //    this.to(device);
        //}
    }
        /// <summary>
        /// Preprocessing method for input audio
        /// </summary>
        /// <param name="audioData">Raw audio tensor</param>
        /// <returns>Padded and normalized audio tensor</returns>
        private Tensor Preprocess(Tensor audioData)
    {
        var length = audioData.size(-1);

        long lcm = MathUtils.LCM(vqStrides[0], this.attnWindowSize ?? 1);
        long padTo = hopLength * lcm;

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

    //T2 INeuralCodec.Encode<T1, T2>(T1 input)
    //{
    //    throw new NotImplementedException();
    //}
}