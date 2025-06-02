using NeuralCodecs.Core;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Utils;
using NeuralCodecs.Torch.AudioTools;
using NeuralCodecs.Torch.Config.Encodec;
using NeuralCodecs.Torch.Modules.Encodec;
using NeuralCodecs.Torch.Utils;
using System.Diagnostics;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Models;

/// <summary>
/// Neural audio codec implementation combining SEANet architecture with residual vector quantization.
/// Provides high-quality audio compression with configurable bandwidth and streaming capabilities.
/// </summary>
public class Encodec : Module<Tensor, Tensor>, INeuralCodec
{
    #region Fields

    private readonly EncodecConfig _config;
    private readonly float _overlap;
    private readonly float? _segment;
    private readonly List<float> _targetBandwidths;
    private readonly SEANetDecoder decoder;
    private readonly SEANetEncoder encoder;
    private readonly ResidualVectorQuantizer quantizer;
    private float? _bandwidth;
    private EncodecLanguageModel? _lm;
    private bool _lmModelWeightsLoaded;
    public torch.Device Device => TorchUtils.GetDevice(_config.Device);

    #endregion Fields

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the Encodec class with the specified configuration.
    /// </summary>
    /// <param name="config">Configuration parameters for the encoder/decoder network and quantizer.</param>
    /// <exception cref="ArgumentException">Thrown when bandwidth configuration is invalid.</exception>
    public Encodec(EncodecConfig config) : base("Encodec")
    {
        _config = config;
        if (config.Bandwidth is null || !((IList<float>)config.TargetBandwidths).Contains(config.Bandwidth.Value))
        {
            throw new ArgumentException(
                $"Invalid bandwidth {config.Bandwidth}. " +
                $"Select one of [{string.Join(", ", config.TargetBandwidths)}]");
        }
        _targetBandwidths = config.TargetBandwidths.ToList();
        _bandwidth = config.Bandwidth;
        encoder = new SEANetEncoder(
            channels: config.Channels,
            dimension: config.HiddenSize,
            norm: config.NormType,
            causal: config.UseCausalConv
            );

        decoder = new SEANetDecoder(
            channels: config.Channels,
            dimension: config.HiddenSize,
            norm: config.NormType,
            causal: config.UseCausalConv);

        var nQ = (int)(1000 * config.TargetBandwidths.Max() /
            (Math.Ceiling(config.SampleRate / (double)encoder.TotalRatio) * 10));

        quantizer = new ResidualVectorQuantizer(
            dimension: config.CodebookDim,
            numQuantizers: nQ,
            codebookSize: config.CodebookSize);

        SampleRate = config.SampleRate;
        _segment = config.ChunkLengthSeconds;
        Channels = config.Channels;
        SampleRate = config.SampleRate;
        Normalize = config.Normalize;
        name = $"encodec_{config.SampleRate / 1000}khz";
        _overlap = config.Overlap ?? 0;

        FrameRate = (int)Math.Ceiling(SampleRate / (float)encoder.TotalRatio);
        BitsPerCodebook = (int)Math.Log2(quantizer.layers[0].CodebookSize);

        RegisterComponents();
    }

    /// <summary>
    /// Initializes a new instance of the Encodec class with explicit components.
    /// </summary>
    /// <param name="encoder">SEANet encoder component.</param>
    /// <param name="decoder">SEANet decoder component.</param>
    /// <param name="quantizer">Residual vector quantizer component.</param>
    /// <param name="targetBandwidths">List of supported target bandwidths in kbps.</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="channels">Number of audio channels.</param>
    /// <param name="normalize">Whether to apply audio normalization.</param>
    /// <param name="segment">Optional segment length in seconds for chunked processing.</param>
    /// <param name="overlap">Overlap factor between segments (default: 0.01).</param>
    /// <exception cref="ArgumentException">Thrown when quantizer bins are not a power of 2.</exception>
    public Encodec(
        SEANetEncoder encoder,
        SEANetDecoder decoder,
        ResidualVectorQuantizer quantizer,
        List<float> targetBandwidths,
        int sampleRate,
        int channels,
        bool normalize = false,
        float? segment = null,
        float overlap = 0.01f) : base("Encodec")
    {
        this.encoder = encoder;
        this.decoder = decoder;
        this.quantizer = quantizer;
        _targetBandwidths = targetBandwidths;
        SampleRate = sampleRate;
        Channels = channels;
        Normalize = normalize;
        _segment = segment;
        _overlap = overlap;
        name = $"encodec_{sampleRate / 1000}khz";

        FrameRate = (int)Math.Ceiling(sampleRate / (float)encoder.TotalRatio);
        BitsPerCodebook = (int)Math.Log2(quantizer.layers[0].CodebookSize);

        if (Math.Pow(2, BitsPerCodebook) != quantizer.layers[0].CodebookSize)
        {
            throw new ArgumentException("Quantizer bins must be a power of 2");
        }

        RegisterComponents();
    }

    #endregion Constructors

    #region Properties

    /// <summary>
    /// Gets the number of bits used per codebook entry.
    /// </summary>
    public int BitsPerCodebook { get; }

    /// <summary>
    /// Gets the number of audio channels supported by the model.
    /// </summary>
    public int Channels { get; }

    /// <summary>
    /// Gets the model configuration.
    /// </summary>
    public IModelConfig Config => _config;

    /// <summary>
    /// Gets the currently set target bandwidth in kbps.
    /// </summary>
    public float? CurrentBandwidth => _bandwidth;

    /// <summary>
    /// Gets the SEANet encoder component.
    /// </summary>
    public SEANetEncoder Encoder => encoder;

    /// <summary>
    /// Gets the frame rate in Hz after encoding.
    /// </summary>
    public int FrameRate { get; }

    /// <summary>
    /// Gets whether audio normalization is enabled.
    /// </summary>
    public bool Normalize { get; }

    /// <summary>
    /// Gets the number of codebooks used in quantization.
    /// </summary>
    public int NumCodebooks => quantizer.layers.Count;

    /// <summary>
    /// Gets the audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; }

    /// <summary>
    /// Gets the segment length in samples, if segmentation is enabled.
    /// </summary>
    public int? SegmentLength => _segment.HasValue ? (int)(_segment.Value * SampleRate) : null;

    /// <summary>
    /// Gets the stride between segments in samples, accounting for overlap.
    /// </summary>
    public int? SegmentStride => SegmentLength.HasValue ?
        Math.Max(1, (int)((1 - _overlap) * SegmentLength.Value)) : null;

    /// <summary>
    /// Gets the list of supported target bandwidths in kbps.
    /// </summary>
    public IReadOnlyList<float> TargetBandwidths => _targetBandwidths;

    #endregion Properties

    #region Methods

    /// <summary>
    /// Decodes a list of encoded frames back to audio.
    /// </summary>
    /// <param name="encodedFrames">List of encoded frames to decode.</param>
    /// <returns>Decoded audio tensor.</returns>
    /// <exception cref="ArgumentException">Thrown when no frames are provided or when segmentation configuration is invalid.</exception>
    public Tensor Decode(List<EncodedFrame> encodedFrames)
    {
        if (encodedFrames.Count == 0)
        {
            throw new ArgumentException("No frames provided to decode");
        }

        if (SegmentLength == null)
        {
            if (encodedFrames.Count != 1)
            {
                throw new ArgumentException("Expected single frame when no segmentation is used");
            }

            return DecodeFrame(encodedFrames[0]);
        }

        var frames = encodedFrames.ConvertAll(DecodeFrame);

        var result = DSP.LinearOverlapAdd(frames, SegmentStride!.Value);

        return result;
    }

    /// <summary>
    /// Encodes raw audio samples into compressed frames.
    /// </summary>
    /// <param name="audioData">Array of audio samples to encode.</param>
    /// <returns>List of encoded frames.</returns>
    /// <exception cref="ArgumentNullException">Thrown when audioData is null.</exception>
    public List<EncodedFrame> Encode(float[] audioData)
    {
        ArgumentNullException.ThrowIfNull(audioData);

        // Convert input audio data to tensor
        var inputTensor = tensor(audioData, dtype: float32, device: Device)
                                .reshape(1, _config.Channels, -1);
        return Encode(inputTensor);
    }

    /// <summary>
    /// Encodes audio tensor into compressed frames.
    /// </summary>
    /// <param name="x">Audio tensor to encode [batch, channels, time].</param>
    /// <returns>List of encoded frames.</returns>
    /// <exception cref="ArgumentException">Thrown when input tensor has invalid shape or channels.</exception>
    public List<EncodedFrame> Encode(Tensor x)
    {
        using var inference = torch.inference_mode();

        x = x.to(Device);
        ValidateInputTensor(x);

        long channels = x.size(1);
        long length = x.size(2);
        if (channels is <= 0 or > 2)
        {
            throw new ArgumentException($"Invalid number of channels: {channels}");
        }

        var segmentLength = SegmentLength ?? length;
        var stride = SegmentStride ?? length;

        var encodedFrames = new List<EncodedFrame>();

        for (long offset = 0; offset < length; offset += stride)
        {
            var frame = x[TensorIndex.Colon, TensorIndex.Colon, (Index)offset..(Index)Math.Min(offset + segmentLength, length)];
            encodedFrames.Add(EncodeFrame(frame, offset));
        }

        return encodedFrames;
    }

    /// <summary>
    /// Performs a complete encode-decode cycle on the input tensor.
    /// </summary>
    /// <param name="x">Input audio tensor [batch, channels, time].</param>
    /// <returns>Reconstructed audio tensor.</returns>
    public override Tensor forward(Tensor x)
    {
        var frames = Encode(x);
        return Decode(frames).slice(2, 0, x.size(-1), 1);
    }

    /// <summary>
    /// Gets or initializes the language model for enhanced compression.
    /// </summary>
    /// <returns>Configured language model instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no pre-trained model is available.</exception>
    public async Task<EncodecLanguageModel> GetLanguageModel()
    {
        if (_lmModelWeightsLoaded)
        {
            return _lm;
        }
        int numCodebooks = quantizer.NumQuantizers;

        var config = new EncodecLanguageModelConfig
        {
            CodebookSize = _config.CodebookSize,
            NumCodebooks = numCodebooks,
            Dimension = 200,
            NumHeads = 8,
            NumLayers = 5,
            PastContext = (int)(3.5f * FrameRate)
        };
        var checkpoints = new Dictionary<int, string>
        {
            [24000] = "https://dl.fbaipublicfiles.com/encodec/v0/encodec_lm_24khz-1608e3c0.th",
            [48000] = "https://dl.fbaipublicfiles.com/encodec/v0/encodec_lm_48khz-7add9fc3.th"
        };
        if (!checkpoints.TryGetValue(SampleRate, out var checkpointUrl))
        {
            throw new InvalidOperationException("No LM pre-trained for the current Encodec model.");
        }
        var loader = new TorchModelLoader();
        _lm = await loader.LoadModelAsync<EncodecLanguageModel, EncodecLanguageModelConfig>(checkpointUrl, config, new ModelLoadOptions() { ValidateModel = false });

        using (torch.no_grad())
        {
            _lm.to(Device);
        }

        _lm.eval();
        _lmModelWeightsLoaded = true;
        return _lm;
    }

    /// <summary>
    /// Loads model weights from the specified file.
    /// </summary>
    /// <param name="path">Path to the weights file.</param>
    /// <exception cref="FileNotFoundException">Thrown when weights file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown when loading weights fails.</exception>
    public void LoadWeights(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Weights not found at {path}");
        }

        try
        {
            set_default_device(CPU);
            if (Device != CPU)
            {
                using (no_grad())
                {
                    this.to(CPU);
                    _lm?.to(CPU);
                }
            }

            switch (FileUtils.DetectFileType(path))
            {
                case ModelFileType.PyTorch:
                case ModelFileType.Weights:
                    this.load_py(path);
                    break;

                case ModelFileType.SafeTensors:
                    this.load_safetensors(path);
                    break;

                case ModelFileType.Checkpoint:
                    this.load_checkpoint(path);
                    break;

                default:
                    this.load(path, false);
                    break;
            }

            if (Device != CPU)
            {
                using (torch.no_grad())
                {
                    this.to(Device);
                    _lm?.to(Device);
                    set_default_device(Device);

                }
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load weights from {path}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Sets the target bandwidth for compression.
    /// </summary>
    /// <param name="bandwidth">Target bandwidth in kbps.</param>
    /// <exception cref="ArgumentException">Thrown when bandwidth is not supported.</exception>
    public void SetTargetBandwidth(float bandwidth)
    {
        if (!_targetBandwidths.Contains(bandwidth))
        {
            throw new ArgumentException(
                $"This model doesn't support the bandwidth {bandwidth} kbps. " +
                $"Select one of [{string.Join(", ", _targetBandwidths)} kbps]");
        }
        _bandwidth = bandwidth;
        _config.Bandwidth = bandwidth;
    }

    /// <summary>
    /// Releases unmanaged and managed resources.
    /// </summary>
    /// <param name="disposing">True to dispose managed resources.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            encoder?.Dispose();
            decoder?.Dispose();
            quantizer?.Dispose();
        }
        base.Dispose(disposing);
    }

    private Tensor DecodeFrame(EncodedFrame frame)
    {
        if (frame.Codes?.IsInvalid != false)
        {
            Debug.WriteLine("Invalid frame codes in Encodec Decode");
            throw new ArgumentException("Invalid frame codes in Encodec Decode");
        }
        using (no_grad())
        {
            var emb = quantizer.Decode(frame.Codes);
            var output = decoder.forward(emb);

            if (frame.Scale is not null)
            {
                output *= frame.Scale.view(-1, 1, 1);
            }

            return output;
        }
    }

    private EncodedFrame EncodeFrame(Tensor x, long offset)
    {
        var length = x.size(-1);
        var duration = length / (float)SampleRate;

        if (_segment.HasValue && duration > _segment.Value + 1e-5)
        {
            throw new ArgumentException($"Frame duration {duration} exceeds segment size {_segment.Value}");
        }

        Tensor? scale = null;
        using var scope = NewDisposeScope();
        if (Normalize)
        {
            var mono = x.mean([1], keepdim: true);
            var volume = mono.pow(2).mean([2], keepdim: true).sqrt();
            scale = volume.add(1e-8f);
            x = x.div(scale);
            if (Math.Abs(Math.Pow(2, BitsPerCodebook) - quantizer.layers[0].CodebookSize) > 1e-5)
            {
                throw new ArgumentException("Quantizer bins must be a power of 2");
            }
            scale = scale.view(-1, 1);
        }

        using (no_grad())
        {
            var emb = encoder.forward(x);
            var codes = quantizer.Encode(emb, FrameRate, _bandwidth);

            return new EncodedFrame(codes.MoveToOuterDisposeScope(), scale?.MoveToOuterDisposeScope());
        }
    }

    private void ValidateInputTensor(Tensor x)
    {
        if (x.dim() != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input tensor [B,C,T], got shape [{string.Join(", ", x.shape)}]");
        }

        if (x.shape[1] != _config.Channels)
        {
            throw new ArgumentException(
                $"Expected {_config.Channels} channels, got {x.shape[1]}");
        }
    }

    #endregion Methods
}