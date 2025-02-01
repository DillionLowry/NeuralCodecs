using NeuralCodecs.Core;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Utils;
using NeuralCodecs.Torch.Config.SNAC;
using NeuralCodecs.Torch.Modules.SNAC;
using NeuralCodecs.Torch.Utils;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using DeviceType = NeuralCodecs.Core.Configuration.DeviceType;

namespace NeuralCodecs.Torch.Models;

/// <summary>
/// Implements a neural audio codec that combines encoding, quantization, and decoding
/// for efficient audio compression and reconstruction.
/// </summary>
public partial class SNAC : Module<Tensor, (Tensor audio, List<Tensor> codes)>, INeuralCodec
{
    private readonly int _hopLength;
    private torch.Device _device => TorchUtils.GetDevice(_config.Device);
    private readonly Encoder encoder;
    private readonly ResidualVectorQuantizer quantizer;
    private readonly Decoder decoder;

    private SNACConfig _config;
    public IModelConfig Config => _config;

    /// <summary>
    /// Initializes a new instance of the SNAC neural audio codec.
    /// </summary>
    /// <param name="config">Configuration parameters for the codec</param>
    public SNAC(SNACConfig config) : base("SNAC")
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

        this.to(_device);
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
        var length = audioData.shape[^1];
        audioData = Preprocess(audioData);

        var z = encoder.forward(audioData);
        var (zQ, codes) = quantizer.forward(z);
        var audioHat = decoder.forward(zQ);

        // Trim to original length
        audioHat = audioHat.narrow(-1, 0, length);

        return (audioHat.MoveToOuterDisposeScope(), codes);
    }

    /// <summary>
    /// Encodes audio data into a list of quantized codes.
    /// </summary>
    /// <param name="audioData">Input audio tensor</param>
    /// <returns>A list of quantized code tensors</returns>
    public List<Tensor> Encode(Tensor audioData)
    {
        audioData = Preprocess(audioData);
        var z = encoder.forward(audioData);
        var (_, codes) = quantizer.forward(z);

        return codes;
    }

    /// <summary>
    /// Encodes audio data into a list of quantized codes.
    /// </summary>
    /// <param name="audioData">Input audio samples</param>
    /// <returns>A list of quantized code arrays</returns>
    public List<float[]> Encode(float[] audioData)
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
        var (_, codes) = quantizer.forward(z);

        // Convert codes to float arrays
        var codeArrays = codes.ConvertAll(code => code.cpu().detach().to(torch.float32).data<float>().ToArray());

        return codeArrays;
    }

    /// <summary>
    /// Decodes a list of quantized codes back into audio.
    /// </summary>
    /// <param name="codes">List of quantized code tensors</param>
    /// <returns>Reconstructed audio tensor</returns>
    public Tensor Decode(List<Tensor> codes)
    {
        using var scope = torch.NewDisposeScope();

        var zQ = quantizer.FromCodes(codes);
        var audio = decoder.forward(zQ);

        return audio.MoveToOuterDisposeScope();
    }


    /// <summary>
    /// Decodes a list of quantized codes back into audio.
    /// </summary>
    /// <param name="codes">List of quantized code arrays</param>
    /// <returns>Reconstructed audio samples</returns>
    public float[] Decode(List<float[]> codes)
    {
        ArgumentNullException.ThrowIfNull(codes);

        if (codes.Count == 0 || codes.Any(code => code == null))
        {
            throw new ArgumentException("Codes list cannot be empty or contain null arrays", nameof(codes));
        }

        using var scope = torch.NewDisposeScope();

        // Convert code arrays to tensors with int64 dtype for indices
        var codeTensors = codes.ConvertAll(code => torch.tensor(code, dtype: torch.int64).reshape(1, -1).to(_device));

        var zQ = quantizer.FromCodes(codeTensors);
        var audio = decoder.forward(zQ);

        // Convert output tensor to float array
        return audio.cpu().detach().data<float>().ToArray();
    }

    /// <summary>
    /// Loads pre-trained model weights from a file.
    /// </summary>
    /// <param name="path">Path to the weights file</param>
    /// <exception cref="FileNotFoundException">Thrown when the weights file is not found</exception>
    /// <exception cref="InvalidOperationException">Thrown when loading weights fails</exception>
    public void LoadWeights(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"PyTorch weights not found at {path}");

        try
        {
            switch (FileUtil.DetectFileType(path))
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
                    this.load(path);
                    break;
            }
        }
        catch (Exception ex) when (ex is not (FileNotFoundException))
        {
            // TODO
            // Write state_dict to console
            foreach (var (key, value) in this.state_dict())
            {
                Console.WriteLine($"{key}: {value}");
            }

            throw new InvalidOperationException($"Failed to load PyTorch weights from {path} {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Processes audio data through the codec pipeline, optionally resampling if needed.
    /// </summary>
    /// <param name="audioData">Raw audio samples</param>
    /// <param name="sampleRate">Sample rate of the input audio</param>
    /// <returns>Processed audio samples</returns>
    /// <exception cref="ArgumentException">Thrown when audio data is null or empty</exception>
    public float[] ProcessAudio(float[] audioData, int sampleRate)
    {
        if (audioData == null || audioData.Length == 0)
            throw new ArgumentException("Audio data cannot be empty", nameof(audioData));

        using var scope = torch.NewDisposeScope();

        // Resample if needed
        if (sampleRate != _config.SamplingRate)
        {
            audioData = ResampleAudio(audioData, sampleRate, _config.SamplingRate);
        }

        // Convert to tensor with correct shape
        var inputTensor = torch.tensor(audioData, dtype: float32)
                              .reshape(1, 1, -1)
                              .to(_device);

        // Process through model
        using (torch.inference_mode())
        {
            var (outputTensor, _) = this.forward(inputTensor.MoveToOuterDisposeScope());
            return outputTensor.cpu()
                             .detach()
                             .data<float>()
                             .ToArray();
        }
    }

    //private torch.Device GetDevice()
    //{
    //    return _config.Device?.Type switch
    //    {
    //        DeviceType.CPU => torch.CPU,
    //        DeviceType.CUDA when cuda.is_available() => torch.CUDA,
    //        DeviceType.CUDA => throw new InvalidOperationException("CUDA requested but not available"),
    //        _ => torch.CPU
    //    };
    //}

    private static float[] ResampleAudio(float[] input, int sourceSampleRate, int targetSampleRate)
    {
        var ratio = (double)targetSampleRate / sourceSampleRate;
        var outputLength = (int)(input.Length * ratio);
        var output = new float[outputLength];

        for (int i = 0; i < outputLength; i++)
        {
            var position = i / ratio;
            var index = (int)position;
            var fraction = position - index;

            if (index >= input.Length - 1)
            {
                output[i] = input[input.Length - 1];
            }
            else
            {
                output[i] = (float)((1 - fraction) * input[index] +
                           (fraction * input[index + 1]));
            }
        }

        return output;
    }
}