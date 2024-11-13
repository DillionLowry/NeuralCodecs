using NeuralCodecs.Torch.SNAC;
using NeuralCodecs.Torch.Utils;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static NeuralCodecs.Torch.SNAC.SNAC;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch;

public partial class SNAC
{
    public static bool UsePyBridge { get; set; } = true;

    public static SNAC LoadModel(string path, Device device = null)
    {
        device ??= torch.CPU;

        try
        {
            // Verify files exist
            var configPath = Path.Combine(path, "config.json");

            if (!File.Exists(configPath))
                throw new FileNotFoundException($"Config file not found at {configPath}");

            Console.WriteLine($"Loading config from {path}");
            var config = LoadConfig(configPath);

            // Create model on CPU first
            var model = new SNAC(
                samplingRate: config.SamplingRate,
                encoderDim: config.EncoderDim,
                encoderRates: config.EncoderRates,
                latentDim: null,
                decoderDim: config.DecoderDim,
                decoderRates: config.DecoderRates,
                attnWindowSize: config.AttnWindowSize,
                codebookSize: config.CodebookSize,
                codebookDim: config.CodebookDim,
                vqStrides: config.VQStrides,
                noise: config.Noise,
                depthwise: config.Depthwise
            );
            // save state_dict keys to file
            var dict = model.state_dict();
            foreach (var kvp in dict)
            {
                //shapetracker.Track(kvp.Key, kvp.Value, "state_dict");
            }

            if (UsePyBridge)
            {
                Console.WriteLine("Using PyBridge to load model weights");
                var modelPath = Path.Combine(@"T:\Models\SNAC", "pytorch_model_24khz.bin");
                if (!File.Exists(modelPath))
                    throw new FileNotFoundException($"Model weights not found at {modelPath}");
                Console.WriteLine($"Loading weights from {modelPath}");

                // Load using PyBridge's load_py
                model.load_py(modelPath);
            }
            else
            {
                Console.WriteLine("Using TorchSharp to load model weights");
                var modelPath = Path.Combine(path, "model_weights_24khz_wo.dat");
                // Load using TorchSharp's native format
                model.load(modelPath);
            }
            model.PrintModelStructure();
            // Move to device after loading
            if (device != null && device.type != DeviceType.CPU)
            {
                model.to(device);
            }

            model.eval();
            return model;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading model from {path}");
            Console.WriteLine($"Exception details: {ex}");
            //shapetracker.DumpShapes();
            throw;
        }
    }

    public static void ConvertPyTorchToTorchSharp(string inputPath, string outputPath)
    {
        try
        {
            Console.WriteLine($"Converting model from {inputPath} to {outputPath}");

            // Load config and create model
            var configPath = Path.Combine(inputPath, "config.json");
            var config = LoadConfig(configPath);

            var model = new SNAC(
                samplingRate: config.SamplingRate,
                encoderDim: config.EncoderDim,
                encoderRates: config.EncoderRates,
                latentDim: null,
                decoderDim: config.DecoderDim,
                decoderRates: config.DecoderRates,
                attnWindowSize: config.AttnWindowSize,
                codebookSize: config.CodebookSize,
                codebookDim: config.CodebookDim,
                vqStrides: config.VQStrides,
                noise: config.Noise,
                depthwise: config.Depthwise
            );

            // Print initial model structure
            Console.WriteLine("\nInitial model structure:");
            foreach (var (name, param) in model.named_parameters())
            {
                Console.WriteLine($"{name}: {string.Join(",", param.shape)} | {param.dtype}");
            }

            // Load PyTorch weights
            var modelPath = Path.Combine(inputPath, "pytorch_model.bin");
            Console.WriteLine("\nLoading PyTorch weights...");
            var loadedParams = new Dictionary<string, bool>();
            model.load_py(modelPath, strict: false, loadedParameters: loadedParams);

            // Print which parameters were loaded
            Console.WriteLine("\nLoaded parameters:");
            foreach (var (name, loaded) in loadedParams)
            {
                Console.WriteLine($"{name}: {(loaded ? "Loaded" : "Not loaded")}");
            }

            // Verify all parameters are in float32
            Console.WriteLine("\nVerifying parameter types...");
            foreach (var (name, param) in model.named_parameters())
            {
                if (param.dtype != torch.float32)
                {
                    Console.WriteLine($"Converting {name} from {param.dtype} to float32");
                    using (torch.no_grad())
                    {
                        param.copy_(param.to(torch.float32));
                    }
                }
            }

            // Save in TorchSharp format
            Console.WriteLine($"\nSaving converted model to {outputPath}");
            var outputDir = Path.GetDirectoryName(outputPath);
            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);

            model.save(outputPath);

            // Verify the saved model
            Console.WriteLine("\nVerifying saved model...");
            var testModel = new SNAC(
                samplingRate: config.SamplingRate,
                encoderDim: config.EncoderDim,
                encoderRates: config.EncoderRates,
                latentDim: null,
                decoderDim: config.DecoderDim,
                decoderRates: config.DecoderRates,
                attnWindowSize: config.AttnWindowSize,
                codebookSize: config.CodebookSize,
                codebookDim: config.CodebookDim,
                vqStrides: config.VQStrides,
                noise: config.Noise,
                depthwise: config.Depthwise
            );

            var verifyParams = new Dictionary<string, bool>();
            testModel.load(outputPath, loadedParameters: verifyParams);

            Console.WriteLine("\nVerification results:");
            foreach (var (name, loaded) in verifyParams)
            {
                Console.WriteLine($"{name}: {(loaded ? "Verified" : "Failed")}");
            }

            Console.WriteLine("\nFinal model parameters:");
            foreach (var (name, param) in testModel.named_parameters())
            {
                Console.WriteLine($"{name}: {string.Join(",", param.shape)} | {param.dtype}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error converting model: {ex}");
            throw;
        }
    }

    private static SNACConfig LoadConfig(string path)
    {
        var json = File.ReadAllText(path);
        Console.WriteLine(json);
        var config = System.Text.Json.JsonSerializer.Deserialize<SNACConfig>(json);

        return config;
    }

    // Helper to get named parameters
    public Dictionary<string, Tensor> state_dict()
    {
        var stateDict = new Dictionary<string, Tensor>();

        // Add encoder parameters
        foreach (var (name, param) in encoder.named_parameters())
        {
            stateDict[$"encoder.{name}"] = param.to(torch.float32);
        }

        // Add quantizer parameters
        foreach (var (name, param) in quantizer.named_parameters())
        {
            stateDict[$"quantizer.{name}"] = param.to(torch.float32);
        }

        // Add decoder parameters
        foreach (var (name, param) in decoder.named_parameters())
        {
            stateDict[$"decoder.{name}"] = param.to(torch.float32);
        }

        return stateDict;
    }

    // Helper to trace model structure
    public void PrintModelStructure()
    {
        Console.WriteLine("Model Structure:");
        foreach (var (name, param) in named_parameters())
        {
            if (param is Parameter p)
            {
                Console.WriteLine($"{name}: {string.Join(", ", p.shape)} | {p.dtype}");
            }
        }
    }
}

/// <summary>
/// Implements a neural audio codec that combines encoding, quantization, and decoding
/// for efficient audio compression and reconstruction.
/// </summary>
public partial class SNAC : Module<Tensor, (Tensor audio, List<Tensor> codes)>
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

    /// <summary>
    /// Preprocessing method for input audio
    /// </summary>
    /// <param name="audioData">Raw audio tensor</param>
    /// <returns>Padded and normalized audio tensor</returns>
    private Tensor Preprocess(Tensor audioData)
    {
        var length = audioData.size(-1);

        // Calculate LCM (Least Common Multiple)
        long lcm = MathUtils.LCM(vqStrides[0], this.attnWindowSize ?? 1);
        long padTo = hopLength * lcm;

        // Calculate right padding
        long rightPad = (long)(Math.Ceiling((double)length / padTo) * padTo) - length;

        audioData = nn.functional.pad(audioData, (0, rightPad));

        return audioData;
    }

    /// <summary>
    /// Loads a pretrained SNAC model from a repository
    /// </summary>
    /// <param name="repoId">Repository identifier</param>
    /// <param name="kwargs">Additional loading parameters</param>
    /// <returns>Initialized SNAC model in evaluation mode</returns>
    /// <exception cref="Exception">Thrown when model loading fails</exception>
    public static SNAC FromPretrained(string repoId, Dictionary<string, object> kwargs = null)
    {
        try
        {
            // Load model
            Console.WriteLine($"Loading model from {repoId}");
            var model = LoadModel(repoId);
            model.eval(); // Set to evaluation mode before use
            return model;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading model from {repoId}");
            Console.WriteLine(ex.Message);
            throw new Exception($"Error loading model from {repoId}", ex);
        }
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
}