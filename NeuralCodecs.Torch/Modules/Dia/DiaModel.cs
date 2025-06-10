using NeuralCodecs.Core;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Utils;
using NeuralCodecs.Torch.Config.Dia;
using NeuralCodecs.Torch.Utils;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Dia;

public class DiaModel : Module<Tensor, EncoderInferenceState, Tensor>, INeuralCodec
{
    private readonly DiaConfig _config;
    private readonly Encoder encoder;
    private readonly Decoder decoder;
    private Device _device;

    public DiaModel(DiaConfig config) : this(config, ScalarType.Float32, TorchUtils.GetDevice(config.Device))
    {
    }

    /// <summary>
    /// Creates a new Dia model.
    /// </summary>
    /// <param name="config">Model configuration</param>
    /// <param name="computeDtype">Computation data type</param>
    public DiaModel(DiaConfig config, ScalarType computeDtype, Device device)
        : base(nameof(DiaModel))
    {
        _config = config;
        encoder = new Encoder(config, computeDtype);
        decoder = new Decoder(config, computeDtype);
        _device = device;
        RegisterComponents();
    }

    /// <summary>
    /// Gets the encoder component of the Dia model.
    /// </summary>
    public Encoder Encoder => encoder;

    /// <summary>
    /// Gets the decoder component of the Dia model.
    /// </summary>
    public Decoder Decoder => decoder;

    /// <summary>
    /// Gets the model configuration.
    /// </summary>
    public IModelConfig Config => _config;

    /// <summary>
    /// Processes the input tensor through the encoder and returns the resulting tensor.
    /// </summary>
    /// <param name="x">The input tensor to be processed by the encoder.</param>
    /// <param name="state">The inference state used to manage the encoder's internal state during processing.</param>
    /// <returns>A tensor representing the output of the encoder after processing the input tensor.</returns>
    public override Tensor forward(Tensor x, EncoderInferenceState state) => encoder.forward(x, state);

    /// <summary>
    /// Loads model weights from the specified file path.
    /// </summary>
    /// <remarks>This method supports multiple file formats, including PyTorch, SafeTensors, and
    /// Checkpoint files. If the model is not currently on the CPU, it will temporarily move to the CPU for
    /// loading. After loading, the model will be moved back to its configured device, if applicable.</remarks>
    /// <param name="path">The file path to the weights file. The file must exist and be in a supported format.</param>
    /// <exception cref="FileNotFoundException">Thrown if the file specified by <paramref name="path"/> does not exist.</exception>
    /// <exception cref="InvalidOperationException">Thrown if an error occurs while loading the weights, such as an
    /// unsupported file format or a failure during processing.</exception>
    public void LoadWeights(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Weights not found at {path}");
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
                    load(path, false);
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
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load weights from {path}: {ex.Message}", ex);
        }
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            encoder.Dispose();
            decoder.Dispose();
        }

        base.Dispose(disposing);
    }
}