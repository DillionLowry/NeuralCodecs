using NeuralCodecs.Core;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Torch.Config.Encodec;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Language Model to estimate probabilities of each codebook entry.
/// Predicts all codebooks in parallel for a given time step.
/// </summary>
public class EncodecLanguageModel : Module<Tensor, (Tensor probabilities, List<Tensor> states, int offset)>, INeuralCodec
{
    private readonly EncodecLanguageModelConfig _config;
    private readonly ModuleList<Embedding> emb;
    private readonly ModuleList<Linear> linears;
    private readonly StreamingTransformerEncoder transformer;

    /// <summary>
    /// Initialize language model with default parameters
    /// </summary>
    public EncodecLanguageModel() : this(new EncodecLanguageModelConfig())
    {
    }

    /// <summary>
    /// Initialize language model with the specified configuration
    /// </summary>
    /// <param name="config">Language model configuration</param>
    public EncodecLanguageModel(EncodecLanguageModelConfig config) : base("LanguageModel")
    {
        _config = config;

        // Validate configuration
        ValidateConfig(config);

        // Create embeddings for each codebook (adding 1 for padding index 0)
        var embeddingsList = new List<Embedding>();
        for (int i = 0; i < config.NumCodebooks; i++)
        {
            embeddingsList.Add(Embedding(config.CodebookSize + 1, config.Dimension));
        }
        emb = ModuleList(embeddingsList.ToArray());

        // Create projection layers for each codebook
        var projectionsList = new List<Linear>();
        for (int i = 0; i < config.NumCodebooks; i++)
        {
            projectionsList.Add(Linear(config.Dimension, config.CodebookSize));
        }
        linears = ModuleList(projectionsList.ToArray());

        // Create transformer encoder
        transformer = new StreamingTransformerEncoder(
            dimension: config.Dimension,
            hiddenScale: config.HiddenScale,
            numHeads: config.NumHeads,
            numLayers: config.NumLayers,
            maxPeriod: config.MaxPeriod,
            pastContext: config.PastContext,
            gelu: config.Gelu,
            normIn: config.NormIn,
            dropout: config.Dropout);

        RegisterComponents();
    }

    /// <summary>
    /// Initialize language model with explicit parameters
    /// </summary>
    /// <param name="numCodebooks">Number of codebooks</param>
    /// <param name="codebookSize">Size of each codebook</param>
    /// <param name="dimension">Hidden dimension</param>
    /// <param name="numHeads">Number of attention heads</param>
    /// <param name="numLayers">Number of transformer layers</param>
    /// <param name="pastContext">Maximum causal context length</param>
    public EncodecLanguageModel(
        int numCodebooks,
        int codebookSize = 1024,
        int dimension = 200,
        int numHeads = 8,
        int numLayers = 5,
        int pastContext = 1000) : this(new EncodecLanguageModelConfig
        {
            NumCodebooks = numCodebooks,
            CodebookSize = codebookSize,
            Dimension = dimension,
            NumHeads = numHeads,
            NumLayers = numLayers,
            PastContext = pastContext
        })
    {
    }

    /// <summary>
    /// Configuration for this language model
    /// </summary>
    public IModelConfig Config => _config;

    /// <summary>
    /// Compress a sequence of codes using the language model
    /// </summary>
    /// <param name="codes">Code indices with shape [B, K, T]</param>
    /// <param name="outputStream">Stream to write compressed data to</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
    public void Compress(Tensor codes, Stream outputStream)
    {
        if (codes is null || codes.IsInvalid)
        {
            throw new ArgumentNullException(nameof(codes), "Codes tensor cannot be null or invalid");
        }

        ArgumentNullException.ThrowIfNull(outputStream);

        if (!outputStream.CanWrite)
        {
            throw new ArgumentException("Output stream must be writable", nameof(outputStream));
        }

        using var scope = NewDisposeScope();

        // Move to CPU for compression
        codes = codes.cpu();

        // Get dimensions
        var (batchSize, numCodebooks, timeSteps) = (
            codes.size(0),
            codes.size(1),
            codes.size(2)
        );

        // Initialize language model state
        List<Tensor> states = null;

        // Initialize arithmetic coder
        using var coder = new ArithmeticCoder(outputStream);

        // Process one time step at a time
        for (int t = 0; t < timeSteps; t++)
        {
            // Get probability distribution for next step
            var (probabilities, newStates, _) = PredictNextStep(codes, states, t);
            states = newStates;

            // Encode each codebook independently
            for (int k = 0; k < numCodebooks; k++)
            {
                // Get probability distribution for current codebook
                var pdf = probabilities[0, .., k, 0];

                // Convert PDF to quantized CDF
                var qCdf = ArithmeticCodingUtils.BuildStableQuantizedCdf(
                    pdf, coder.TotalRangeBits);

                // Get actual symbol to encode
                var symbol = codes[0, k, t].item<int>();

                // Encode the symbol
                coder.Push(symbol, qCdf);
            }
        }
    }

    /// <summary>
    /// Decompress data to code indices using the language model
    /// </summary>
    /// <param name="inputStream">Stream containing compressed data</param>
    /// <param name="numCodebooks">Number of codebooks</param>
    /// <param name="timeSteps">Number of time steps to decode</param>
    /// <param name="device">Device to place output tensor on</param>
    /// <returns>Decompressed code indices with shape [1, K, T]</returns>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
    public Tensor Decompress(Stream inputStream, int numCodebooks, int timeSteps, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(inputStream);

        if (!inputStream.CanRead)
        {
            throw new ArgumentException("Input stream must be readable", nameof(inputStream));
        }

        if (numCodebooks <= 0)
        {
            throw new ArgumentException($"Number of codebooks must be positive, got {numCodebooks}");
        }

        if (timeSteps <= 0)
        {
            throw new ArgumentException($"Number of time steps must be positive, got {timeSteps}");
        }

        // Use CPU by default for decompression
        device ??= CPU;

        // Initialize output tensor
        var codes = zeros(1, numCodebooks, timeSteps, dtype: int64, device: device);

        // Initialize language model state
        List<Tensor> states = null;

        // Initialize arithmetic decoder
        using var decoder = new ArithmeticDecoder(inputStream);

        // Process one time step at a time
        for (int t = 0; t < timeSteps; t++)
        {
            // Get probability distribution for next step
            var (probabilities, newStates, _) =
                t > 0 ? PredictNextStep(codes, states, t) : forward(codes.narrow(2, 0, 1), null, 0);
            states = newStates;

            // Decode each codebook independently
            for (int k = 0; k < numCodebooks; k++)
            {
                // Get probability distribution for current codebook
                var pdf = probabilities[0, .., k, 0];

                // Convert PDF to quantized CDF
                var qCdf = ArithmeticCodingUtils.BuildStableQuantizedCdf(
                    pdf, decoder.TotalRangeBits);

                // Decode symbol
                var symbol = decoder.Pull(qCdf);

                if (!symbol.HasValue)
                {
                    throw new EndOfStreamException("End of stream reached before decoding completed");
                }

                // Store decoded symbol
                codes[0, k, t] = symbol.Value;
            }
        }

        return codes;
    }

    /// <summary>
    /// Forward pass for standard inference
    /// </summary>
    /// <param name="indices">Input indices with shape [B, n_q, T]</param>
    /// <returns>Probabilities tensor with shape [B, codebook_size, n_q, T]</returns>
    public override (Tensor probabilities, List<Tensor> states, int offset) forward(Tensor indices)
    {
        return forward(indices, null, 0);
    }

    /// <summary>
    /// Forward pass with streaming support
    /// </summary>
    /// <param name="indices">Input indices with shape [B, n_q, T]</param>
    /// <param name="states">Previous states for streaming</param>
    /// <param name="offset">Current time offset</param>
    /// <returns>Tuple of (probabilities, new_states, new_offset)</returns>
    public (Tensor probabilities, List<Tensor> states, int offset) forward(
        Tensor indices, List<Tensor>? states, int offset = 0)
    {
        using var scope = NewDisposeScope();

        // Validate input
        ValidateInput(indices);

        var (batchSize, numCodebooks, timeSteps) = (
            indices.size(0),
            indices.size(1),
            indices.size(2)
        );

        // Combine embeddings from all codebooks - input shape: [B, K, T]
        var input = zeros(batchSize, timeSteps, _config.Dimension, device: indices.device);

        for (int k = 0; k < Math.Min(numCodebooks, emb.Count); k++)
        {
            var indices_k = indices.select(1, k);
            input.add_(emb[k].forward(indices_k));
        }

        // Process through transformer
        var (transformerOutput, newStates, newOffset) = transformer.forward(input, states, offset);

        // Generate projections for each codebook
        var logitsList = new List<Tensor>();
        for (int k = 0; k < Math.Min(numCodebooks, linears.Count); k++)
        {
            logitsList.Add(linears[k].forward(transformerOutput));
        }

        // Stack projections and permute to [B, card, K, T]
        var logits = stack(logitsList, dim: 1);
        var permuted = logits.permute(0, 3, 1, 2);

        // Apply softmax to get probabilities
        var probabilities = softmax(permuted, dim: 1).MoveToOuterDisposeScope();

        return (probabilities, newStates, newOffset);
    }

    /// <summary>
    /// Load pre-trained weights from a file
    /// </summary>
    /// <param name="path">Path to weights file</param>
    public void LoadWeights(string path)
    {
        // Validate path
        if (string.IsNullOrEmpty(path))
        {
            throw new ArgumentException("Path cannot be null or empty", nameof(path));
        }

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Weights file not found at {path}");
        }

        try
        {
            if (path.EndsWith(".th") || path.EndsWith(".pth"))
            {
                // Load PyTorch format
                this.load_py(path);
            }
            else
            {
                // Default TorchSharp format
                this.load(path);
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load weights from {path}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Generate a single step prediction for compression
    /// </summary>
    /// <param name="indices">Current indices [B, K, T]</param>
    /// <param name="states">Previous transformer states</param>
    /// <param name="timeStep">Current time step</param>
    /// <returns>Probabilities for next step [B, card, K, 1]</returns>
    public (Tensor probabilities, List<Tensor> states, int nextTimeStep) PredictNextStep(
        Tensor indices, List<Tensor> states, int timeStep)
    {
        // Extract the current timestep's indices - shape [B, K]
        var currentIndices = indices.select(2, timeStep);

        // Add dimension to make it [B, K, 1]
        var expandedIndices = currentIndices.unsqueeze(2);

        // Forward pass
        var (probabilities, newStates, newOffset) = forward(expandedIndices, states, timeStep);

        return (probabilities, newStates, newOffset);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            emb?.Dispose();
            linears?.Dispose();
            transformer?.Dispose();
        }

        base.Dispose(disposing);
    }

    private static void ValidateConfig(EncodecLanguageModelConfig config)
    {
        if (config.NumCodebooks <= 0)
        {
            throw new ArgumentException($"Number of codebooks must be positive, got {config.NumCodebooks}");
        }

        if (config.CodebookSize <= 0)
        {
            throw new ArgumentException($"Codebook size must be positive, got {config.CodebookSize}");
        }

        if (config.Dimension <= 0)
        {
            throw new ArgumentException($"Dimension must be positive, got {config.Dimension}");
        }

        if (config.NumHeads <= 0)
        {
            throw new ArgumentException($"Number of heads must be positive, got {config.NumHeads}");
        }

        if (config.Dimension % config.NumHeads != 0)
        {
            throw new ArgumentException(
                $"Dimension ({config.Dimension}) must be divisible by number of heads ({config.NumHeads})");
        }

        if (config.NumLayers <= 0)
        {
            throw new ArgumentException($"Number of layers must be positive, got {config.NumLayers}");
        }

        if (config.PastContext <= 0)
        {
            throw new ArgumentException($"Past context must be positive, got {config.PastContext}");
        }

        if (config.Dropout is < 0 or > 1)
        {
            throw new ArgumentException($"Dropout must be between 0 and 1, got {config.Dropout}");
        }
    }

    private void ValidateInput(Tensor indices)
    {
        if (indices is null || indices.IsInvalid)
        {
            throw new ArgumentNullException(nameof(indices), "Input tensor cannot be null or invalid");
        }

        if (indices.dim() != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input tensor [B, K, T], got shape {string.Join("×", indices.shape)}");
        }
    }
}