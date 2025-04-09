using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Residual Vector Quantizer implementation that applies multiple quantization layers sequentially.
/// Each layer quantizes the residual error from previous layers, allowing for better reconstruction
/// quality through hierarchical/multi-scale quantization.
/// </summary>
public class ResidualVectorQuantizer : Module<Tensor, int, (Tensor quantized, Tensor codes, Tensor loss)>
{
    /// <summary>
    /// List of vector quantizer layers applied sequentially.
    /// </summary>
    public readonly ModuleList<VectorQuantizer> layers;

    /// <summary>
    /// Number of entries in each quantizer's codebook.
    /// </summary>
    private readonly int _codebookSize;

    /// <summary>
    /// EMA decay factor for codebook updates.
    /// </summary>
    private readonly float _decay;

    /// <summary>
    /// The dimension of the input vectors to quantize.
    /// </summary>
    private readonly int _dimension;

    /// <summary>
    /// Whether to initialize codebooks using k-means.
    /// </summary>
    private readonly bool _kmeansInit;

    /// <summary>
    /// Number of k-means iterations for initialization.
    /// </summary>
    private readonly int _kmeansIters;

    /// <summary>
    /// Number of quantizer layers to apply sequentially.
    /// </summary>
    private readonly int _numQuantizers;

    /// <summary>
    /// Threshold for detecting and replacing dead codes.
    /// </summary>
    private readonly int _thresholdEmaDeadCode;

    /// <summary>
    /// Initializes a new instance of the ResidualVectorQuantizer class.
    /// </summary>
    /// <param name="dimension">Dimension of input vectors to quantize.</param>
    /// <param name="numQuantizers">Number of quantizer layers to use.</param>
    /// <param name="codebookSize">Number of entries in each quantizer's codebook.</param>
    /// <param name="decay">EMA decay factor for updating codebooks (default: 0.99).</param>
    /// <param name="kmeansInit">Whether to initialize codebooks using k-means (default: true).</param>
    /// <param name="kmeansIters">Number of k-means iterations for initialization (default: 50).</param>
    /// <param name="thresholdEmaDeadCode">Threshold for detecting dead codes (default: 2).</param>
    /// <param name="device">Optional device to place the model on.</param>
    public ResidualVectorQuantizer(
        int dimension = 256,
        int numQuantizers = 8,
        int codebookSize = 1024,
        float decay = 0.99f,
        bool kmeansInit = true,
        int kmeansIters = 50,
        int thresholdEmaDeadCode = 2,
        Device device = null) : base("RVQ")
    {
        _dimension = dimension;
        _numQuantizers = numQuantizers;
        _codebookSize = codebookSize;
        _decay = decay;
        _kmeansInit = kmeansInit;
        _kmeansIters = kmeansIters;
        _thresholdEmaDeadCode = thresholdEmaDeadCode;

        var vqModules = new List<VectorQuantizer>();
        for (int i = 0; i < numQuantizers; i++)
        {
            vqModules.Add(new VectorQuantizer(
                dim: dimension,
                codebookSize: codebookSize,
                decay: decay,
                kmeansInit: kmeansInit,
                kmeansIters: kmeansIters,
                thresholdEmaDeadCode: thresholdEmaDeadCode));
        }
        layers = [.. vqModules];
        RegisterComponents();

        // Move to device if specified
        if (device != null)
        {
            this.to(device);
        }
    }

    /// <summary>
    /// Gets the number of quantizer layers in this RVQ.
    /// </summary>
    public int NumQuantizers => _numQuantizers;

    /// <summary>
    /// Decodes discrete codes back to continuous vectors.
    /// </summary>
    /// <param name="codes">Tensor of codebook indices from Encode.</param>
    /// <returns>Reconstructed continuous tensor.</returns>
    public Tensor Decode(Tensor codes)
    {
        using var scope = NewDisposeScope();

        // Ensure input is on the same device as the model
        codes = codes.to(this.parameters().First().device);
        var quantizedOut = zeros(1, device: codes.device);
        var nQ = codes.size(1);

        for (int i = 0; i < nQ; i++)
        {
            var indices = codes.select(1, i);
            var quantized = layers[i].Decode(indices);
            quantizedOut += quantized;
        }

        return quantizedOut.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Encodes input tensor to discrete codes using specified bandwidth constraint.
    /// </summary>
    /// <param name="x">Input tensor to encode.</param>
    /// <param name="frameRate">Frame rate of the input in Hz.</param>
    /// <param name="bandwidth">Target bandwidth in kbps (optional).</param>
    /// <returns>Tensor of codebook indices for each layer used.</returns>
    public Tensor Encode(Tensor x, int frameRate, float? bandwidth = null)
    {
        // Ensure input is on the same device as the model
        x = x.to(this.parameters().First().device);
        var bwPerQ = GetBandwidthPerQuantizer(frameRate);
        var nQ = _numQuantizers;

        if (bandwidth.HasValue && bandwidth.Value > 0)
        {
            // bandwidth is a thousandth of what it is, e.g. 6kbps bandwidth is 6.0
            nQ = (int)Math.Max(1, Math.Floor(bandwidth.Value * 1000 / bwPerQ));
        }

        var residual = x;
        var codes = new List<Tensor>();

        for (int i = 0; i < nQ; i++)
        {
            var (quantized, indices, _) = layers[i].forward(residual);
            residual -= quantized;
            codes.Add(indices);
        }

        return stack(codes, dim: 1).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Performs forward pass of residual vector quantization.
    /// </summary>
    /// <param name="z">Input tensor to quantize.</param>
    /// <param name="nQ">Number of quantizer layers to use (must be less than or equal to NumQuantizers).</param>
    /// <returns>
    /// Tuple containing:
    /// - Quantized tensor after applying nQ layers
    /// - Codebook indices for each layer
    /// - Loss values for each layer
    /// </returns>
    public override (Tensor quantized, Tensor codes, Tensor loss) forward(
       Tensor z, int nQ)
    {
        using var scope = NewDisposeScope();

        // Ensure input is on the same device as the model
        z = z.to(this.parameters().First().device);
        var quantizedOut = zeros_like(z, dtype: z.dtype, device: z.device);
        var residual = z.to(float32).contiguous();
        var allLosses = new List<Tensor>();
        var allIndices = new List<Tensor>();

        for (int i = 0; i < nQ; i++)
        {
            var (quantized, indices, loss) = layers[i].forward(residual);
            residual -= quantized;
            quantizedOut += quantized;

            allIndices.Add(indices);
            allLosses.Add(loss);
        }

        var outLosses = stack(allLosses);
        var codeIndices = stack(allIndices);

        return (
            quantizedOut.MoveToOuterDisposeScope(),
            codeIndices.MoveToOuterDisposeScope(),
            outLosses.MoveToOuterDisposeScope());
    }

    /// <summary>
    /// Quantizes input with bandwidth constraint and returns detailed metrics.
    /// </summary>
    /// <param name="x">Input tensor to quantize.</param>
    /// <param name="frameRate">Frame rate of the input in Hz.</param>
    /// <param name="bandwidth">Target bandwidth in kbps (optional).</param>
    /// <returns>QuantizedResult containing quantized tensor, codes, bandwidth used and loss metrics.</returns>
    public QuantizedResult QuantizeWithBandwidth(
        Tensor x, int frameRate, float? bandwidth = null)
    {
        // Ensure input is on the same device as the model
        x = x.to(this.parameters().First().device);
        var bwPerQ = GetBandwidthPerQuantizer(frameRate);
        var nQ = GetNumQuantizersForBandwidth(frameRate, bandwidth);

        var (quantized, codes, commitLoss) = forward(x, nQ);
        var bw = full([x.size(0)], nQ * bwPerQ, device: x.device);

        return new QuantizedResult
        {
            Quantized = quantized,
            Codes = codes,
            Bandwidth = bw,
            Penalty = commitLoss.mean()
        };
    }

    /// <summary>
    /// Disposes the managed resources used by the module.
    /// </summary>
    /// <param name="disposing">True to dispose managed resources, false otherwise.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            layers?.Dispose();
        }
        base.Dispose(disposing);
    }

    /// <summary>
    /// Calculates the bandwidth used per quantizer layer.
    /// </summary>
    /// <param name="frameRate">Frame rate in Hz.</param>
    /// <returns>Bandwidth in bits per second used by each quantizer layer.</returns>
    private float GetBandwidthPerQuantizer(int frameRate)
    {
        // Each quantizer encodes a frame with log2(bins) bits
        return (float)(Math.Log2(_codebookSize) * frameRate);
    }

    /// <summary>
    /// Determines how many quantizer layers to use for given bandwidth target.
    /// </summary>
    /// <param name="frameRate">Frame rate in Hz.</param>
    /// <param name="bandwidth">Target bandwidth in kbps (optional).</param>
    /// <returns>Number of quantizer layers to use.</returns>
    private int GetNumQuantizersForBandwidth(int frameRate, float? bandwidth)
    {
        var bwPerQ = GetBandwidthPerQuantizer(frameRate);
        var nQ = _numQuantizers;

        if (bandwidth.HasValue && bandwidth.Value > 0)
        {
            // bandwidth is kbps (e.g. 6.0 for 6kbps)
            nQ = (int)Math.Max(1, Math.Floor(bandwidth.Value * 1000 / bwPerQ));
        }

        return nQ;
    }

    /// <summary>
    /// Validates that input tensor has correct shape [B, C, T].
    /// </summary>
    /// <param name="x">Input tensor to validate.</param>
    /// <exception cref="ArgumentException">Thrown if tensor shape is invalid.</exception>
    private void ValidateInputShape(Tensor x)
    {
        if (x.dim() != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input tensor [B, C, T], got shape [{string.Join(", ", x.shape)}]");
        }
    }
}