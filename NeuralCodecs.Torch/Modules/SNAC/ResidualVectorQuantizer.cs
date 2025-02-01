using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.SNAC;

/// <summary>
/// Implements Residual Vector Quantization (RVQ) for neural compression.
/// RVQ uses multiple codebooks to progressively quantize input vectors,
/// with each stage operating on the residual from previous stages.
/// </summary>
public class ResidualVectorQuantizer : Module<Tensor, (Tensor zQ, List<Tensor> codes)>
{
    /// <summary>
    /// Number of codebooks used in the quantization process
    /// </summary>
    public int NumCodebooks { get; }

    /// <summary>
    /// Dimension of each codebook entry
    /// </summary>
    public int CodebookDim { get; }

    /// <summary>
    /// Size of each codebook (number of entries)
    /// </summary>
    public int CodebookSize { get; }

    /// <summary>
    /// List of vector quantizers, one for each codebook
    /// </summary>
    private readonly ModuleList<VectorQuantizer> quantizers;

    /// <summary>
    /// Initializes a new instance of ResidualVectorQuantizer
    /// </summary>
    /// <param name="inputDim">Dimension of input vectors</param>
    /// <param name="codebookSize">Number of entries in each codebook</param>
    /// <param name="codebookDim">Dimension of codebook entries</param>
    /// <param name="vqStrides">Strides for each vector quantizer</param>
    public ResidualVectorQuantizer(
        int inputDim = 512,
        int codebookSize = 1024,
        int codebookDim = 8,
        int[] vqStrides = null) : base("RVQ")
    {
        vqStrides ??= [1, 1, 1, 1];

        NumCodebooks = vqStrides.Length;
        CodebookDim = codebookDim;
        CodebookSize = codebookSize;

        VectorQuantizer[] quantizerList = vqStrides.Select(stride =>
            new VectorQuantizer(inputDim, codebookSize, codebookDim, stride)).ToArray();

        quantizers = ModuleList(quantizerList);
        RegisterComponents();
    }

    /// <summary>
    /// Performs forward pass of RVQ encoding
    /// </summary>
    /// <param name="z">Input tensor to quantize</param>
    /// <returns>
    /// Tuple containing:
    /// - zQ: Quantized tensor
    /// - codes: List of codebook indices for each quantization stage
    /// </returns>
    public override (Tensor zQ, List<Tensor> codes) forward(Tensor z)
    {
        using var scope = NewDisposeScope();
        z = z.to(float32).contiguous();
        var zQ = zeros_like(z, dtype: z.dtype, device: z.device);
        var residual = z.clone();
        var codes = new List<Tensor>();

        // Apply each quantizer to the residual
        for (int i = 0; i < quantizers.Count; i++)
        {
            var (zQi, indicesI) = quantizers[i].forward(residual);

            // Add quantized values and update residual
            zQ = add(zQ, zQi, alpha: 1.0f);
            residual = sub(residual, zQi);

            codes.Add(indicesI.clone().MoveToOuterDisposeScope());
        }

        return (zQ.MoveToOuterDisposeScope(), codes);
    }

    /// <summary>
    /// Reconstructs tensor from list of codebook indices
    /// </summary>
    /// <param name="codes">List of codebook indices from encoding</param>
    /// <returns>Reconstructed tensor</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when number of provided codes doesn't match number of codebooks
    /// </exception>
    public Tensor FromCodes(List<Tensor> codes)
    {
        using var scope = NewDisposeScope();
        if (codes.Count != NumCodebooks)
            throw new ArgumentException($"Expected {NumCodebooks} codebooks but got {codes.Count}");

        var zQ = zeros(1, dtype: codes[0].dtype, device: codes[0].device);
        bool first = true;

        for (int i = 0; i < NumCodebooks; i++)
        {
            // Decode codes to embeddings
            var zPi = quantizers[i].DecodeCode(codes[i]);

            // Project to output space
            var zQi = quantizers[i].OutProj(zPi);

            // Apply stride if needed
            if (quantizers[i].Stride > 1)
            {
                zQi = zQi.repeat_interleave(quantizers[i].Stride, dim: -1);
            }

            if (first)
            {
                zQ = zQi;
                first = false;
            }
            else
            {
                zQ = add(zQ, zQi, alpha: 1.0f);
            }
        }

        return zQ.MoveToOuterDisposeScope();
    }
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            quantizers?.Dispose();
        }
        base.Dispose(disposing);
    }
}