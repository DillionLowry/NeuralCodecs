using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

/// <summary>
/// Implements Residual Vector Quantization (RVQ) for neural audio compression.
/// Introduced in SoundStream: An end-to-end neural audio codec.
/// https://arxiv.org/abs/2107.03312
/// </summary>
public class ResidualVectorQuantizer : Module<Tensor, (Tensor quantized, Tensor codes, Tensor latents, Tensor commitmentLoss, Tensor codebookLoss)>
{
    private readonly int _nCodebooks;
    private readonly int _codebookSize;
    private readonly int _codebookDim;
    private readonly float _quantizerDropout;
    private readonly ModuleList<VectorQuantizer> quantizers;

    /// <summary>
    /// Initializes a new instance of ResidualVectorQuantize
    /// </summary>
    /// <param name="inputDim">Dimension of input vectors</param>
    /// <param name="nCodebooks">Number of codebooks to use</param>
    /// <param name="codebookSize">Size of each codebook</param>
    /// <param name="codebookDim">Dimension for each codebook (single int or array)</param>
    /// <param name="quantizerDropout">Dropout probability for quantizers</param>
    public ResidualVectorQuantizer(
        int inputDim = 512,
        int nCodebooks = 9,
        int codebookSize = 1024,
        int codebookDim = 8,
        float quantizerDropout = 0.0f) : base("RVQ")
    {
        _nCodebooks = nCodebooks;
        _codebookSize = codebookSize;
        _codebookDim = codebookDim;
        _quantizerDropout = quantizerDropout;

        var quantizerList = new VectorQuantizer[_nCodebooks];
        for (int i = 0; i < _nCodebooks; i++)
        {
            quantizerList[i] = new VectorQuantizer(
                inputDim,
                _codebookSize,
                codebookDim
            );
        }
        quantizers = ModuleList(quantizerList);
        RegisterComponents();
    }

    public override (Tensor quantized, Tensor codes, Tensor latents, Tensor commitmentLoss, Tensor codebookLoss) forward(Tensor z)
    {
        using var scope = NewDisposeScope();
        var residual = z.clone();
        Tensor zQ = torch.zeros_like(z);
        var codes = new List<Tensor>();
        var latents = new List<Tensor>();

        for (int i = 0; i < quantizers.Count; i++)
        {
            // Process through each quantizer
            var (zQI, commitLossI, codebookLossI, codesI, latentsI) = quantizers[i].forward(residual);

            // Add to accumulated quantized tensor and update residual
            zQ.add_(zQI);
            residual.sub_(zQI);

            // Store codebook indices and latents for return
            codes.Add(codesI);
            latents.Add(latentsI);

            // Dispose intermediate tensors we don't need to return
            commitLossI.Dispose();
            codebookLossI.Dispose();
            zQI.Dispose();
        }

        // Stack the per-codebook tensors along a new dimension
        var stackedCodes = torch.stack(codes.ToArray(), 1);
        var catLatents = torch.cat(latents.ToArray(), 1);

        // Create dummy loss tensors (placeholders in this method)
        var dummyCommitmentLoss = torch.tensor(0.0f);
        var dummyCodebookLoss = torch.tensor(0.0f);

        // Clear lists to help GC
        codes.Clear();
        latents.Clear();

        // Dispose residual as we no longer need it
        residual.Dispose();

        return (
            zQ.MoveToOuterDisposeScope(),
            stackedCodes.MoveToOuterDisposeScope(),
            catLatents.MoveToOuterDisposeScope(),
            dummyCommitmentLoss.MoveToOuterDisposeScope(),
            dummyCodebookLoss.MoveToOuterDisposeScope()
        );
    }

    public (Tensor quantized, Tensor codes, Tensor latents, Tensor commitmentLoss, Tensor codebookLoss)
        forward(Tensor z, int? nQuantizers)
    {
        if (nQuantizers is null)
        {
            return forward(z);
        }
        using var scope = NewDisposeScope();

        var zQ = zeros_like(z, dtype: z.dtype, device: z.device);
        var residual = z.contiguous().clone();
        var commitmentLoss = tensor(0.0f, device: z.device);
        var codebookLoss = tensor(0.0f, device: z.device);
        var codebookIndices = new List<Tensor>();
        var latentsList = new List<Tensor>();

        // Handle number of quantizers and dropout during training
        var batchSize = z.size(0);
        Tensor nQuantizersPerBatch = null;

        if (training)
        {
            // Initialize with max number of quantizers
            nQuantizersPerBatch = full(batchSize, _nCodebooks + 1).to(z.device);

            if (_quantizerDropout > 0)
            {
                // Create random dropout pattern for batch
                using var dropout = randint(1, _nCodebooks + 1, batchSize).to(z.device);
                var nDropout = (int)(batchSize * _quantizerDropout);

                if (nDropout > 0)
                {
                    // Apply dropout to first n items in batch
                    nQuantizersPerBatch.narrow(0, 0, nDropout).copy_(dropout.narrow(0, 0, nDropout));
                }
            }
        }
        else
        {
            // During inference, use fixed number of quantizers if specified
            nQuantizersPerBatch = full(batchSize, nQuantizers ?? _nCodebooks).to(z.device);
        }

        // Apply each quantizer in sequence
        for (int i = 0; i < _nCodebooks; i++)
        {
            // During inference, stop if we've reached the requested number of quantizers
            if (!training && nQuantizers.HasValue && i >= nQuantizers.Value)
            {
                break;
            }

            // Get output from this quantizer
            var (zqi, commitmentLossi, codebookLossi, indicesi, zei) =
                quantizers[i].forward(residual);

            // Create mask for this quantizer
            using var mask = full(batchSize, i, device: z.device).lt(nQuantizersPerBatch);

            // Apply mask along batch dimension and expand to match tensor dimensions
            using var expandedMask = mask.view(batchSize, 1, 1).expand_as(zqi);

            // Compute the masked quantizer output
            using var maskedOutput = where(expandedMask, zqi, zeros_like(zqi));

            // Add the masked output to the accumulated result
            zQ.add_(maskedOutput);

            // Update residual for the next quantizer
            residual.sub_(zqi);

            // Process losses with the mask
            using var maskedCommitmentLoss = mul(commitmentLossi, mask);
            using var meanCommitmentLoss = maskedCommitmentLoss.mean();
            commitmentLoss.add_(meanCommitmentLoss);

            using var maskedCodebookLoss = mul(codebookLossi, mask);
            using var meanCodebookLoss = maskedCodebookLoss.mean();
            codebookLoss.add_(meanCodebookLoss);

            // Store results
            codebookIndices.Add(indicesi);
            latentsList.Add(zei);
        }

        // Combine tensors from all quantizers
        var codes = stack(codebookIndices, dim: 1);
        var latents = cat(latentsList, dim: 1);

        // Clear lists to help GC
        codebookIndices.Clear();
        latentsList.Clear();

        return (
            zQ.MoveToOuterDisposeScope(),
            codes.MoveToOuterDisposeScope(),
            latents.MoveToOuterDisposeScope(),
            commitmentLoss.MoveToOuterDisposeScope(),
            codebookLoss.MoveToOuterDisposeScope()
        );
    }

    /// <summary>
    /// Reconstructs signal from quantized codes
    /// </summary>
    public (Tensor zQ, Tensor projected, Tensor codes) FromCodes(Tensor codes)
    {
        using var scope = NewDisposeScope();

        var zQ = zeros(1, dtype: codes.dtype, device: codes.device);
        var projected = new List<Tensor>();
        var nCodebooks = codes.size(1);

        for (int i = 0; i < nCodebooks; i++)
        {
            // Get the codes for this quantizer
            using var codesSlice = codes[.., i, ..];

            // Decode codes to embeddings
            var zPi = quantizers[i].DecodeCode(codesSlice);
            projected.Add(zPi.clone().MoveToOuterDisposeScope());

            // Project to output space and add to accumulated result
            using var zQi = quantizers[i].OutProj(zPi);
            zQ = zQ.add(zQi);
        }

        return (
            zQ.MoveToOuterDisposeScope(),
            cat(projected, dim: 1).MoveToOuterDisposeScope(),
            codes.MoveToOuterDisposeScope()
        );
    }

    /// <summary>
    /// Reconstructs signal from latent representations
    /// </summary>
    public (Tensor zQ, Tensor projected, Tensor codes) FromLatents(Tensor latents)
    {
        using var scope = NewDisposeScope();

        var zQ = zeros(1, device: latents.device);
        var projected = new List<Tensor>();
        var codesList = new List<Tensor>();

        // Calculate cumulative dimensions
        var dims = new int[quantizers.Count + 1];
        dims[0] = 0;
        for (int i = 0; i < quantizers.Count; i++)
        {
            dims[i + 1] = dims[i] + quantizers[i].CodebookDim;
        }

        // Find number of codebooks that fit in latent dimension
        var latentSize = latents.size(1);
        var nCodebooks = 0;
        for (int i = 0; i < dims.Length - 1; i++)
        {
            if (dims[i + 1] <= latentSize)
            {
                nCodebooks = i + 1;
            }
        }

        for (int i = 0; i < nCodebooks; i++)
        {
            var start = dims[i];
            var end = dims[i + 1];

            // Use using statement for the slice to ensure proper disposal
            using var latentSlice = latents.narrow(1, start, end - start);

            var (zPi, codesI) = quantizers[i].DecodeLatents(latentSlice);
            projected.Add(zPi.clone().MoveToOuterDisposeScope());
            codesList.Add(codesI.clone().MoveToOuterDisposeScope());

            // Use using statement for the projected output to ensure proper disposal
            using var zQi = quantizers[i].OutProj(zPi);
            zQ = zQ.add(zQi);

            // Ensure zPi is disposed after usage since we've cloned it for the output
            zPi.Dispose();
            // Ensure codesI is disposed after usage since we've cloned it for the output
            codesI.Dispose();
        }

        return (
            zQ.MoveToOuterDisposeScope(),
            cat(projected, dim: 1).MoveToOuterDisposeScope(),
            stack(codesList, dim: 1).MoveToOuterDisposeScope()
        );
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