using System.Diagnostics;
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
            var (zQI, _, _, codesI, latentsI) = quantizers[i].forward(residual);
            zQ += zQI;
            residual -= zQI;
            codes.Add(codesI);
            latents.Add(latentsI);
        }

        return (
            zQ.MoveToOuterDisposeScope(),
            torch.stack(codes.ToArray(), 1).MoveToOuterDisposeScope(),
            torch.cat(latents.ToArray(), 1).MoveToOuterDisposeScope(),
            torch.tensor(0.0f).MoveToOuterDisposeScope(),  // Placeholder
            torch.tensor(0.0f).MoveToOuterDisposeScope()   // Placeholder
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
        Tensor nQuantizersPerBatch;

        if (training)
        {
            // Initialize with max number of quantizers
            nQuantizersPerBatch = full(batchSize, _nCodebooks + 1).to(z.device);

            if (_quantizerDropout > 0)
            {
                // Create random dropout pattern for batch
                var dropout = randint(1, _nCodebooks + 1, batchSize).to(z.device);
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

            var (zqi, commitmentLossi, codebookLossi, indicesi, zei) =
                quantizers[i].forward(residual);

            // Create mask for this quantizer
            var mask = full(batchSize, i, device: z.device).lt(nQuantizersPerBatch);

            // Apply mask along batch dimension and expand to match tensor dimensions
            var expandedMask = mask.view(batchSize, 1, 1).expand_as(zqi);

            // Add masked quantizer output
            zQ.add_(where(expandedMask, zqi, zeros_like(zqi)));
            residual.sub_(zqi);

            // Sum losses with mask applied
            commitmentLoss.add_(mul(commitmentLossi, mask).mean());
            codebookLoss.add_(mul(codebookLossi, mask).mean());

            codebookIndices.Add(indicesi);
            latentsList.Add(zei);
        }

        var codes = stack(codebookIndices, dim: 1);
        var latents = cat(latentsList, dim: 1);

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
            // Decode codes to embeddings
             var zPi = quantizers[i].DecodeCode(codes[.., i, ..]);
            projected.Add(zPi.clone());

            // Project to output space
            var zQi = quantizers[i].OutProj(zPi);
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
            var latentSlice = latents.narrow(1, start, end - start);

            var (zPi, codesI) = quantizers[i].DecodeLatents(latentSlice);
            projected.Add(zPi.clone());
            codesList.Add(codesI.clone());

            var zQi = quantizers[i].OutProj(zPi);
            zQ = zQ.add(zQi);
        }

        return (
            zQ.MoveToOuterDisposeScope(),
            cat(projected, dim: 1).MoveToOuterDisposeScope(),
            // Condense codes into single tensor
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
