using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.DAC;

public class ResidualVectorQuantizer : Module<Tensor, (Tensor quantized, Tensor codes, Tensor latents, Tensor commitmentLoss, Tensor codebookLoss)>
{
    private readonly int _nCodebooks;
    private readonly int _codebookSize;
    private readonly int _codebookDim;
    private readonly float _quantizerDropout;
    private readonly ModuleList<VectorQuantizer> quantizers;

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

    public override (Tensor quantized, Tensor codes, Tensor latents, Tensor commitmentLoss, Tensor codebookLoss)
        forward(Tensor z) => forward(z, null);

    public (Tensor quantized, Tensor codes, Tensor latents, Tensor commitmentLoss, Tensor codebookLoss)
        forward(Tensor z, int? nQuantizers)
    {
        using var scope = NewDisposeScope();

        var zQ = zeros_like(z, dtype: z.dtype, device: z.device);
        var residual = z.contiguous().clone();
        var codebookIndices = new List<Tensor>();
        var latentsList = new List<Tensor>();
        var commitmentLoss = tensor(0.0f, device: z.device);
        var codebookLoss = tensor(0.0f, device: z.device);

        // Handle number of quantizers and dropout during training
        var batchSize = z.size(0);
        Tensor nQuantizersPerBatch;

        if (nQuantizers == null)
        {
            nQuantizersPerBatch = full(batchSize, _nCodebooks).to(z.device);
        }

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
}