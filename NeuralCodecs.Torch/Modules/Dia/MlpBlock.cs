using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace NeuralCodecs.Torch.Modules.Dia;

/// <summary>
/// MLP block with SiLU gating.
/// </summary>
public class MlpBlock : Module<Tensor, Tensor>
{
    private readonly ScalarType _dtype;
    private readonly int _intermediateDim;
    private readonly DenseGeneral wi_fused;
    private readonly DenseGeneral wo;

    /// <summary>
    /// Creates a new MLP block.
    /// </summary>
    /// <param name="embedDim">Embedding dimension</param>
    /// <param name="intermediateDim">Intermediate dimension</param>
    /// <param name="computeDtype">Computation data type</param>
    public MlpBlock(int embedDim, int intermediateDim, ScalarType computeDtype)
        : base(nameof(MlpBlock))
    {
        _dtype = computeDtype;
        _intermediateDim = intermediateDim;
        // Fused input projection (for gate and up projection)
        wi_fused = new DenseGeneral(
            inShapes: new[] { embedDim },
            outFeatures: new[] { 2, intermediateDim },
            axis: new[] { -1 },
            weightDtype: computeDtype
        );

        // Output projection
        wo = new DenseGeneral(
            inShapes: new[] { intermediateDim },
            outFeatures: new[] { embedDim },
            axis: new[] { -1 },
            weightDtype: computeDtype
        );

        RegisterComponents();
    }

    /// <summary>
    /// Processes the input tensor through a series of transformations, including a fused projection,  activation,
    /// and output projection, to produce the final output tensor.
    /// </summary>
    /// <remarks>This method performs a fused projection to split the input into gate and up
    /// components, applies the SiLU activation  function to the gate, multiplies it with the up component, and then
    /// processes the result through an output projection.</remarks>
    /// <param name="x">The input tensor to be processed. Must be compatible with the expected dimensions of the model.</param>
    /// <returns>A tensor representing the transformed output after applying the fused projection, activation, and output
    /// projection.</returns>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var fusedX = wi_fused.forward(x);

        // Split into gate and up projections
        using var gate = fusedX.index(TensorIndex.Ellipsis, 0, TensorIndex.Colon);
        using var up = fusedX.index(TensorIndex.Ellipsis, 1, TensorIndex.Colon);

        Tensor hidden = mul(silu(gate), up).to(_dtype);
        var output = wo.forward(hidden);

        return output.MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            wi_fused?.Dispose();
            wo?.Dispose();
        }
        base.Dispose(disposing);
    }
}