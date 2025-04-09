using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Convolution-friendly LayerNorm that properly handles channel dimensions
/// by moving channels to last dimension before normalization.
/// </summary>
public class ConvLayerNorm : Module<Tensor, Tensor>
{
    private readonly bool _elementwiseAffine;
    private readonly float _eps;
    private readonly LayerNorm _norm;
    private readonly int _normalizedShape;

    public ConvLayerNorm(
        int normalizedShape,
        float eps = 1e-5f,
        bool elementwiseAffine = true) : base("ConvLayerNorm")
    {
        _normalizedShape = normalizedShape;
        _eps = eps;
        _elementwiseAffine = elementwiseAffine;

        // Create standard LayerNorm with proper shape
        _norm = LayerNorm(normalizedShape, eps: eps, elementwise_affine: elementwiseAffine);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();

        // Rearrange tensor to put channels last: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1).contiguous();

        x = _norm.forward(x);

        // Restore original shape: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1).contiguous();

        return x.MoveToOuterDisposeScope();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _norm?.Dispose();
        }
        base.Dispose(disposing);
    }
}