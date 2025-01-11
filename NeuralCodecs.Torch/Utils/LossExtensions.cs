using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Utils;

public static class LossExtensions
{
    public static Tensor CombineLosses(this IEnumerable<(Tensor loss, float weight)> losses)
    {
        return losses.Aggregate(zeros(1),
            (current, loss) => current + loss.loss * loss.weight);
    }
}