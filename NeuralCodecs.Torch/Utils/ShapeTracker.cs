using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Utils;

public class ShapeTracker
{
    private readonly Dictionary<string, (long[] shape, string description)> shapes
        = new();

    public void Track(string name, Tensor tensor, string description)
    {
        shapes[name] = (tensor.shape, description);
    }

    public void DumpShapes()
    {
        foreach (var (name, (shape, description)) in shapes)
        {
            Console.WriteLine($"{name}: {string.Join("x", shape)} - {description}");
        }
    }

    public bool ValidateShape(string name, long[] expectedShape)
    {
        if (!shapes.ContainsKey(name))
            return false;

        var actualShape = shapes[name].shape;
        if (actualShape.Length != expectedShape.Length)
            return false;

        for (int i = 0; i < actualShape.Length; i++)
        {
            if (expectedShape[i] != -1 && expectedShape[i] != actualShape[i])
                return false;
        }

        return true;
    }
}