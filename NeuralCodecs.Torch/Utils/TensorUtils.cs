using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Utils;

public static class TensorUtils
{
    public static Tensor AudioToTensor(float[] audioData, int batchSize = 1, int channels = 1, torch.Device device = null)
    {
        ArgumentNullException.ThrowIfNull(audioData);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(channels, 1);
        device ??= torch.CPU;

        using var scope = torch.NewDisposeScope();

        // Convert to tensor and reshape to (batch, channels, time)
        var tensor = torch.tensor(audioData, dtype: torch.float32)
                 .reshape(batchSize, channels, -1)
                 .to(device);

        return tensor.MoveToOuterDisposeScope();
    }

    public static void ValidateShape(string name, Tensor tensor, long[] expectedShape)
    {
        var actualShape = tensor.shape;
        if (actualShape.Length != expectedShape.Length)
        {
            throw new ArgumentException(
                $"Shape mismatch for {name}. " +
                $"Expected rank {expectedShape.Length}, got {actualShape.Length}. " +
                $"Expected shape: [{string.Join(", ", expectedShape)}], " +
                $"Got: [{string.Join(", ", actualShape)}]"
            );
        }

        for (int i = 0; i < expectedShape.Length; i++)
        {
            if (expectedShape[i] != -1 && expectedShape[i] != actualShape[i])
            {
                throw new ArgumentException(
                    $"Shape mismatch for {name} at dimension {i}. " +
                    $"Expected {expectedShape[i]}, got {actualShape[i]}. " +
                    $"Full expected shape: [{string.Join(", ", expectedShape)}], " +
                    $"Got: [{string.Join(", ", actualShape)}]"
                );
            }
        }
    }

    public static void ValidateShapeRange(string name, Tensor tensor,
        (long min, long max)[] expectedRanges)
    {
        var shape = tensor.shape;
        if (shape.Length != expectedRanges.Length)
        {
            throw new ArgumentException(
                $"Shape rank mismatch for {name}. " +
                $"Expected rank {expectedRanges.Length}, got {shape.Length}"
            );
        }

        for (int i = 0; i < expectedRanges.Length; i++)
        {
            var (min, max) = expectedRanges[i];
            if (shape[i] < min || shape[i] > max)
            {
                throw new ArgumentException(
                    $"Shape out of range for {name} at dimension {i}. " +
                    $"Expected range [{min}, {max}], got {shape[i]}"
                );
            }
        }
    }

    public static void LogTensor(Tensor tensor, string name)
    {
        Console.WriteLine($"{name} shape: {string.Join(", ", tensor.shape)}");
        tensor.WriteTensorToFile($"{name}.txt", count: 5000);
    }

    public static float[] Normalize(float[] input, float epsilon = 1e-5f)
    {
        float mean = 0;
        float variance = 0;

        // Calculate mean
        for (int i = 0; i < input.Length; i++)
            mean += input[i];
        mean /= input.Length;

        // Calculate variance
        for (int i = 0; i < input.Length; i++)
        {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= input.Length;

        // Normalize
        var output = new float[input.Length];
        float std = (float)Math.Sqrt(variance + epsilon);
        for (int i = 0; i < input.Length; i++)
            output[i] = (input[i] - mean) / std;

        return output;
    }

    public static float[] LayerNorm(float[] input, int channels, float[]? weight = null, float[]? bias = null)
    {
        if (input.Length % channels != 0)
            throw new ArgumentException("Input length must be divisible by number of channels");

        int timeSteps = input.Length / channels;
        var output = new float[input.Length];

        for (int t = 0; t < timeSteps; t++)
        {
            var slice = new Span<float>(input, t * channels, channels);
            var normalized = Normalize(slice.ToArray());

            for (int c = 0; c < channels; c++)
            {
                float value = normalized[c];
                if (weight != null)
                    value *= weight[c];
                if (bias != null)
                    value += bias[c];
                output[t * channels + c] = value;
            }
        }

        return output;
    }

    public static float[] Reshape(float[] input, params int[] shape)
    {
        int totalSize = 1;
        foreach (int dim in shape)
            totalSize *= dim;

        if (totalSize != input.Length)
            throw new ArgumentException("New shape must have same total size as input");

        return input.ToArray(); // Since we're working with flat arrays
    }
}