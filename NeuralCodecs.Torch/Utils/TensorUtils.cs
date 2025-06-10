using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Utils;

/// <summary>
/// Provides utility methods for working with TorchSharp tensors and audio data conversions.
/// Contains helper methods for tensor manipulation, validation, and audio processing operations.
/// </summary>
public static class TensorUtils
{
    /// <summary>
    /// Converts audio sample data to a batched tensor suitable for neural network processing.
    /// </summary>
    /// <param name="audioData">Raw audio samples as a float array.</param>
    /// <param name="batchSize">Number of samples in the batch. Defaults to 1.</param>
    /// <param name="channels">Number of audio channels. Defaults to 1 (mono).</param>
    /// <param name="device">Target device for the tensor. Defaults to CPU if null.</param>
    /// <returns>A tensor with shape (batch, channels, time).</returns>
    /// <exception cref="ArgumentNullException">Thrown when audioData is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when batchSize or channels is less than 1.</exception>
    public static Tensor AudioToTensor(float[] audioData, int batchSize = 1, int channels = 1, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(audioData);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(channels, 1);
        device ??= CPU;

        using var scope = NewDisposeScope();

        // Convert to tensor and reshape to (batch, channels, time)
        var tensor = torch.tensor(audioData, dtype: float32)
                 .reshape(batchSize, channels, -1)
                 .to(device);

        return tensor.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Checks if a tensor has NaN or Inf values
    /// </summary>
    /// <param name="tensor">Tensor to check</param>
    /// <param name="name">Name of tensor for logging</param>
    /// <returns>True if tensor contains NaN or Inf values</returns>
    public static bool HasNaNOrInf(Tensor tensor, string name = "")
    {
        bool hasNaN = tensor.isnan().any().item<bool>();
        bool hasInf = tensor.isinf().any().item<bool>();

        if (hasNaN || hasInf)
        {
            Debug.WriteLine($"Tensor '{name}' contains {(hasNaN ? "NaN" : "")} {(hasInf ? "Inf" : "")} values");
            return true;
        }

        return false;
    }

    /// <summary>
    /// Logs tensor shape information to the console and writes tensor data to a file.
    /// </summary>
    /// <param name="tensor">The tensor to log.</param>
    /// <param name="name">Name identifier for the tensor in logs and filename.</param>
    public static void LogTensor(Tensor tensor, string name, int precision = 10, int count = 1500)
    {
        Console.WriteLine($"{name} shape: {string.Join(", ", tensor.shape)}");
        tensor.WriteTensorToFile($"{name}.txt", precision, count: count);
    }

    /// <summary>
    /// Performs padding for 1D tensors with special handling for reflection padding
    /// </summary>
    /// <param name="x">Input tensor</param>
    /// <param name="paddings">Left and right padding amounts</param>
    /// <param name="mode">Padding mode: "zero", "reflect", "replicate", or "circular"</param>
    /// <param name="value">Value for constant padding</param>
    /// <returns>Padded tensor</returns>
    public static Tensor Pad1d(Tensor x, (int left, int right) paddings, string mode = "zero", float value = 0f)
    {
        int length = (int)x.shape[^1];
        int paddingLeft = paddings.left;
        int paddingRight = paddings.right;

        if (paddingLeft < 0 || paddingRight < 0)
        {
            throw new ArgumentException($"Padding must be non-negative, got ({paddingLeft}, {paddingRight})");
        }

        PaddingModes paddingMode = mode switch
        {
            "zero" => PaddingModes.Constant,
            "reflect" => PaddingModes.Reflect,
            "replicate" => PaddingModes.Replicate,
            "circular" => PaddingModes.Circular,
            _ => throw new ArgumentException($"Unsupported padding mode: {mode}")
        };

        // Special handling for reflect padding on small inputs
        if (mode == "reflect")
        {
            int maxPad = Math.Max(paddingLeft, paddingRight);
            if (length <= maxPad)
            {
                int extraPad = maxPad - length + 1;

                // First add zero padding to make the input larger
                var extraPadded = nn.functional.pad(x, new long[] { 0, extraPad }, PaddingModes.Constant, value);

                // Then apply reflect padding
                var padded = nn.functional.pad(extraPadded, new long[] { paddingLeft, paddingRight },
                                             PaddingModes.Reflect);

                // Trim any extra padding if needed
                if (extraPad > 0)
                {
                    int end = (int)(padded.shape[^1] - extraPad);
                    return padded.narrow(-1, 0, end);
                }

                return padded;
            }
        }

        // Standard padding
        return nn.functional.pad(x, new long[] { paddingLeft, paddingRight }, paddingMode, value);
    }

    /// <summary>
    /// Removes padding from the left and right sides of a tensor
    /// </summary>
    /// <param name="x">Input tensor</param>
    /// <param name="paddingLeft">Left padding to remove</param>
    /// <param name="paddingRight">Right padding to remove</param>
    /// <returns>Unpadded tensor</returns>
    public static Tensor Unpad1d(Tensor x, int paddingLeft, int paddingRight)
    {
        int length = (int)x.shape[^1];

        if (paddingLeft < 0 || paddingRight < 0)
        {
            throw new ArgumentException($"Padding must be non-negative, got ({paddingLeft}, {paddingRight})");
        }

        if (paddingLeft + paddingRight >= length)
        {
            throw new ArgumentException(
                $"Total padding ({paddingLeft} + {paddingRight}) exceeds tensor length {length}");
        }

        int end = length - paddingRight;
        return x.narrow(-1, paddingLeft, end - paddingLeft);
    }

    /// <summary>
    /// Validates that a tensor's shape matches the expected dimensions exactly.
    /// </summary>
    /// <param name="name">Identifier for the tensor being validated.</param>
    /// <param name="tensor">Tensor to validate.</param>
    /// <param name="expectedShape">Expected shape array. Use -1 for dynamic dimensions.</param>
    /// <exception cref="ArgumentException">Thrown when tensor shape doesn't match expected shape.</exception>
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

    /// <summary>
    /// Validates that each dimension of a tensor's shape falls within specified ranges.
    /// </summary>
    /// <param name="name">Identifier for the tensor being validated.</param>
    /// <param name="tensor">Tensor to validate.</param>
    /// <param name="expectedRanges">Array of (min, max) tuples specifying valid ranges for each dimension.</param>
    /// <exception cref="ArgumentException">Thrown when tensor dimensions fall outside specified ranges.</exception>
    public static void ValidateShapeRange(string name, Tensor tensor, (long min, long max)[] expectedRanges)
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
}