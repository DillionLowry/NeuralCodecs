using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

public static class ConvUtils
{
    /// <summary>
    /// Calculate the extra padding needed for a 1D convolution to keep the output length the same.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="kernelSize"></param>
    /// <param name="stride"></param>
    /// <param name="paddingTotal"></param>
    /// <returns></returns>
    public static long GetExtraPaddingForConv1d(
        Tensor x, long kernelSize, long stride, long paddingTotal = 0)
    {
        var length = x.shape[^1];
        var nFrames = ((length - kernelSize + paddingTotal) / (double)stride) + 1;
        var idealLength = (((long)Math.Ceiling(nFrames) - 1) * stride) + (kernelSize - paddingTotal);
        return idealLength - length;
    }

    /// <summary>
    /// Get the padding mode from a string.
    /// </summary>
    /// <param name="padMode"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static PaddingModes GetPaddingMode(string padMode)
    {
        return padMode switch
        {
            "reflect" => PaddingModes.Reflect,
            "replicate" => PaddingModes.Replicate,
            "circular" => PaddingModes.Circular,
            "zero" => PaddingModes.Zeros,
            _ => throw new ArgumentException($"Invalid padding mode '{padMode}'")
        };
    }

    /// <summary>
    /// Pad a 1D tensor for a convolution operation.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="padding"></param>
    /// <param name="mode"></param>
    /// <returns></returns>
    public static Tensor Pad1d(Tensor x, (long left, long right) padding, string mode)
        => Pad1d(x, padding, GetPaddingMode(mode));

    /// <summary>
    /// Pad a 1D tensor for a convolution operation.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="padding"></param>
    /// <param name="mode"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static Tensor Pad1d(Tensor x, (long left, long right) padding, PaddingModes mode)
    {
        using var scope = NewDisposeScope();
        var (paddingLeft, paddingRight) = padding;

        if (paddingLeft < 0 || paddingRight < 0)
        {
            throw new ArgumentException(
                $"Invalid padding values: ({paddingLeft}, {paddingRight})");
        }

        var length = x.size(-1);

        if (mode == PaddingModes.Reflect)
        {
            var maxPad = Math.Max(paddingLeft, paddingRight);
            var extraPad = 0;

            if (length <= maxPad)
            {
                extraPad = (int)(maxPad - length + 1);
                var zeroPadded = functional.pad(x, new long[] { 0, extraPad }, PaddingModes.Constant);
                x = zeroPadded;
            }

            var padded = functional.pad(x, new[] { paddingLeft, paddingRight }, mode);

            if (extraPad > 0)
            {
                var end = padded.size(-1) - extraPad;
                return padded.slice(-1, 0, end, 1).MoveToOuterDisposeScope();
            }

            return padded.MoveToOuterDisposeScope();
        }

        return functional.pad(x, new[] { paddingLeft, paddingRight }, mode)
            .MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Pad a 1D tensor for a convolution operation.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="kernelSize"></param>
    /// <param name="stride"></param>
    /// <param name="paddingTotal"></param>
    /// <returns></returns>
    public static Tensor PadForConv1d(
        Tensor x, int kernelSize, int stride, int paddingTotal = 0)
    {
        var extraPadding = GetExtraPaddingForConv1d(x, kernelSize, stride, paddingTotal);
        return functional.pad(x, [0, extraPadding]);
    }

    /// <summary>
    /// Unpad a 1D tensor after a convolution operation.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="paddings"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static Tensor Unpad1d(Tensor x, (int left, int right) paddings)
    {
        var (paddingLeft, paddingRight) = paddings;
        if (paddingLeft < 0 || paddingRight < 0)
        {
            throw new ArgumentException($"Invalid padding: ({paddingLeft}, {paddingRight})");
        }

        if (paddingLeft + paddingRight > x.shape[^1])
        {
            throw new ArgumentException("Total padding exceeds tensor length");
        }

        var end = x.shape[^1] - paddingRight;
        return x.slice(-1, paddingLeft, end, 1);
    }
}