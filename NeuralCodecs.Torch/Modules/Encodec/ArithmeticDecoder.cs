using NeuralCodecs.Torch.Modules.Encodec;
using static TorchSharp.torch;

/// <summary>
/// Improved Arithmetic decoder for entropy decoding.
/// Follows the reference Python implementation closely for binary compatibility.
/// </summary>
public class ArithmeticDecoder : IDisposable
{
    private readonly List<(long, long, long)> _debug;
    private readonly int _totalRangeBits;
    private readonly BitUnpacker _unpacker;
    private long _current;
    private bool _disposed;
    private long _high;
    private object _last;
    private long _low;
    private int _maxBit;

    /// <summary>
    /// Initialize an arithmetic decoder for decompressing data from a stream
    /// </summary>
    /// <param name="stream">Input stream containing compressed data</param>
    /// <param name="totalRangeBits">Total range bits, typically 24 (default) or less</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
    public ArithmeticDecoder(Stream stream, int totalRangeBits = 24)
    {
        ValidateParameters(stream, totalRangeBits);

        _totalRangeBits = totalRangeBits;
        _unpacker = new BitUnpacker(1, stream);
        _low = 0;
        _high = 0;
        _current = 0;
        _maxBit = -1;
        _debug = new List<(long, long, long)>();
    }

    /// <summary>
    /// Gets the current range width
    /// </summary>
    public long Delta => _high - _low + 1;

    /// <summary>
    /// Gets the total range bits used for coding
    /// </summary>
    public int TotalRangeBits => _totalRangeBits;

    /// <summary>
    /// Dispose resources used by the arithmetic decoder
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _unpacker.Dispose();
            _disposed = true;
        }
    }

    /// <summary>
    /// Pull a symbol from the stream using arithmetic decoding
    /// </summary>
    /// <param name="quantizedCdf">Quantized CDF for the symbol probability distribution</param>
    /// <returns>Decoded symbol, or null if end of stream reached</returns>
    /// <exception cref="ArgumentException">Thrown when CDF is invalid</exception>
    /// <exception cref="InvalidOperationException">Thrown when decoding state becomes invalid</exception>
    public int? Pull(Tensor quantizedCdf)
    {
        if (quantizedCdf is null || quantizedCdf.IsInvalid)
        {
            throw new ArgumentException("Quantized CDF must be valid", nameof(quantizedCdf));
        }

        if (quantizedCdf.dim() != 1)
        {
            throw new ArgumentException(
                $"CDF must be 1-dimensional, got {quantizedCdf.dim()}D", nameof(quantizedCdf));
        }

        if (quantizedCdf.shape[0] < 2)
        {
            throw new ArgumentException(
                $"CDF must have at least 2 entries, got {quantizedCdf.shape[0]}", nameof(quantizedCdf));
        }

        // Ensure our range is large enough, reading bits as needed
        while (Delta < (1L << TotalRangeBits))
        {
            var bit = _unpacker.Pull();
            if (!bit.HasValue)
            {
                // End of stream
                return null;
            }

            _low *= 2;
            _high = (_high * 2) + 1;
            _current = (_current * 2) + bit.Value;
            _maxBit++;
        }

        // Save the state before binary search
        _last = (_low, _high, _current, _maxBit);

        // Binary search to find the symbol
        int symbol;
        (int symbol, long low, long high, long current) BinarySearch(long lowIdx, long highIdx)
        {
            if (highIdx < lowIdx)
            {
                throw new InvalidOperationException("Binary search failed: high index < low index");
            }

            var mid = (lowIdx + highIdx) / 2;

            // Get range for the symbol from CDF
            var rangeLow = mid > 0 ? quantizedCdf[mid - 1].item<long>() : 0;
            var rangeHigh = quantizedCdf[mid].item<long>() - 1;

            // Scale the range to the current interval
            double scale = Delta / (double)(1L << TotalRangeBits);
            var effectiveLow = (long)Math.Ceiling(rangeLow * scale);
            var effectiveHigh = (long)Math.Floor(rangeHigh * scale);

            // Compute absolute position in the range
            long low = effectiveLow + _low;
            long high = effectiveHigh + _low;

            // Check if current value falls within this range
            if (_current >= low && _current <= high)
            {
                // Symbol found
                return ((int)mid, low, high, _current);
            }

            if (_current > high)
            {
                // Search in upper half
                return BinarySearch(mid + 1, highIdx);
            }

            // Search in lower half
            return BinarySearch(lowIdx, mid - 1);
        }
        try
        {
            (symbol, _low, _high, _current) = BinarySearch(0, quantizedCdf.shape[0] - 1);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Binary search failed: {ex.Message}", ex);
        }

        // Store debug information
        _debug.Add((_low, _high, _current));

        // Flush common prefix bits
        FlushCommonPrefix();

        // Store updated state after flushing
        _debug.Add((_low, _high, _current));

        return symbol;
    }

    private static void ValidateParameters(Stream stream, int totalRangeBits)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanRead)
        {
            throw new ArgumentException("Stream must be readable", nameof(stream));
        }

        if (totalRangeBits <= 0)
        {
            throw new ArgumentException("Total range bits must be positive", nameof(totalRangeBits));
        }

        if (totalRangeBits > 30)
        {
            throw new ArgumentException("Total range bits must be <= 30 for numerical stability",
                                       nameof(totalRangeBits));
        }
    }

    /// <summary>
    /// Flush common prefix bits to maintain numerical precision
    /// </summary>
    private void FlushCommonPrefix()
    {
        // Validate invariants
        if (_high < _low)
        {
            throw new InvalidOperationException($"Invalid range: low={_low}, high={_high}");
        }

        // While the most significant bits are the same, we can remove them
        while (_maxBit >= 0)
        {
            var b1 = (_low >> _maxBit) & 1;
            var b2 = (_high >> _maxBit) & 1;

            if (b1 == b2)
            {
                // Remove the common bit from low, high, and current
                _low -= b1 << _maxBit;
                _high -= b1 << _maxBit;
                _current -= b1 << _maxBit;

                // Check invariants after bit removal
                if (_high < _low)
                {
                    throw new InvalidOperationException(
                        $"Invalid range after bit flush: low={_low}, high={_high}");
                }

                if (_low < 0)
                {
                    throw new InvalidOperationException($"Negative low value: {_low}");
                }

                // Move to next bit position
                _maxBit--;
            }
            else
            {
                // Bits differ, can't flush any more
                break;
            }
        }
    }
}