using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Improved Arithmetic coder implementation for entropy coding.
/// Follows the reference Python implementation closely for binary compatibility.
/// </summary>
public class ArithmeticCoder : IDisposable
{
    private readonly List<(long, long)> _debug;
    private readonly BitPacker _packer;
    private readonly int _totalRangeBits;
    private bool _disposed;
    private long _high;
    private long _low;
    private int _maxBit;

    /// <summary>
    /// Initialize an arithmetic coder for compressing data to a stream
    /// </summary>
    /// <param name="stream">Output stream for compressed data</param>
    /// <param name="totalRangeBits">Total range bits, typically 24 (default) or less</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
    public ArithmeticCoder(Stream stream, int totalRangeBits = 24)
    {
        ValidateParameters(stream, totalRangeBits);

        _totalRangeBits = totalRangeBits;
        _packer = new BitPacker(1, stream);
        _low = 0;
        _high = 0;
        _maxBit = -1;
        _debug = new List<(long, long)>();
    }

    /// <summary>
    /// Gets the current range width
    /// </summary>
    public long Delta => _high - _low + 1;

    /// <summary>
    /// Gets the total range bits used for coding
    /// </summary>
    public int TotalRangeBits => _totalRangeBits;

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Flush any remaining bits to the stream
    /// </summary>
    public void Flush()
    {
        // Send any remaining bits in the low value
        while (_maxBit >= 0)
        {
            var bit = (_low >> _maxBit) & 1;
            _packer.Push((int)bit);
            _maxBit--;
        }

        // Flush the bit packer to ensure all bits are written
        _packer.Flush();
    }

    /// <summary>
    /// Push a symbol to the stream using arithmetic coding
    /// </summary>
    /// <param name="symbol">Symbol to encode</param>
    /// <param name="quantizedCdf">Quantized CDF for the symbol probability distribution</param>
    /// <exception cref="ArgumentException">Thrown when CDF is invalid</exception>
    /// <exception cref="InvalidOperationException">Thrown when coding state becomes invalid</exception>
    public void Push(int symbol, Tensor quantizedCdf)
    {
        if (quantizedCdf is null || quantizedCdf.IsInvalid)
        {
            throw new ArgumentException("Quantized CDF must be valid", nameof(quantizedCdf));
        }

        if (symbol < 0 || symbol > quantizedCdf.shape[0] - 1)
        {
            throw new ArgumentException(
                $"Symbol {symbol} out of range [0, {quantizedCdf.shape[0] - 1}]", nameof(symbol));
        }

        // Ensure our range is large enough, pushing bits as needed
        while (Delta < (1L << TotalRangeBits))
        {
            _low *= 2;
            _high = (_high * 2) + 1;
            _maxBit++;
        }

        // Get range bounds for the symbol from the CDF
        long rangeLow = symbol == 0 ? 0 : quantizedCdf[symbol - 1].item<long>();
        long rangeHigh = quantizedCdf[symbol].item<long>() - 1;

        // Scale the range to the current interval
        double scale = Delta / (double)(1L << TotalRangeBits);
        var effectiveLow = (long)Math.Ceiling(rangeLow * scale);
        var effectiveHigh = (long)Math.Floor(rangeHigh * scale);

        // Check for invalid range
        if (effectiveLow > effectiveHigh)
        {
            throw new InvalidOperationException(
                $"Invalid range for symbol {symbol}: low={effectiveLow}, high={effectiveHigh}");
        }

        // Update the current range
        _high = _low + effectiveHigh;
        _low += effectiveLow;

        // Check for range validity after update
        if (_low > _high)
        {
            throw new InvalidOperationException(
                $"Invalid range after update: low={_low}, high={_high}");
        }

        // Store range for debugging
        _debug.Add((_low, _high));

        // Flush any common prefix bits
        FlushCommonPrefix();

        // Validate max bit after flushing
        if (_maxBit < -1)
        {
            throw new InvalidOperationException($"Invalid max bit: {_maxBit}");
        }

        if (_maxBit > 61)
        {
            throw new InvalidOperationException($"Max bit too large: {_maxBit}, at risk of overflow");
        }
    }

    /// <summary>
    /// Dispose resources used by the arithmetic coder
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            try
            {
                Flush();
            }
            catch
            {
                // Ignore exceptions during disposal
            }

            _packer.Dispose();
            _disposed = true;
        }
    }

    private static void ValidateParameters(Stream stream, int totalRangeBits)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanWrite)
        {
            throw new ArgumentException("Stream must be writable", nameof(stream));
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
    /// Flush common prefix bits to the stream to maintain numerical precision
    /// </summary>
    private void FlushCommonPrefix()
    {
        // Validate invariants
        if (_high < _low)
        {
            throw new InvalidOperationException($"Invalid range: low={_low}, high={_high}");
        }

        if (_maxBit >= 0 && _high >= (1L << (_maxBit + 1)))
        {
            throw new InvalidOperationException(
                $"High value {_high} exceeds maximum bit position {_maxBit}");
        }

        // While the most significant bits are the same, we can send them to the output
        while (_maxBit >= 0)
        {
            // Get the most significant bit of low and high
            var b1 = (_low >> _maxBit) & 1;
            var b2 = (_high >> _maxBit) & 1;

            if (b1 == b2)
            {
                // Remove the bit from both low and high
                _low -= b1 << _maxBit;
                _high -= b1 << _maxBit;

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

                // Send the common bit to the output
                _packer.Push((int)b1);
            }
            else
            {
                // Bits differ, can't flush any more
                break;
            }
        }
    }
}