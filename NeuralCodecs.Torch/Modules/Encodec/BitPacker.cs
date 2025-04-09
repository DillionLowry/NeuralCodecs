namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Packs individual bits into bytes for efficient storage.
/// </summary>
public class BitPacker : IDisposable
{
    private const int MaxBits = 24;
    private readonly int _bits;
    private readonly bool _disposeStream;
    private readonly Stream _outputStream;
    private int _currentBits;
    private long _currentValue;
    private bool _disposed;
    private long _totalBitsWritten;

    /// <summary>
    /// Initializes a new instance of the BitPacker class.
    /// </summary>
    /// <param name="bits">Number of bits per value</param>
    /// <param name="outputStream">Input stream for reading packed bits</param>
    /// <param name="disposeStream"> Manage the disposal of the output stream</param>
    /// <exception cref="ArgumentException"></exception>
    public BitPacker(int bits, Stream outputStream, bool disposeStream = false)
    {
        ValidateParameters(bits, outputStream);

        _bits = bits;
        _outputStream = outputStream;
        _disposeStream = disposeStream;
        _currentValue = 0;
        _currentBits = 0;
        _totalBitsWritten = 0;
    }

    /// <summary>
    /// Number of bits used per value
    /// </summary>
    public int Bits => _bits;

    /// <summary>
    /// Total number of bits written
    /// </summary>
    public long TotalBitsWritten => _totalBitsWritten;

    /// <summary>
    /// Total number of bytes written (including pending bits)
    /// </summary>
    public long TotalBytesWritten => (_totalBitsWritten + _currentBits + 7) / 8;

    /// <summary>
    /// Dispose the bit packer, flushing any remaining bits, and optionally disposing the output stream
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Flush any pending bits to the stream
    /// </summary>
    public void Flush()
    {
        if (_disposed)
        {
            return;
        }

        // Write any remaining bits
        if (_currentBits > 0)
        {
            _outputStream.WriteByte((byte)_currentValue);
            _currentValue = 0;
            _currentBits = 0;
        }

        // Flush the underlying stream
        _outputStream.Flush();
    }

    /// <summary>
    /// Push a value to the stream
    /// </summary>
    /// <param name="value">Value to push (must fit in specified bits)</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when value doesn't fit in specified bits
    /// </exception>
    /// <exception cref="ObjectDisposedException">Thrown when BitPacker is disposed</exception>
    public void Push(int value)
    {
        ObjectDisposedException.ThrowIf(_disposed, typeof(BitPacker));

        // Validate value fits in specified bits
        long maxValue = (1L << _bits) - 1;
        if (value < 0 || value > maxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(value),
                $"Value must be between 0 and {maxValue}");
        }

        // Add value to current buffer
        _currentValue |= (long)value << _currentBits;
        _currentBits += _bits;
        _totalBitsWritten += _bits;

        // Write complete bytes to stream
        while (_currentBits >= 8)
        {
            var byteValue = (byte)(_currentValue & 0xFF);
            _outputStream.WriteByte(byteValue);
            _currentValue >>= 8;
            _currentBits -= 8;
        }
    }

    /// <summary>
    /// Push an array of values to the stream
    /// </summary>
    /// <param name="values">Values to push</param>
    public void PushMany(IEnumerable<int> values)
    {
        ArgumentNullException.ThrowIfNull(values);

        foreach (var value in values)
        {
            Push(value);
        }
    }

    /// <summary>
    /// Dispose the bit packer, flushing any remaining bits, and optionally disposing the output stream
    /// </summary>
    /// <param name="disposing">Whether the method is called from Dispose method</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                Flush();
                if (_disposeStream)
                {
                    _outputStream.Dispose();
                }
            }
            _disposed = true;
        }
    }

    /// <summary>
    /// Validate constructor parameters
    /// </summary>
    private static void ValidateParameters(int bits, Stream outputStream)
    {
        if (bits <= 0)
        {
            throw new ArgumentException("Bits must be positive", nameof(bits));
        }

        if (bits > MaxBits)
        {
            throw new ArgumentException($"Bits must be <= {MaxBits}", nameof(bits));
        }

        if (outputStream == null)
        {
            throw new ArgumentNullException(nameof(outputStream), "Output stream cannot be null");
        }

        if (!outputStream.CanWrite)
        {
            throw new ArgumentException("Stream must be writable", nameof(outputStream));
        }
    }
}