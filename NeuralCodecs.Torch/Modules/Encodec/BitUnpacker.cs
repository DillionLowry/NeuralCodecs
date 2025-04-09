namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Unpacks bytes from a stream into individual values with specified bit width.
/// </summary>
public class BitUnpacker : IDisposable
{
    private const int MaxBits = 32;
    private readonly int _bits;
    private readonly bool _disposeStream;
    private readonly Stream _inputStream;
    private readonly long _mask;
    private int _currentBits;
    private long _currentValue;
    private bool _disposed;
    private long _totalBitsRead;

    /// <summary>
    /// Initialize a bit unpacker
    /// </summary>
    /// <param name="bits">Number of bits per value</param>
    /// <param name="inputStream">Input stream for reading packed bits</param>
    /// <param name="disposeStream"> Manage the disposal of the input stream</param>
    public BitUnpacker(int bits, Stream inputStream, bool disposeStream = false)
    {
        ValidateParameters(bits, inputStream);

        _bits = bits;
        _inputStream = inputStream;
        _disposeStream = disposeStream;
        _mask = (1L << bits) - 1;
        _currentValue = 0;
        _currentBits = 0;
        _totalBitsRead = 0;
    }

    /// <summary>
    /// Number of bits used per value
    /// </summary>
    public int Bits => _bits;

    /// <summary>
    /// Total number of bits read
    /// </summary>
    public long TotalBitsRead => _totalBitsRead;

    /// <summary>
    /// Dispose of the Bit Unpacker and the input stream if required
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Pull a value from the stream
    /// </summary>
    /// <returns>The next value, or null if end of stream</returns>
    public int? Pull()
    {
        // Ensure we have enough bits for a value
        while (_currentBits < _bits)
        {
            var nextByte = _inputStream.ReadByte();
            if (nextByte == -1)
            {
                // End of stream
                return null;
            }

            _currentValue |= (long)nextByte << _currentBits;
            _currentBits += 8;
        }

        // Extract the value
        var value = (int)(_currentValue & _mask);
        _currentValue >>= _bits;
        _currentBits -= _bits;
        _totalBitsRead += _bits;

        return value;
    }

    /// <summary>
    /// Pull multiple values from the stream
    /// </summary>
    /// <param name="count">Number of values to pull</param>
    /// <returns>List of values, or null if end of stream</returns>
    public List<int>? PullMany(int count)
    {
        if (count <= 0)
        {
            throw new ArgumentException("Count must be positive", nameof(count));
        }

        var values = new List<int>(count);

        for (int i = 0; i < count; i++)
        {
            var value = Pull();
            if (!value.HasValue)
            {
                // End of stream
                break;
            }

            values.Add(value.Value);
        }

        return values.Count > 0 ? values : null;
    }

    /// <summary>
    /// Dispose of the Bit Unpacker and the input stream if required
    /// </summary>
    /// <param name="disposing"></param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing && _disposeStream)
            {
                _inputStream.Dispose();
            }
            _disposed = true;
        }
    }

    /// <summary>
    /// Validate constructor parameters
    /// </summary>
    private static void ValidateParameters(int bits, Stream inputStream)
    {
        if (bits <= 0)
        {
            throw new ArgumentException("Bits must be positive", nameof(bits));
        }

        if (bits > MaxBits)
        {
            throw new ArgumentException($"Bits must be <= {MaxBits}", nameof(bits));
        }

        if (inputStream == null)
        {
            throw new ArgumentNullException(nameof(inputStream), "Input stream cannot be null");
        }

        if (!inputStream.CanRead)
        {
            throw new ArgumentException("Stream must be readable", nameof(inputStream));
        }
    }
}