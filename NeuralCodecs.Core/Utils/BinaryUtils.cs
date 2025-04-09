namespace NeuralCodecs.Core.Utils;

public class BinaryUtils
{
    /// <summary>
    /// Reads a 32-bit integer from the stream in big-endian byte order
    /// </summary>
    public static int ReadInt32BigEndian(Stream stream)
    {
        byte[] bytes = new byte[4];
        if (stream.Read(bytes, 0, 4) != 4)
        {
            throw new EndOfStreamException("Failed to read integer from stream");
        }

        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(bytes);
        }

        return BitConverter.ToInt32(bytes, 0);
    }

    /// <summary>
    /// Reads a single-precision float from the stream in big-endian byte order
    /// </summary>
    public static float ReadSingleBigEndian(Stream stream)
    {
        byte[] bytes = new byte[4];
        if (stream.Read(bytes, 0, 4) != 4)
        {
            throw new EndOfStreamException("Failed to read float from stream");
        }

        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(bytes);
        }

        return BitConverter.ToSingle(bytes, 0);
    }

    /// <summary>
    /// Writes a 32-bit integer to the stream in big-endian byte order
    /// </summary>
    public static void WriteInt32BigEndian(Stream stream, int value)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(bytes);
        }

        stream.Write(bytes, 0, bytes.Length);
    }

    /// <summary>
    /// Writes a single-precision float to the stream in big-endian byte order
    /// </summary>
    public static void WriteSingleBigEndian(Stream stream, float value)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(bytes);
        }

        stream.Write(bytes, 0, bytes.Length);
    }
}