using System.Text;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Binary I/O utilities for Encodec model files
/// </summary>
public static class BinaryIO
{
    private const ushort CURRENT_VERSION = 0;
    private const string MAGIC = "ECDC";

    private static readonly System.Text.Json.JsonSerializerOptions JsonOptions = new()
    {
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowReadingFromString
    };

    /// <summary>
    /// Represents supported metadata value types
    /// </summary>
    private enum MetadataType : byte
    {
        String = 0,
        Int32 = 1,
        Boolean = 2,
        Float = 3,
        Int64 = 4
    }

    /// <summary>
    /// Reads an Encodec header with metadata from a stream
    /// </summary>
    /// <param name="stream"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentNullException"></exception>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="EndOfStreamException"></exception>
    /// <exception cref="InvalidDataException"></exception>
    /// <exception cref="IOException"></exception>
    public static async Task<Dictionary<string, object>> ReadHeaderAsync(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanRead)
        {
            throw new ArgumentException("Stream must be readable", nameof(stream));
        }

        try
        {
            // Read and verify magic number
            var magic = new byte[4];
            if (await stream.ReadAsync(magic) != 4)
            {
                throw new EndOfStreamException("Incomplete header magic number");
            }

            if (Encoding.ASCII.GetString(magic) != MAGIC)
            {
                throw new InvalidDataException("Invalid Encodec header magic number");
            }

            // Read and verify version
            int version = stream.ReadByte();
            if (version != CURRENT_VERSION)
            {
                throw new InvalidDataException($"Unsupported header version: {version}");
            }

            // Read metadata length
            var lengthBytes = new byte[4];
            if (await stream.ReadAsync(lengthBytes) != 4)
            {
                throw new EndOfStreamException("Incomplete metadata length");
            }

            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(lengthBytes);
            }

            var metaLength = BitConverter.ToInt32(lengthBytes);

            // Read metadata JSON
            var metaBytes = new byte[metaLength];
            var bytesRead = await stream.ReadAsync(metaBytes);
            if (bytesRead != metaLength)
            {
                throw new EndOfStreamException("Incomplete metadata");
            }

            var metaJson = Encoding.UTF8.GetString(metaBytes);
            return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(
                metaJson, JsonOptions) ?? new Dictionary<string, object>();
        }
        catch (Exception ex) when (ex is not InvalidDataException)
        {
            throw new IOException("Failed to read Encodec header", ex);
        }
    }

    /// <summary>
    /// Validates Encodec metadata entries for required keys and types
    /// </summary>
    /// <param name="metadata"></param>
    /// <exception cref="ArgumentNullException"></exception>
    /// <exception cref="ArgumentException"></exception>
    public static void ValidateMetadata(Dictionary<string, object> metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);

        var requiredKeys = new[] { "m", "al", "nc", "lm" };
        foreach (var key in requiredKeys)
        {
            if (!metadata.ContainsKey(key))
            {
                throw new ArgumentException($"Missing required metadata key: {key}");
            }
        }

        // Basic type validation - note that JSON deserialization may give us JsonElement
        if (metadata["m"] is not string and not System.Text.Json.JsonElement { ValueKind: System.Text.Json.JsonValueKind.String })
        {
            throw new ArgumentException("Model name must be string");
        }

        if (metadata["al"] is not int and not long and not System.Text.Json.JsonElement { ValueKind: System.Text.Json.JsonValueKind.Number })
        {
            throw new ArgumentException("Audio length must be integer");
        }

        if (metadata["nc"] is not int and not System.Text.Json.JsonElement { ValueKind: System.Text.Json.JsonValueKind.Number })
        {
            throw new ArgumentException("Number of codebooks must be integer");
        }

        if (metadata["lm"] is not bool and not System.Text.Json.JsonElement { ValueKind: System.Text.Json.JsonValueKind.True or System.Text.Json.JsonValueKind.False })
        {
            throw new ArgumentException("Language model flag must be boolean");
        }
    }

    /// <summary>
    /// Writes an Encodec header with metadata to a stream
    /// </summary>
    /// <param name="stream"></param>
    /// <param name="metadata"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentNullException"></exception>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="IOException"></exception>
    public static async Task WriteHeaderAsync(Stream stream, Dictionary<string, object> metadata)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanWrite)
        {
            throw new ArgumentException("Stream must be writable", nameof(stream));
        }

        ArgumentNullException.ThrowIfNull(metadata);

        try
        {
            // Serialize metadata to JSON bytes
            var metaJson = System.Text.Json.JsonSerializer.Serialize(metadata, JsonOptions);
            var metaBytes = Encoding.UTF8.GetBytes(metaJson);

            // Write header: MAGIC (4 bytes) + VERSION (1 byte) + META_LENGTH (4 bytes)
            await stream.WriteAsync(Encoding.ASCII.GetBytes(MAGIC));
            stream.WriteByte((byte)CURRENT_VERSION);
            var lengthBytes = BitConverter.GetBytes(metaBytes.Length);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(lengthBytes);  // Convert to big-endian
            }

            await stream.WriteAsync(lengthBytes);

            // Write metadata JSON
            await stream.WriteAsync(metaBytes);
            await stream.FlushAsync();
        }
        catch (Exception ex)
        {
            throw new IOException("Failed to write Encodec header", ex);
        }
    }

    private static (string key, object value, int size) ReadMetadataEntry(
        BinaryReader reader)
    {
        // Read key
        var keyLength = reader.ReadByte();
        var keyBytes = reader.ReadBytes(keyLength);
        var key = Encoding.UTF8.GetString(keyBytes);

        // Read type
        var type = (MetadataType)reader.ReadByte();

        // Read value based on type
        object value;
        int valueSize;

        switch (type)
        {
            case MetadataType.String:
                var strLength = reader.ReadUInt16();
                var strBytes = reader.ReadBytes(strLength);
                value = Encoding.UTF8.GetString(strBytes);
                valueSize = strLength;
                break;

            case MetadataType.Int32:
                value = reader.ReadInt32();
                valueSize = 4;
                break;

            case MetadataType.Int64:
                value = reader.ReadInt64();
                valueSize = 8;
                break;

            case MetadataType.Boolean:
                value = reader.ReadBoolean();
                valueSize = 1;
                break;

            case MetadataType.Float:
                value = reader.ReadSingle();
                valueSize = 4;
                break;

            default:
                throw new InvalidDataException($"Unknown metadata type: {type}");
        }

        return (key, value, keyLength + 1 + valueSize); // +1 for type byte
    }

    private static void WriteMetadataEntry(BinaryWriter writer, string key, object value)
    {
        // Write key
        var keyBytes = Encoding.UTF8.GetBytes(key);
        if (keyBytes.Length > 255)
        {
            throw new ArgumentException($"Key too long: {key}");
        }

        writer.Write((byte)keyBytes.Length);
        writer.Write(keyBytes);

        // Write value based on type
        switch (value)
        {
            case string s:
                writer.Write((byte)MetadataType.String);
                var strBytes = Encoding.UTF8.GetBytes(s);
                if (strBytes.Length > 65535)
                {
                    throw new ArgumentException($"String value too long: {key}");
                }

                writer.Write((ushort)strBytes.Length);
                writer.Write(strBytes);
                break;

            case int i:
                writer.Write((byte)MetadataType.Int32);
                writer.Write(i);
                break;

            case long l:
                writer.Write((byte)MetadataType.Int64);
                writer.Write(l);
                break;

            case bool b:
                writer.Write((byte)MetadataType.Boolean);
                writer.Write(b);
                break;

            case float f:
                writer.Write((byte)MetadataType.Float);
                writer.Write(f);
                break;

            default:
                throw new ArgumentException(
                    $"Unsupported metadata type: {value.GetType()} for key: {key}");
        }
    }
}