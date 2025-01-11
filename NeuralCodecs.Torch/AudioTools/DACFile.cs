using NeuralCodecs.Torch.Config.DAC;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Represents a serialized DAC model file containing compressed audio codes and configuration.
/// </summary>
public class DACFile : IDisposable
{
    private bool _disposed = false;

    public List<Tensor> Codes { get; private set; }
    public DACConfig Config { get; private set; }

    public DACFile(List<Tensor> codes, DACConfig config)
    {
        Codes = codes;
        Config = config;
    }

    /// <summary>
    /// Loads a DAC file from the specified path.
    /// </summary>
    /// <param name="path">Path to the DAC file.</param>
    /// <returns>A DACFile instance containing the loaded codes and configuration.</returns>
    public static async Task<DACFile> LoadAsync(string path)
    {
        await using var file = File.OpenRead(path);
        using var reader = new BinaryReader(file);

        // Read config
        var configLength = reader.ReadInt32();
        var configJson = reader.ReadString();
        var config = System.Text.Json.JsonSerializer.Deserialize<DACConfig>(configJson);

        // Read codes
        var codes = new List<Tensor>();
        var numCodes = reader.ReadInt32();

        for (int i = 0; i < numCodes; i++)
        {
            var shape = new long[reader.ReadInt32()];
            for (int j = 0; j < shape.Length; j++)
            {
                shape[j] = reader.ReadInt64();
            }

            var dataLength = reader.ReadInt32();
            var data = new int[dataLength];
            for (int j = 0; j < dataLength; j++)
            {
                data[j] = reader.ReadInt32();
            }

            var code = tensor(data).reshape(shape);
            codes.Add(code);
        }

        return new DACFile(codes, config);
    }

    /// <summary>
    /// Saves the DAC file to the specified path.
    /// </summary>
    /// <param name="path">Path where the DAC file should be saved.</param>
    public async Task SaveAsync(string path)
    {
        await using var file = File.Create(path);
        await using var writer = new BinaryWriter(file);

        // Write config
        var configJson = System.Text.Json.JsonSerializer.Serialize(Config);
        writer.Write(configJson.Length);
        writer.Write(configJson);

        // Write codes
        writer.Write(Codes.Count);

        foreach (var code in Codes)
        {
            // Write shape
            writer.Write(code.shape.Length);
            foreach (var dim in code.shape)
            {
                writer.Write(dim);
            }

            // Write data
            var data = code.cpu().to(int32).data<int>().ToArray();
            writer.Write(data.Length);
            foreach (var value in data)
            {
                writer.Write(value);
            }
        }

        await writer.BaseStream.FlushAsync();
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                foreach (var code in Codes)
                {
                    code?.Dispose();
                }
            }
            _disposed = true;
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
}