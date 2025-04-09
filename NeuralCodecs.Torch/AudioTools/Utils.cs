using NAudio.Wave;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Utility functions for audio processing.
/// </summary>
public static partial class Utils
{
    public static readonly string[] AudioExtensions = [".wav", ".flac", ".mp3", ".mp4"];

    public static IDisposable ChangeDirScope(string newDir)
    {
        return new ChangeDirectoryScope(newDir);
    }

    public static (T item, int sourceIndex, int itemIndex) ChooseFromListOfLists<T>(
            Random state,
            List<List<T>> listOfLists,
            float[] p = null)
    {
        var sourceIdx = p == null ?
            state.Next(listOfLists.Count) :
            RandomChoice(state, Enumerable.Range(0, listOfLists.Count).ToArray(), p);

        var itemIdx = state.Next(listOfLists[sourceIdx].Count);
        return (listOfLists[sourceIdx][itemIdx], sourceIdx, itemIdx);
    }

    public static Tensor EnsureTensor(object x, int? ndim = null, int? batchSize = null)
    {
        Tensor tensor;

        if (x is Tensor t)
        {
            tensor = t;
        }
        else if (x is float[] floatArray)
        {
            tensor = torch.tensor(floatArray);
        }
        else if (x is double[] doubleArray)
        {
            tensor = torch.tensor(doubleArray.Select(d => (float)d).ToArray());
        }
        else if (x is float f)
        {
            tensor = torch.tensor(new float[] { f });
        }
        else if (x is double d)
        {
            tensor = torch.tensor(new float[] { (float)d });
        }
        else
        {
            throw new ArgumentException($"Cannot convert type {x.GetType()} to tensor");
        }

        if (ndim.HasValue)
        {
            while (tensor.dim() < ndim.Value)
            {
                tensor = tensor.unsqueeze(-1);
            }
        }

        if (batchSize.HasValue && tensor.shape[0] != batchSize.Value)
        {
            var shape = tensor.shape.ToArray();
            shape[0] = batchSize.Value;
            tensor = tensor.expand(shape);
        }

        return tensor;
    }

    public static IEnumerable<string> FindAudioFiles(string folder, IEnumerable<string> extensions = null)
    {
        extensions ??= AudioExtensions;

        var path = Path.GetFullPath(folder);

        // Handle direct file path
        if (extensions.Any(ext => path.EndsWith(ext, StringComparison.OrdinalIgnoreCase)))
        {
            if (path.Contains("*"))
            {
                return Directory.EnumerateFiles(
                    Path.GetDirectoryName(path),
                    Path.GetFileName(path),
                    SearchOption.AllDirectories);
            }
            return new[] { path };
        }

        return extensions.SelectMany(ext =>
            Directory.EnumerateFiles(path, $"*{ext}", SearchOption.AllDirectories));
    }

    /// <summary>
    /// Gets information about an audio file.
    /// </summary>
    /// <param name="audioPath">Path to the audio file.</param>
    /// <returns>AudioInfo containing file metadata.</returns>
    public static AudioInfo GetAudioInfo(string audioPath)
    {
        using var reader = new AudioFileReader(audioPath);
        return new AudioInfo
        {
            SampleRate = reader.WaveFormat.SampleRate,
            NumFrames = (int)(reader.Length / (reader.WaveFormat.BitsPerSample / 8) / reader.WaveFormat.Channels)
        };
    }

    public static Random GetRandomState(object seed)
    {
        if (seed == null)
        {
            return new Random();
        }
        else if (seed is int intSeed)
        {
            return new Random(intSeed);
        }
        else if (seed is Random rng)
        {
            return rng;
        }
        else
        {
            throw new ArgumentException($"Cannot use {seed} to seed Random");
        }
    }

    public static Tensor HzToBin(Tensor hz, int nFft, int sampleRate)
    {
        var shape = hz.shape;
        hz = hz.flatten();
        var freqs = linspace(0, sampleRate / 2f, 2 + (nFft / 2));

        // Clamp frequencies to Nyquist
        hz = clamp(hz, 0, sampleRate / 2f);

        var closest = (hz.unsqueeze(0) - freqs.unsqueeze(1)).abs();
        var closestBins = closest.min(dim: 0).indexes;

        return closestBins.reshape(shape);
    }

    public static object PrepareBatch(object batch, Device device)
    {
        switch (batch)
        {
            case IDictionary<string, object> dict:
                return dict.ToDictionary(
                    kvp => kvp.Key,
                    kvp => PrepareBatch(kvp.Value, device));

            case Tensor tensor:
                return tensor.to(device);

            case IList<object> list:
                for (int i = 0; i < list.Count; i++)
                {
                    list[i] = PrepareBatch(list[i], device);
                }
                return list;

            default:
                return batch;
        }
    }

    public static T SampleFromDist<T>((string distribution, object[] args) distTuple, Random state = null)
    {
        state ??= new Random();

        return distTuple.distribution.ToLowerInvariant() switch
        {
            "const" => (T)distTuple.args[0],

            "uniform" => distTuple.args switch
            {
                var args when typeof(T) == typeof(float) || typeof(T) == typeof(double) =>
                                (T)(object)(float)((state.NextDouble() *
                                    ((float)args[1] - (float)args[0])) + (float)args[0]),

                var args when typeof(T) == typeof(int) =>
                    (T)(object)state.Next((int)args[0], (int)args[1]),

                _ => throw new ArgumentException($"Unsupported type {typeof(T)} for uniform distribution")
            },

            "normal" => distTuple.args switch
            {
                var args when typeof(T) == typeof(float) || typeof(T) == typeof(double) =>
                    (T)(object)(float)((SampleNormal(state) *
                        (float)args[1]) + (float)args[0]),

                _ => throw new ArgumentException($"Unsupported type {typeof(T)} for normal distribution")
            },

            _ => throw new ArgumentException($"Unknown distribution: {distTuple.distribution}")
        };
    }

    public static void SetSeed(int randomSeed, bool setCudnn = false)
    {
        manual_seed(randomSeed);
    }

    private static int RandomChoice(Random rng, int[] items, float[] probabilities)
    {
        var cumsum = new float[probabilities.Length];
        cumsum[0] = probabilities[0];
        for (int i = 1; i < probabilities.Length; i++)
        {
            cumsum[i] = cumsum[i - 1] + probabilities[i];
        }

        var value = (float)rng.NextDouble() * cumsum[^1];
        var index = Array.BinarySearch(cumsum, value);

        if (index < 0)
        {
            index = ~index;
        }
        return items[index];
    }

    private static double SampleNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private class ChangeDirectoryScope : IDisposable
    {
        private readonly string _originalDir;

        public ChangeDirectoryScope(string newDir)
        {
            _originalDir = Directory.GetCurrentDirectory();
            Directory.SetCurrentDirectory(newDir);
        }

        public void Dispose()
        {
            Directory.SetCurrentDirectory(_originalDir);
        }
    }
}