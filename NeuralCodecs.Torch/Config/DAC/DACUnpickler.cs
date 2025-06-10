using Razorvine.Pickle;
using Razorvine.Pickle.Objects;
using System.Collections;
using System.IO.Compression;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Config.DAC;

/// <summary>
/// Provides functionality to unpickle and load DAC (Descript Audio Codec) model weights and configurations
/// from PyTorch-compatible pickle files and safetensors format.
/// </summary>
/// <remarks>
/// This class is based on TorchSharp.PyBridge.PytorchUnpickler and handles the deserialization of DAC models
/// that have been saved in Python PyTorch format. It supports both traditional pickle-based .pth files
/// and the newer safetensors format.
/// </remarks>
public class DACUnpickler
{
    /// <summary>
    /// Custom unpickler implementation that extends Razorvine.Pickle.Unpickler to handle DAC-specific tensor loading.
    /// </summary>
    private class CustomDACUnpickler : Unpickler
    {
        private readonly ZipArchive _archive;
        private bool _skipTensorRead;

        /// <summary>
        /// Initializes a new instance of the <see cref="CustomDACUnpickler"/> class.
        /// </summary>
        /// <param name="archive">The ZIP archive containing the model data.</param>
        /// <param name="skipTensorRead">If true, tensors will not be immediately loaded into memory.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="archive"/> is null.</exception>
        public CustomDACUnpickler(ZipArchive archive, bool skipTensorRead = false)
        {
            _archive = archive ?? throw new ArgumentNullException(nameof(archive));
            _skipTensorRead = skipTensorRead;
        }

        /// <summary>
        /// Handles persistent loading of tensor storage data from the pickle stream.
        /// </summary>
        /// <param name="pid">The persistent ID object containing storage information.</param>
        /// <returns>An <see cref="NCTensorStream"/> object representing the tensor data.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the persistent load data is invalid or archive entry is not found.</exception>
        /// <exception cref="NotImplementedException">Thrown when an unknown persistent ID type is encountered.</exception>
        protected override object persistentLoad(object pid)
        {
            if (pid is not object[] array)
                throw new InvalidOperationException("Invalid persistent load data");

            if ((string)array[0] != "storage")
                throw new NotImplementedException("Unknown persistent id loaded");

            string name = ((ClassDictConstructor)array[1]).name;
            string archiveKey = (string)array[2];
            var scalarType = GetScalarTypeFromStorageName(name);

            var entry = _archive.Entries
                .Select((e, i) => (Entry: e, Index: i))
                .FirstOrDefault(e => e.Entry.FullName.EndsWith("data/" + archiveKey));

            if (entry.Entry == null)
                throw new InvalidOperationException($"Archive entry not found: data/{archiveKey}");

            return new NCTensorStream
            {
                ArchiveIndex = entry.Index,
                ArchiveEntry = entry.Entry,
                DType = scalarType,
                SkipTensorRead = _skipTensorRead
            };
        }
    }

    /// <summary>
    /// Maps PyTorch storage type names to corresponding TorchSharp scalar types.
    /// </summary>
    /// <param name="storage">The PyTorch storage type name (e.g., "FloatStorage", "DoubleStorage").</param>
    /// <returns>The corresponding <see cref="torch.ScalarType"/>.</returns>
    /// <exception cref="NotImplementedException">Thrown when the storage type is not supported.</exception>
    private static torch.ScalarType GetScalarTypeFromStorageName(string storage)
    {
        return storage switch
        {
            "DoubleStorage" => torch.float64,
            "FloatStorage" => torch.@float,
            "HalfStorage" => torch.half,
            "LongStorage" => torch.@long,
            "IntStorage" => torch.@int,
            "ShortStorage" => torch.int16,
            "CharStorage" => torch.int8,
            "ByteStorage" => torch.uint8,
            "BoolStorage" => torch.@bool,
            "BFloat16Storage" => torch.bfloat16,
            "ComplexDoubleStorage" => torch.cdouble,
            "ComplexFloatStorage" => torch.cfloat,
            _ => throw new NotImplementedException(),
        };
    }

    /// <summary>
    /// Represents an ordered dictionary specifically designed for DAC model state dictionaries.
    /// Extends <see cref="Hashtable"/> to provide tensor conversion capabilities.
    /// </summary>
    private class DACOrderedDict : Hashtable
    {
        /// <summary>
        /// Converts the ordered dictionary to a strongly-typed dictionary of tensors.
        /// </summary>
        /// <returns>A dictionary with string keys and <see cref="Tensor"/> values.</returns>
        /// <exception cref="InvalidOperationException">Thrown when a key is null in the state dictionary.</exception>
        public Dictionary<string, Tensor> AsTensorDict()
        {
            var dict = new Dictionary<string, Tensor>();
            foreach (DictionaryEntry entry in this)
            {
                var key = entry.Key?.ToString() ??
                    throw new InvalidOperationException("Null key in state dict");

                if (entry.Value is Tensor tensor)
                {
                    dict[key] = tensor;
                }
                else if (entry.Value is NCTensorConstructorArgs args)
                {
                    dict[key] = args.ReadTensorFromStream();
                }
            }
            return dict;
        }

        /// <summary>
        /// Sets the state of the ordered dictionary from a hashtable, typically called during unpickling.
        /// </summary>
        /// <param name="state">The hashtable containing the state data to restore.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="state"/> is null.</exception>
        public void __setstate__(Hashtable state)
        {
            ArgumentNullException.ThrowIfNull(state);

            foreach (DictionaryEntry entry in state)
            {
                this[entry.Key] = entry.Value;
            }
        }
    }

    /// <summary>
    /// Contains the arguments and data needed to construct a tensor from a stream during unpickling.
    /// </summary>
    internal class NCTensorConstructorArgs
    {
        private bool _alreadyRead;

        /// <summary>
        /// Gets or sets the index of the archive entry containing the tensor data.
        /// </summary>
        public int ArchiveIndex { get; init; }

        /// <summary>
        /// Gets or sets the stream containing the tensor's raw data.
        /// </summary>
        public Stream Data { get; init; }

        /// <summary>
        /// Gets or sets the scalar type of the tensor data.
        /// </summary>
        public torch.ScalarType DType { get; init; }

        /// <summary>
        /// Gets or sets the storage offset for the tensor data.
        /// </summary>
        public int StorageOffset { get; init; }

        /// <summary>
        /// Gets or sets the shape dimensions of the tensor.
        /// </summary>
        public long[] Shape { get; init; }

        /// <summary>
        /// Gets or sets the stride values for the tensor.
        /// </summary>
        public long[] Stride { get; init; }

        /// <summary>
        /// Gets or sets a value indicating whether the tensor requires gradient computation.
        /// </summary>
        public bool RequiresGrad { get; init; }

        /// <summary>
        /// Reads the tensor data from the stream and constructs a <see cref="torch.Tensor"/>.
        /// </summary>
        /// <returns>A new <see cref="torch.Tensor"/> instance with the data loaded from the stream.</returns>
        /// <exception cref="InvalidOperationException">Thrown when attempting to read the tensor more than once.</exception>
        public torch.Tensor ReadTensorFromStream()
        {
            if (_alreadyRead)
            {
                throw new InvalidOperationException("The tensor has already been constructed, cannot read tensor twice.");
            }

            torch.Tensor tensor = torch.empty(Shape, DType, torch.CPU).as_strided(Shape, Stride, StorageOffset);
            tensor.ReadBytesFromStream(Data);
            Data.Close();
            _alreadyRead = true;
            return tensor;
        }
    }

    /// <summary>
    /// Represents a stream reference to tensor data within a ZIP archive.
    /// </summary>
    private class NCTensorStream
    {
        /// <summary>
        /// Gets or sets the index of the archive entry.
        /// </summary>
        public int ArchiveIndex { get; init; }

        /// <summary>
        /// Gets or sets the ZIP archive entry containing the tensor data.
        /// </summary>
        public ZipArchiveEntry ArchiveEntry { get; init; }

        /// <summary>
        /// Gets or sets the scalar type of the tensor data.
        /// </summary>
        public torch.ScalarType DType { get; init; }

        /// <summary>
        /// Gets or sets a value indicating whether tensor reading should be skipped.
        /// </summary>
        public bool SkipTensorRead { get; init; }
    }

    /// <summary>
    /// Object constructor that creates tensor objects during the unpickling process.
    /// Implements <see cref="IObjectConstructor"/> to handle PyTorch tensor reconstruction.
    /// </summary>
    private class NCTensorObjectConstructor : IObjectConstructor
    {
        /// <summary>
        /// Constructs a tensor object from the provided arguments during unpickling.
        /// </summary>
        /// <param name="args">Array containing tensor construction arguments including stream, offset, shape, stride, and gradient requirements.</param>
        /// <returns>Either a <see cref="torch.Tensor"/> or <see cref="NCTensorConstructorArgs"/> depending on the skip tensor read setting.</returns>
        public object construct(object[] args)
        {
            NCTensorStream tensorStream = (NCTensorStream)args[0];
            NCTensorConstructorArgs tensorConstructorArgs = new NCTensorConstructorArgs
            {
                ArchiveIndex = tensorStream.ArchiveIndex,
                Data = tensorStream.ArchiveEntry.Open(),
                DType = tensorStream.DType,
                StorageOffset = (int)args[1],
                Shape = ((IEnumerable<object>)(object[])args[2]).Select((Func<object, long>)((object i) => (int)i)).ToArray(),
                Stride = ((IEnumerable<object>)(object[])args[3]).Select((Func<object, long>)((object i) => (int)i)).ToArray(),
                RequiresGrad = (bool)args[4]
            };
            if (!tensorStream.SkipTensorRead)
            {
                return tensorConstructorArgs.ReadTensorFromStream();
            }

            return tensorConstructorArgs;
        }
    }

    /// <summary>
    /// Object constructor that creates <see cref="DACOrderedDict"/> instances during unpickling.
    /// Implements <see cref="IObjectConstructor"/> to handle ordered dictionary reconstruction.
    /// </summary>
    private class DACDictConstructor : IObjectConstructor
    {
        /// <summary>
        /// Constructs a new <see cref="DACOrderedDict"/> instance.
        /// </summary>
        /// <param name="args">Construction arguments (currently unused).</param>
        /// <returns>A new <see cref="DACOrderedDict"/> instance.</returns>
        public object construct(object[] args)
        {
            return new DACOrderedDict();
        }
    }

    /// <summary>
    /// Static constructor that registers custom object constructors with the unpickler.
    /// </summary>
    static DACUnpickler()
    {
        // Register our custom constructors
        Unpickler.registerConstructor("torch._utils", "_rebuild_tensor", new NCTensorObjectConstructor());
        Unpickler.registerConstructor("torch._utils", "_rebuild_tensor_v2", new NCTensorObjectConstructor());
        Unpickler.registerConstructor("collections", "OrderedDict", new DACDictConstructor());
    }

    /// <summary>
    /// Loads DAC model weights from a file, supporting both pickle (.pth) and safetensors formats.
    /// </summary>
    /// <param name="path">The file path to the model weights.</param>
    /// <returns>A <see cref="DACWeights"/> object containing the loaded model weights and metadata.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="path"/> is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the specified file does not exist.</exception>
    public static DACWeights LoadFromFile(string path)
    {
        if (string.IsNullOrEmpty(path))
            throw new ArgumentNullException(nameof(path));

        if (!File.Exists(path))
            throw new FileNotFoundException("Model weights file not found", path);

        // Handle different file formats
        if (path.EndsWith(".safetensors"))
            return LoadFromSafetensors(path);

        return LoadFromStream(File.OpenRead(path));
    }

    /// <summary>
    /// Loads DAC model weights from a safetensors format file.
    /// </summary>
    /// <param name="path">The file path to the safetensors file.</param>
    /// <returns>A <see cref="DACWeights"/> object containing the loaded model weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the safetensors file cannot be loaded.</exception>
    public static DACWeights LoadFromSafetensors(string path)
    {
        var previousDevice = torch.get_default_device();
        set_default_device(CPU);
        var stateDict = Safetensors.LoadStateDict(path) ??
            throw new InvalidOperationException("Failed to load safetensor weights");

        var normalizedDict = StateDictNameConverter.ConvertStateDict(stateDict, fromSafetensor: true);

        if (previousDevice != CPU)
        {
            set_default_device(previousDevice);
        }
        return new DACWeights(normalizedDict);
    }

    /// <summary>
    /// Loads DAC model weights from a stream containing pickle data in ZIP format.
    /// </summary>
    /// <param name="stream">The stream containing the model data.</param>
    /// <param name="leaveOpen">If true, the stream will not be closed after reading.</param>
    /// <returns>A <see cref="DACWeights"/> object containing the loaded model weights and metadata.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="stream"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the stream is not readable.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the file format is invalid or required data is missing.</exception>
    public static DACWeights LoadFromStream(Stream stream, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable", nameof(stream));

        // Verify zip magic number
        byte[] magic = new byte[4];
        if (stream.Read(magic, 0, 4) != 4)
            throw new InvalidOperationException("Failed to read magic number");

        stream.Seek(0, SeekOrigin.Begin);

        if (magic[0] != 0x50 || magic[1] != 0x4B || magic[2] != 0x03 || magic[3] != 0x04)
        {
            throw new InvalidOperationException(
                "Invalid .pth file format - must be a zip archive. File may be corrupted or saved in legacy format.");
        }

        using var archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen);
        var entry = archive.Entries.FirstOrDefault(e => e.Name.EndsWith("data.pkl")) ??
            throw new InvalidOperationException("Model archive missing data.pkl");

        var unpickler = new CustomDACUnpickler(archive);
        var result = unpickler.load(entry.Open()) as Hashtable ??
            throw new InvalidOperationException("Failed to unpickle model data");

        // Extract state dict
        var stateDict = result["state_dict"] as DACOrderedDict ??
            throw new InvalidOperationException("Missing or invalid state_dict");
        var weights = stateDict.AsTensorDict();

        // Extract metadata
        var metadata = result["metadata"] as Hashtable ??
            throw new InvalidOperationException("Missing or invalid metadata");
        var convertedMetadata = ConvertToMetadataDict(metadata);

        return new DACWeights(weights, convertedMetadata);
    }

    /// <summary>
    /// Loads DAC model weights and creates a configuration from the embedded metadata.
    /// </summary>
    /// <param name="path">The file path to the model weights.</param>
    /// <returns>A tuple containing the loaded <see cref="DACWeights"/> and the generated <see cref="DACConfig"/>.</returns>
    public static (DACWeights Weights, DACConfig Config) LoadWithConfig(string path)
    {
        var weights = LoadFromFile(path);
        var config = CreateConfigFromMetadata(weights);
        return (weights, config);
    }

    /// <summary>
    /// Creates a DAC configuration object from the metadata contained in the loaded weights.
    /// </summary>
    /// <param name="weights">The DAC weights containing metadata to extract configuration from.</param>
    /// <returns>A <see cref="DACConfig"/> object with parameters extracted from the metadata, or default values if metadata is missing.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="weights"/> is null.</exception>
    public static DACConfig CreateConfigFromMetadata(DACWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (weights.Metadata is null || weights.Metadata.Count == 0)
        {
            return new DACConfig();
        }

        return new DACConfig
        {
            SampleRate = GetMetadataValue<int>(weights.Metadata, "sampling_rate", 44100),
            EncoderDim = GetMetadataValue<int>(weights.Metadata, "encoder_dim", 64),
            LatentDim = GetMetadataValue<int?>(weights.Metadata, "latent_dim"),
            EncoderRates = GetMetadataValue<int[]>(weights.Metadata, "encoder_rates", [2, 4, 8, 8]),
            DecoderRates = GetMetadataValue<int[]>(weights.Metadata, "decoder_rates", [8, 8, 4, 2]),
            DecoderDim = GetMetadataValue<int>(weights.Metadata, "decoder_dim", 1536),
            NumCodebooks = GetMetadataValue<int>(weights.Metadata, "n_codebooks", 9),
            CodebookSize = GetMetadataValue<int>(weights.Metadata, "codebook_size", 4096),
            CodebookDim = GetMetadataValue<int>(weights.Metadata, "codebook_dim", 8),
            QuantizerDropout = GetMetadataValue<float>(weights.Metadata, "quantizer_dropout", 0.0f),
        };
    }

    /// <summary>
    /// Converts a hashtable containing metadata to a strongly-typed dictionary.
    /// </summary>
    /// <param name="metadata">The hashtable containing the metadata to convert.</param>
    /// <returns>A dictionary with string keys and object values, with nested hashtables recursively converted.</returns>
    private static Dictionary<string, object> ConvertToMetadataDict(Hashtable metadata)
    {
        var dict = new Dictionary<string, object>();

        foreach (DictionaryEntry entry in metadata)
        {
            var key = entry.Key?.ToString();
            var value = entry.Value;

            if (key == null)
            {
                continue;
            }

            if (value is Hashtable subTable)
            {
                dict[key] = ConvertToMetadataDict(subTable);
            }
            else
            {
                dict[key] = value;
            }
        }

        return dict;
    }

    /// <summary>
    /// Retrieves a strongly-typed value from the metadata dictionary with optional default value.
    /// </summary>
    /// <typeparam name="T">The type to convert the metadata value to.</typeparam>
    /// <param name="metadata">The metadata dictionary to search.</param>
    /// <param name="key">The key to look up in the metadata.</param>
    /// <param name="defaultValue">The default value to return if the key is not found or conversion fails.</param>
    /// <returns>The converted value if found and successfully converted, otherwise the default value.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="metadata"/> is null or <paramref name="key"/> is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when type conversion fails.</exception>
    private static T GetMetadataValue<T>(
            Dictionary<string, object> metadata,
            string key,
            T defaultValue = default)
    {
        ArgumentNullException.ThrowIfNull(metadata);

        if (string.IsNullOrEmpty(key))
            throw new ArgumentNullException(nameof(key));

        if (!metadata.TryGetValue(key, out var value))
            return defaultValue;

        try
        {
            if (typeof(T).IsArray && value is Array arr)
            {
                var elementType = typeof(T).GetElementType();
                var converted = Array.CreateInstance(elementType, arr.Length);
                for (int i = 0; i < arr.Length; i++)
                {
                    converted.SetValue(Convert.ChangeType(arr.GetValue(i), elementType), i);
                }
                return (T)(object)converted;
            }

            return (T)Convert.ChangeType(value, typeof(T));
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to convert metadata value for key '{key}' to type {typeof(T).Name}", ex);
        }
    }
}