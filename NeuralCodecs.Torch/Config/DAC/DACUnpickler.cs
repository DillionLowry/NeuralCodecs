using Razorvine.Pickle;
using Razorvine.Pickle.Objects;
using System.Collections;
using System.IO.Compression;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

// Based on TorchSharp.PyBridge.PytorchUnpickler
// NC prefix is used to avoid conflicts with existing classes
namespace NeuralCodecs.Torch.Config.DAC
{
    public class DACUnpickler
    {
        private class CustomDACUnpickler : Unpickler
        {
            private readonly ZipArchive _archive;
            private bool _skipTensorRead;

            public CustomDACUnpickler(ZipArchive archive, bool skipTensorRead = false)
            {
                _archive = archive ?? throw new ArgumentNullException(nameof(archive));
                _skipTensorRead = skipTensorRead;
            }

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

        private class DACOrderedDict : Hashtable
        {
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

            public void __setstate__(Hashtable state)
            {
                ArgumentNullException.ThrowIfNull(state);

                foreach (DictionaryEntry entry in state)
                {
                    this[entry.Key] = entry.Value;
                }
            }
        }

        internal class NCTensorConstructorArgs
        {
            private bool _alreadyRead;

            public int ArchiveIndex { get; init; }

            public Stream Data { get; init; }

            public torch.ScalarType DType { get; init; }

            public int StorageOffset { get; init; }

            public long[] Shape { get; init; }

            public long[] Stride { get; init; }

            public bool RequiresGrad { get; init; }

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

        private class NCTensorStream
        {
            public int ArchiveIndex { get; init; }

            public ZipArchiveEntry ArchiveEntry { get; init; }

            public torch.ScalarType DType { get; init; }

            public bool SkipTensorRead { get; init; }
        }

        private class NCTensorObjectConstructor : IObjectConstructor
        {
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

        private class DACDictConstructor : IObjectConstructor
        {
            public object construct(object[] args)
            {
                return new DACOrderedDict();
            }
        }

        static DACUnpickler()
        {
            // Register our custom constructors
            Unpickler.registerConstructor("torch._utils", "_rebuild_tensor", new NCTensorObjectConstructor());
            Unpickler.registerConstructor("torch._utils", "_rebuild_tensor_v2", new NCTensorObjectConstructor());
            Unpickler.registerConstructor("collections", "OrderedDict", new DACDictConstructor());
        }

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

        public static DACWeights LoadFromSafetensors(string path)
        {
            var stateDict = Safetensors.LoadStateDict(path) ??
                throw new InvalidOperationException("Failed to load safetensor weights");

            // Convert weights using dynamic converter
            var normalizedDict = StateDictNameConverter.ConvertStateDict(stateDict, fromSafetensor: true);

            return new DACWeights(normalizedDict);
        }

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

        public static (DACWeights Weights, DACConfig Config) LoadWithConfig(string path)
        {
            var weights = LoadFromFile(path);
            var config = CreateConfigFromMetadata(weights);
            return (weights, config);
        }

        public static DACConfig CreateConfigFromMetadata(DACWeights weights)
        {
            ArgumentNullException.ThrowIfNull(weights);
            if (weights.Metadata is null || weights.Metadata.Count == 0)
            {
                return new DACConfig();
            }

            return new DACConfig
            {
                SamplingRate = GetMetadataValue<int>(weights.Metadata, "sampling_rate", 44100),
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
}