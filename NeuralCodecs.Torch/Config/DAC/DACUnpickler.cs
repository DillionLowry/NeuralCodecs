using Razorvine.Pickle;
using Razorvine.Pickle.Objects;
using System.Collections;
using System.IO.Compression;
using TorchSharp;
using static TorchSharp.torch;

// Based on TorchSharp.PyBridge.PytorchUnpickler
namespace NeuralCodecs.Torch.Config.DAC
{
    public class DACUnpickler
    {
        private class CustomUnpickler : Unpickler
        {
            private readonly ZipArchive _archive;
            private bool _skipTensorRead;

            public CustomUnpickler(ZipArchive archive, bool skipTensorRead = false)
            {
                _archive = archive;
                _skipTensorRead = skipTensorRead;
            }

            protected override object persistentLoad(object pid)
            {
                var array = (object[])pid;
                if ((string)array[0] != "storage")
                {
                    throw new NotImplementedException("Unknown persistent id loaded");
                }

                string name = ((ClassDictConstructor)array[1]).name;
                string archiveKey = (string)array[2];
                var scalarType = GetScalarTypeFromStorageName(name);

                var entry = _archive.Entries
                    .Select((e, i) => (e, i))
                    .First(e => e.e.FullName.EndsWith("data/" + archiveKey));

                return new TensorStream
                {
                    ArchiveIndex = entry.i,
                    ArchiveEntry = entry.e,
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
                    if (entry.Value is Tensor tensor)
                    {
                        dict[entry.Key.ToString()] = tensor;
                    }
                    else if (entry.Value is TensorConstructorArgs args)
                    {
                        dict[entry.Key.ToString()] = args.ReadTensorFromStream();
                    }
                }
                return dict;
            }

            public void __setstate__(Hashtable state)
            {
                foreach (DictionaryEntry entry in state)
                {
                    this[entry.Key] = entry.Value;
                }
            }
        }

        internal class TensorConstructorArgs
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

        private class TensorStream
        {
            public int ArchiveIndex { get; init; }

            public ZipArchiveEntry ArchiveEntry { get; init; }

            public torch.ScalarType DType { get; init; }

            public bool SkipTensorRead { get; init; }
        }

        private class TensorObjectConstructor : IObjectConstructor
        {
            public object construct(object[] args)
            {
                TensorStream tensorStream = (TensorStream)args[0];
                TensorConstructorArgs tensorConstructorArgs = new TensorConstructorArgs
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
            Unpickler.registerConstructor("torch._utils", "_rebuild_tensor", new TensorObjectConstructor());
            Unpickler.registerConstructor("torch._utils", "_rebuild_tensor_v2", new TensorObjectConstructor());
            Unpickler.registerConstructor("collections", "OrderedDict", new DACDictConstructor());
        }

        public static DACWeights LoadFromFile(string path)
        {
            using var stream = File.OpenRead(path);
            return LoadFromStream(stream);
        }

        public static DACWeights LoadFromStream(Stream stream, bool leaveOpen = false)
        {
            // Verify zip magic number
            byte[] magic = new byte[4];
            stream.Read(magic, 0, 4);
            stream.Seek(0, SeekOrigin.Begin);

            if (magic[0] != 0x50 || magic[1] != 0x4B || magic[2] != 0x03 || magic[3] != 0x04)
            {
                throw new InvalidOperationException("Invalid .pth file format - must be a zip archive");
            }

            using var archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen);
            var entry = archive.Entries.First(e => e.Name.EndsWith("data.pkl"));

            var unpickler = new CustomUnpickler(archive);

            if (unpickler.load(entry.Open()) is not Hashtable result)
                throw new InvalidOperationException("Failed to load model weights");

            var weights = new DACWeights();

            // Extract state dict
            if (result["state_dict"] is DACOrderedDict stateDict)
            {
                weights.StateDict = stateDict.AsTensorDict();
            }
            else
            {
                throw new InvalidOperationException("Missing or invalid state_dict");
            }

            // Extract metadata
            if (result["metadata"] is Hashtable metadata)
            {
                weights.Metadata = ConvertToMetadataDict(metadata);
            }

            return weights;
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
                    continue; // Skip null keys
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

        // Helper method to safely get metadata values
        public static T GetMetadataValue<T>(Dictionary<string, object> metadata, string key, T defaultValue = default)
        {
            if (!metadata.TryGetValue(key, out object? value))
                return defaultValue;

            try
            {
                return (T)Convert.ChangeType(value, typeof(T));
            }
            catch
            {
                return defaultValue;
            }
        }
    }
}