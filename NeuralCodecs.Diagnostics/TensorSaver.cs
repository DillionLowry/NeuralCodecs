using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralCodecs.Diagnostics
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Runtime.InteropServices;
    using TorchSharp;
    using static TorchSharp.torch;

    namespace NeuralCodecs.Diagnostics
    {
        /// <summary>
        /// Provides advanced tensor saving capabilities in various formats
        /// </summary>
        public class TensorSaver
        {
            /// <summary>
            /// Supported file formats for tensor data
            /// </summary>
            public enum TensorFormat
            {
                /// <summary>Custom binary format</summary>
                Binary,
                /// <summary>NumPy compatible format</summary>
                Numpy,
                /// <summary>CSV text format</summary>
                Csv,
                /// <summary>JSON text format</summary>
                Json,
                /// <summary>Raw binary values</summary>
                Raw
            }

            private readonly string _outputDirectory;

            /// <summary>
            /// Creates a new TensorSaver
            /// </summary>
            /// <param name="outputDirectory">Directory to save tensor files</param>
            public TensorSaver(string outputDirectory)
            {
                _outputDirectory = outputDirectory;
                Directory.CreateDirectory(outputDirectory);
            }

            /// <summary>
            /// Saves a tensor to file in the specified format
            /// </summary>
            /// <param name="tensor">The tensor to save</param>
            /// <param name="name">Name for the file (without extension)</param>
            /// <param name="format">Format to use for saving</param>
            /// <returns>Path to the saved file</returns>
            public string SaveTensor(Tensor tensor, string name, TensorFormat format = TensorFormat.Binary)
            {
                if (tensor.IsInvalid)
                    throw new ArgumentException("Cannot save an invalid tensor", nameof(tensor));

                using var scope = torch.NewDisposeScope();
                var detachedTensor = tensor.detach().cpu().clone().MoveToOuterDisposeScope();

                var extension = GetExtensionForFormat(format);
                var filePath = Path.Combine(_outputDirectory, $"{name.Replace('.', '_')}{extension}");

                switch (format)
                {
                    case TensorFormat.Binary:
                        SaveTensorAsBinary(detachedTensor, filePath);
                        break;
                    case TensorFormat.Numpy:
                        SaveTensorAsNumpy(detachedTensor, filePath);
                        break;
                    case TensorFormat.Csv:
                        SaveTensorAsCsv(detachedTensor, filePath);
                        break;
                    case TensorFormat.Json:
                        SaveTensorAsJson(detachedTensor, filePath);
                        break;
                    case TensorFormat.Raw:
                        SaveTensorAsRaw(detachedTensor, filePath);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(format), format, "Unsupported tensor format");
                }

                return filePath;
            }

            /// <summary>
            /// Gets the appropriate file extension for a tensor format
            /// </summary>
            private string GetExtensionForFormat(TensorFormat format) => format switch
            {
                TensorFormat.Binary => ".bin",
                TensorFormat.Numpy => ".npy",
                TensorFormat.Csv => ".csv",
                TensorFormat.Json => ".json",
                TensorFormat.Raw => ".raw",
                _ => ".bin"
            };

            /// <summary>
            /// Saves tensor in custom binary format with header
            /// </summary>
            private void SaveTensorAsBinary(Tensor tensor, string filePath)
            {
                using var fs = File.Create(filePath);
                using var writer = new BinaryWriter(fs);

                // Write header - format:
                // - Magic number (0x1E2D3C4B)
                // - Number of dimensions
                // - Shape dimensions
                // - Data type code
                // - Data

                writer.Write(0x1E2D3C4B); // Magic number
                writer.Write(tensor.shape.Length);
                foreach (var dim in tensor.shape)
                {
                    writer.Write(dim);
                }

                // Data type code (matching numpy dtypes)
                var dtype = GetDTypeCode(tensor.dtype);
                writer.Write((byte)dtype);

                // Write data based on type
                switch (tensor.dtype)
                {
                    case ScalarType.Float32:
                        WriteTypedData<float>(tensor, writer);
                        break;
                    case ScalarType.Float64:
                        WriteTypedData<double>(tensor, writer);
                        break;
                    case ScalarType.Int32:
                        WriteTypedData<int>(tensor, writer);
                        break;
                    case ScalarType.Int64:
                        WriteTypedData<long>(tensor, writer);
                        break;
                    default:
                        // Convert to float32 and write
                        var floatTensor = tensor.to(ScalarType.Float32);
                        WriteTypedData<float>(floatTensor, writer);
                        break;
                }
            }

            private void WriteTypedData<T>(Tensor tensor, BinaryWriter writer)
                where T : unmanaged // non-nullable value types
            {
                var flatTensor = tensor.flatten();
                var data = flatTensor.data<T>().ToArray();
                var byteSize = Marshal.SizeOf<T>();
                var buffer = new byte[data.Length * byteSize];

                Buffer.BlockCopy(data, 0, buffer, 0, buffer.Length);
                writer.Write(buffer);
            }

            /// <summary>
            /// Saves tensor in NumPy compatible format
            /// </summary>
            private void SaveTensorAsNumpy(Tensor tensor, string filePath)
            {
                using var fs = File.Create(filePath);
                using var writer = new BinaryWriter(fs);

                // Simple implementation of numpy format
                // https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

                // Magic string: \x93NUMPY
                writer.Write(new byte[] { 0x93 });
                writer.Write(Encoding.ASCII.GetBytes("NUMPY"));

                // Version
                writer.Write((byte)1);
                writer.Write((byte)0);

                // Build header dict
                var shapeStr = "(" + string.Join(", ", tensor.shape.Select(d => d.ToString())) +
                               (tensor.shape.Length == 1 ? ",)" : ")");

                string dtypeStr = tensor.dtype switch
                {
                    ScalarType.Float32 => "<f4",
                    ScalarType.Float64 => "<f8",
                    ScalarType.Int32 => "<i4",
                    ScalarType.Int64 => "<i8",
                    _ => "<f4" // Default to float32
                };

                var headerDict = $"{{'descr': '{dtypeStr}', 'fortran_order': False, 'shape': {shapeStr}, }}";

                // Pad to 64-byte alignment
                int headerLen = headerDict.Length;
                int padLen = 64 - ((headerLen + 10) % 64);
                if (padLen < 16) padLen += 64;

                var paddedHeader = headerDict.PadRight(headerLen + padLen, ' ');

                // Write header length
                writer.Write((ushort)paddedHeader.Length);

                // Write header
                writer.Write(Encoding.ASCII.GetBytes(paddedHeader));

                // Write data
                switch (tensor.dtype)
                {
                    case ScalarType.Float32:
                        WriteTypedData<float>(tensor, writer);
                        break;
                    case ScalarType.Float64:
                        WriteTypedData<double>(tensor, writer);
                        break;
                    case ScalarType.Int32:
                        WriteTypedData<int>(tensor, writer);
                        break;
                    case ScalarType.Int64:
                        WriteTypedData<long>(tensor, writer);
                        break;
                    default:
                        // Convert to float32 and write
                        var floatTensor = tensor.to(ScalarType.Float32);
                        WriteTypedData<float>(floatTensor, writer);
                        break;
                }
            }
            /// <summary>
            /// Saves tensor in CSV format
            /// </summary>
            private void SaveTensorAsCsv(Tensor tensor, string filePath)
            {
                using var writer = new StreamWriter(filePath);
                var flatTensor = tensor.flatten();

                switch (tensor.dtype)
                {
                    case ScalarType.Float32:
                        var floatData = flatTensor.data<float>().ToArray();
                        writer.WriteLine(string.Join(",", floatData));
                        break;
                    case ScalarType.Float64:
                        var doubleData = flatTensor.data<double>().ToArray();
                        writer.WriteLine(string.Join(",", doubleData));
                        break;
                    case ScalarType.Int32:
                        var intData = flatTensor.data<int>().ToArray();
                        writer.WriteLine(string.Join(",", intData));
                        break;
                    case ScalarType.Int64:
                        var longData = flatTensor.data<long>().ToArray();
                        writer.WriteLine(string.Join(",", longData));
                        break;
                    default:
                        var defaultData = tensor.to(ScalarType.Float32).flatten().data<float>().ToArray();
                        writer.WriteLine(string.Join(",", defaultData));
                        break;
                }
            }

            /// <summary>
            /// Saves tensor in JSON format
            /// </summary>
            private void SaveTensorAsJson(Tensor tensor, string filePath)
            {
                using var writer = new StreamWriter(filePath);
                var flatTensor = tensor.flatten();

                writer.Write("{\"shape\":[");
                writer.Write(string.Join(",", tensor.shape));
                writer.Write("],\"dtype\":\"");
                writer.Write(tensor.dtype.ToString());
                writer.Write("\",\"data\":[");

                switch (tensor.dtype)
                {
                    case ScalarType.Float32:
                        writer.Write(string.Join(",", flatTensor.data<float>().ToArray()));
                        break;
                    case ScalarType.Float64:
                        writer.Write(string.Join(",", flatTensor.data<double>().ToArray()));
                        break;
                    case ScalarType.Int32:
                        writer.Write(string.Join(",", flatTensor.data<int>().ToArray()));
                        break;
                    case ScalarType.Int64:
                        writer.Write(string.Join(",", flatTensor.data<long>().ToArray()));
                        break;
                    default:
                        writer.Write(string.Join(",", tensor.to(ScalarType.Float32).flatten().data<float>().ToArray()));
                        break;
                }
                writer.Write("]}");
            }

            /// <summary>
            /// Saves tensor as raw binary data without headers
            /// </summary>
            private void SaveTensorAsRaw(Tensor tensor, string filePath)
            {
                using var fs = File.Create(filePath);
                using var writer = new BinaryWriter(fs);

                switch (tensor.dtype)
                {
                    case ScalarType.Float32:
                        WriteTypedData<float>(tensor, writer);
                        break;
                    case ScalarType.Float64:
                        WriteTypedData<double>(tensor, writer);
                        break;
                    case ScalarType.Int32:
                        WriteTypedData<int>(tensor, writer);
                        break;
                    case ScalarType.Int64:
                        WriteTypedData<long>(tensor, writer);
                        break;
                    default:
                        WriteTypedData<float>(tensor.to(ScalarType.Float32), writer);
                        break;
                }
            }

            /// <summary>
            /// Gets the data type code for numpy compatibility
            /// </summary>
            private static byte GetDTypeCode(ScalarType dtype) => dtype switch
            {
                ScalarType.Float32 => 0x17,  // 'f4'
                ScalarType.Float64 => 0x18,  // 'f8'
                ScalarType.Int32 => 0x13,    // 'i4'
                ScalarType.Int64 => 0x14,    // 'i8'
                _ => 0x17                    // Default to float32
            };

        }
    }
}
