using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Loading
{
    [Flags]
    public enum ModelFileType
    {
        Unknown = 0,
        PyTorch = 1,              // .pt, .pth files using pickle protocol
        SafeTensors = 2,          // .safetensors files
        Checkpoint = 4,           // .ckpt files
        ONNX = 8,                // .onnx files
        TensorFlow = 16,         // .pb files
        TensorFlowLite = 32,     // .tflite files
        TorchScript = 64,        // .torchscript files
        TorchJit = 128,          // .jit files
        Config = 256,            // .json config files
        Weights = 512,           // .bin, .h5 weight files
        StateDict = 1024         // PyTorch state dictionary
    }
    public static class ModelFileTypeDetector
    {
        private static readonly Dictionary<string, ModelFileType> ExtensionMap = new()
        {
            { ".pt", ModelFileType.PyTorch },
            { ".pth", ModelFileType.PyTorch },
            { ".safetensors", ModelFileType.SafeTensors },
            { ".ckpt", ModelFileType.Checkpoint },
            { ".onnx", ModelFileType.ONNX },
            { ".pb", ModelFileType.TensorFlow },
            { ".tflite", ModelFileType.TensorFlowLite },
            { ".torchscript", ModelFileType.TorchScript },
            { ".jit", ModelFileType.TorchJit },
            { ".json", ModelFileType.Config },
            { ".bin", ModelFileType.Weights },
            { ".h5", ModelFileType.Weights }
        };

        private static readonly Dictionary<byte[], ModelFileType> SignatureMap = new()
        {
            { new byte[] { 0x80, 0x02 }, ModelFileType.PyTorch },                 // Pickle protocol
            { new byte[] { 0x50, 0x4B, 0x03, 0x04 }, ModelFileType.PyTorch },     // ZIP archive
            { new byte[] { 0x73, 0x61, 0x66, 0x65 }, ModelFileType.SafeTensors }, // "safe" in ASCII
            { new byte[] { 0x08, 0x00, 0x00, 0x00 }, ModelFileType.ONNX },        // ONNX proto
            { new byte[] { 0x50, 0x4B, 0x05, 0x06 }, ModelFileType.Checkpoint }   // ZIP central directory
        };

        /// <summary>
        /// Detects the model file type from a file path and optionally its contents
        /// </summary>
        /// <param name="filePath">Path to the model file</param>
        /// <param name="checkContents">Whether to also check file contents for signatures</param>
        /// <returns>Detected ModelFileType</returns>
        public static async Task<ModelFileType> DetectFileType(string filePath, bool checkContents = true)
        {
            var extension = Path.GetExtension(filePath).ToLowerInvariant();

            // First check extension
            if (ExtensionMap.TryGetValue(extension, out var typeFromExt))
            {
                // For config and weight files, extension is sufficient
                if (typeFromExt is ModelFileType.Config or ModelFileType.Weights)
                {
                    return typeFromExt;
                }
            }

            // Check file contents if requested
            if (checkContents && File.Exists(filePath))
            {
                try
                {
                    const int readSize = 16;
                    var buffer = new byte[readSize];

                    await using (var stream = File.OpenRead(filePath))
                    {
                        await stream.ReadAsync(buffer.AsMemory(0, readSize));
                    }

                    foreach (var (signature, fileType) in SignatureMap)
                    {
                        if (buffer.Take(signature.Length).SequenceEqual(signature))
                        {
                            return fileType;
                        }
                    }

                    // Special checks for specific formats
                    if (await IsPyTorchStateDict(filePath))
                    {
                        return ModelFileType.StateDict;
                    }
                }
                catch { /* fall back to extension-based detection */}
            }

            // Fall back to extension-based type if we found one
            return typeFromExt != 0 ? typeFromExt : ModelFileType.Unknown;
        }

        /// <summary>
        /// Checks if a file appears to be a PyTorch state dictionary by examining its structure
        /// </summary>
        private static async Task<bool> IsPyTorchStateDict(string filePath)
        {
            try
            {
                using var stream = File.OpenRead(filePath);
                using var reader = new BinaryReader(stream);

                // Check for pickle protocol
                var header = reader.ReadBytes(2);
                if (header[0] != 0x80) // Not a pickle
                    return false;

                // Check for typical state dict patterns
                var buffer = new byte[1024];
                var bytesRead = await stream.ReadAsync(buffer);
                var content = Encoding.ASCII.GetString(buffer, 0, bytesRead);

                var stateDictPatterns = new[]
                {
                    @"state_dict",
                    @"weight",
                    @"bias",
                    @"running_mean",
                    @"running_var",
                    @"layer\d+",
                    @"model\.",
                    @"module\."
                };

                return stateDictPatterns.Any(pattern =>
                    Regex.IsMatch(content, pattern, RegexOptions.IgnoreCase));
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Gets a user-friendly description of the model file type
        /// </summary>
        public static string GetFileTypeDescription(ModelFileType fileType)
        {
            return fileType switch
            {
                ModelFileType.PyTorch => "PyTorch Model",
                ModelFileType.SafeTensors => "SafeTensors Model",
                ModelFileType.Checkpoint => "Model Checkpoint",
                ModelFileType.ONNX => "ONNX Model",
                ModelFileType.TensorFlow => "TensorFlow Model",
                ModelFileType.TensorFlowLite => "TensorFlow Lite Model",
                ModelFileType.TorchScript => "TorchScript Model",
                ModelFileType.TorchJit => "TorchJit Model",
                ModelFileType.Config => "Configuration File",
                ModelFileType.Weights => "Weight File",
                ModelFileType.StateDict => "PyTorch State Dictionary",
                _ => "Unknown File Type"
            };
        }

        /// <summary>
        /// Checks if a file is a valid model file of any supported type
        /// </summary>
        public static async Task<bool> IsValidModelFile(string filePath)
        {
            var fileType = await DetectFileType(filePath);
            return fileType != ModelFileType.Unknown;
        }

        /// <summary>
        /// Gets the default extension for a given model file type
        /// </summary>
        public static string GetDefaultExtension(ModelFileType fileType)
        {
            return fileType switch
            {
                ModelFileType.PyTorch => ".pt",
                ModelFileType.SafeTensors => ".safetensors",
                ModelFileType.Checkpoint => ".ckpt",
                ModelFileType.ONNX => ".onnx",
                ModelFileType.TensorFlow => ".pb",
                ModelFileType.TensorFlowLite => ".tflite",
                ModelFileType.TorchScript => ".torchscript",
                ModelFileType.TorchJit => ".jit",
                ModelFileType.Config => ".json",
                ModelFileType.Weights => ".bin",
                ModelFileType.StateDict => ".pth",
                _ => ""
            };
        }
    }
}