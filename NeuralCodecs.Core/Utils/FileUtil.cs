using NeuralCodecs.Core.Loading;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Utils
{
    internal class FileUtil
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
        /// <returns>Detected ModelFileType</returns>
        public static ModelFileType DetectFileType(string filePath)
        {
            var extension = Path.GetExtension(filePath).ToLowerInvariant();
           
            // First check extension
            if (ExtensionMap.TryGetValue(extension, out var typeFromExt))
            {
                return typeFromExt;
            }
            return ModelFileType.Unknown;
        }
        /// <summary>
        /// Detects the model file type from a file path and optionally its contents
        /// </summary>
        /// <param name="filePath">Path to the model file</param>
        /// <param name="checkContents">Whether to also check file contents for signatures</param>
        /// <returns>Detected ModelFileType</returns>
        public static async Task<ModelFileType> DetectFileTypeFromContentsAsync(string filePath)
        {
            var extension = Path.GetExtension(filePath).ToLowerInvariant();

            // First check extension
            if (ExtensionMap.TryGetValue(extension, out var typeFromExt))
            {
                //For config and weight files, extension is sufficient
                if (typeFromExt is ModelFileType.Config or ModelFileType.Weights)
                {
                    return typeFromExt;
                }
            }

            // Check file contents if requested
            if (File.Exists(filePath))
            {
                try
                {
                    const int readSize = 16;
                    var buffer = new byte[readSize];

                    using (var stream = File.OpenRead(filePath))
                    {
                        await stream.ReadAsync(buffer.AsMemory(0, readSize));
                    }

                    // Check each signature
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
                catch
                {
                    // If we can't read the file, fall back to extension-based detection
                }
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

                // Read first few bytes to check for pickle protocol
                var header = reader.ReadBytes(2);
                if (header[0] != 0x80) // Not a pickle
                    return false;

                // Read a small portion of the file to check for typical state dict patterns
                var buffer = new byte[1024];
                var bytesRead = await stream.ReadAsync(buffer);
                var content = Encoding.ASCII.GetString(buffer, 0, bytesRead);

                // Look for typical PyTorch state dict patterns
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

        ///// <summary>
        ///// Gets a user-friendly description of the model file type
        ///// </summary>
        //public static string GetFileTypeDescription(ModelFileType fileType)
        //{
        //    return fileType switch
        //    {
        //        ModelFileType.PyTorch => "PyTorch Model",
        //        ModelFileType.SafeTensors => "SafeTensors Model",
        //        ModelFileType.Checkpoint => "Model Checkpoint",
        //        ModelFileType.ONNX => "ONNX Model",
        //        ModelFileType.TensorFlow => "TensorFlow Model",
        //        ModelFileType.TensorFlowLite => "TensorFlow Lite Model",
        //        ModelFileType.TorchScript => "TorchScript Model",
        //        ModelFileType.TorchJit => "TorchJit Model",
        //        ModelFileType.Config => "Configuration File",
        //        ModelFileType.Weights => "Weight File",
        //        ModelFileType.StateDict => "PyTorch State Dictionary",
        //        _ => "Unknown File Type"
        //    };
        //}

        /// <summary>
        /// Checks if a file is a valid model file of any supported type
        /// </summary>
        public static bool IsValidModelFile(string filePath)
        {
            var fileType = FileUtil.DetectFileType(filePath);
            // TODO: optimize this
            return fileType is (ModelFileType.PyTorch or ModelFileType.SafeTensors or ModelFileType.Checkpoint or ModelFileType.ONNX or ModelFileType.Weights);
        }

        ///// <summary>
        ///// Gets the default extension for a given model file type
        ///// </summary>
        //public static string GetDefaultExtension(ModelFileType fileType)
        //{
        //    return fileType switch
        //    {
        //        ModelFileType.PyTorch => ".pt",
        //        ModelFileType.SafeTensors => ".safetensors",
        //        ModelFileType.Checkpoint => ".ckpt",
        //        ModelFileType.ONNX => ".onnx",
        //        ModelFileType.TensorFlow => ".pb",
        //        ModelFileType.TensorFlowLite => ".tflite",
        //        ModelFileType.TorchScript => ".torchscript",
        //        ModelFileType.TorchJit => ".jit",
        //        ModelFileType.Config => ".json",
        //        ModelFileType.Weights => ".bin",
        //        ModelFileType.StateDict => ".pth",
        //        _ => ""
        //    };
        //}
    }
}

