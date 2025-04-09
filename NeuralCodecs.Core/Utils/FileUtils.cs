using NeuralCodecs.Core.Loading;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Utils
{
    /// <summary>
    /// Provides utility methods for file operations.
    /// </summary>
    public static class FileUtils
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
        /// Computes the SHA-256 hash of a file.
        /// </summary>
        /// <param name="filePath">Path to the file.</param>
        /// <returns>Base64-encoded hash string.</returns>
        public static async Task<string> ComputeFileHashAsync(string filePath)
        {
            using var sha256 = SHA256.Create();
            using var stream = File.OpenRead(filePath);
            var hash = await sha256.ComputeHashAsync(stream);
            return Convert.ToBase64String(hash);
        }

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
        /// Detects the type of a model file based on its contents.
        /// </summary>
        /// <param name="filePath">Path to the model file.</param>
        /// <returns>The detected model file type.</returns>
        public static async Task<ModelFileType> DetectFileTypeFromContentsAsync(string filePath)
        {
            if (!File.Exists(filePath))
                return ModelFileType.Unknown;

            try
            {
                string extension = Path.GetExtension(filePath).ToLowerInvariant();
                
                // Quick check based on extension
                if (extension == ".json")
                    return ModelFileType.Config;
                    
                if (extension == ".safetensors")
                    return ModelFileType.SafeTensors;
                    
                if (extension == ".onnx")
                    return ModelFileType.ONNX;

                // Read first few bytes for signature detection
                byte[] signature = new byte[8];
                using (var stream = File.OpenRead(filePath))
                {
                    if (stream.Length < 8)
                        return ModelFileType.Unknown;
                        
                    await stream.ReadAsync(signature.AsMemory(0, 8));
                }

                // PyTorch files often start with these signatures
                if (signature[0] == 0x80 && signature[1] == 0x02 ||  // Pickle protocol
                    signature[0] == 0x50 && signature[1] == 0x4B)    // ZIP file (PyTorch checkpoint)
                {
                    return extension switch
                    {
                        ".pt" or ".pth" => ModelFileType.PyTorch,
                        ".ckpt" => ModelFileType.Checkpoint,
                        _ => ModelFileType.PyTorch  // Default to PyTorch for these signatures
                    };
                }

                // Other file types based on extension
                return extension switch
                {
                    ".pt" or ".pth" => ModelFileType.PyTorch,
                    ".bin" => ModelFileType.Weights,
                    ".pb" => ModelFileType.TensorFlow,
                    ".tflite" => ModelFileType.TensorFlowLite,
                    ".torchscript" => ModelFileType.TorchScript,
                    ".jit" => ModelFileType.TorchJit,
                    ".ckpt" => ModelFileType.Checkpoint,
                    _ => ModelFileType.Unknown
                };
            }
            catch
            {
                return ModelFileType.Unknown;
            }
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

        /// <summary>
        /// Checks if a file is a valid model file of any supported type
        /// </summary>
        public static bool IsValidModelFile(string filePath)
        {
            var fileType = DetectFileType(filePath);
            // TODO: optimize this
            return fileType is ModelFileType.PyTorch or ModelFileType.SafeTensors or ModelFileType.Checkpoint or ModelFileType.ONNX or ModelFileType.Weights;
        }
    }
}

