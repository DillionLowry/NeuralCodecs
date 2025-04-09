namespace NeuralCodecs.Core.Loading.Repository
{
    /// <summary>
    /// Provides extension methods for working with model file types.
    /// </summary>
    public static class ModelFileTypeExtensions
    {
        /// <summary>
        /// Gets the default file extensions for supported model files.
        /// </summary>
        /// <returns>An array of file extensions.</returns>
        public static string[] GetDefaultExtensions()
        {
            return new[]
            {
                ".pt", ".pth", // PyTorch models
                ".bin",        // Binary weights
                ".ckpt",       // Checkpoints
                ".safetensors", // SafeTensors format
                ".onnx",       // ONNX models
                ".json",       // Configuration files
                ".txt",        // Text files (sometimes used for vocabularies)
                ".yml", ".yaml" // YAML configuration files
            };
        }

        /// <summary>
        /// Gets file extensions for a specific model file type.
        /// </summary>
        /// <param name="fileType">The model file type.</param>
        /// <returns>An array of file extensions for the specified type.</returns>
        public static string[] GetExtensionsForType(ModelFileType fileType)
        {
            return fileType switch
            {
                ModelFileType.PyTorch => new[] { ".pt", ".pth" },
                ModelFileType.SafeTensors => new[] { ".safetensors" },
                ModelFileType.Checkpoint => new[] { ".ckpt" },
                ModelFileType.ONNX => new[] { ".onnx" },
                ModelFileType.TensorFlow => new[] { ".pb" },
                ModelFileType.TensorFlowLite => new[] { ".tflite" },
                ModelFileType.TorchScript => new[] { ".torchscript" },
                ModelFileType.TorchJit => new[] { ".jit" },
                ModelFileType.Config => new[] { ".json", ".yml", ".yaml" },
                ModelFileType.Weights => new[] { ".bin", ".h5" },
                ModelFileType.StateDict => new[] { ".pth" },
                _ => Array.Empty<string>()
            };
        }
    }
}
