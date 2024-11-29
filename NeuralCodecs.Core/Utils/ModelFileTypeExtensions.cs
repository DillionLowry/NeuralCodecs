using NeuralCodecs.Core.Loading;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Utils
{
     static class ModelFileTypeExtensions
    {
        /// <summary>
        /// Gets the default extension for a given model file type
        /// </summary>
        public static string GetDefaultExtension(this ModelFileType fileType)
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

        public static IEnumerable<string> GetDefaultExtensions()
        {
            foreach (ModelFileType fileType in Enum.GetValues(typeof(ModelFileType)))
            {
                yield return fileType.GetDefaultExtension();
            }
            
        }
        public static string GetFileTypeDescription(this ModelFileType fileType)
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
    }

}

