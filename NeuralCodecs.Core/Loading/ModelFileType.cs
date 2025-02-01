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
}