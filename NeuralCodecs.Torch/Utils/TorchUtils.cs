using NeuralCodecs.Core.Configuration;
using TorchSharp;
using static TorchSharp.torch;
using DeviceType = NeuralCodecs.Core.Configuration.DeviceType;

namespace NeuralCodecs.Torch.Utils
{
    public class TorchUtils
    {
        public static torch.Device GetDevice(DeviceConfiguration device)
        {
            return device?.Type switch
            {
                DeviceType.CPU => torch.CPU,
                DeviceType.CUDA when cuda.is_available() => torch.CUDA,
                DeviceType.CUDA => throw new InvalidOperationException("CUDA requested but not available"),
                _ => torch.CPU
            };
        }
    }
}