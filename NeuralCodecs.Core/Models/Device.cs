namespace NeuralCodecs.Core.Models //TODO THIS NEEDS WORK
{
    public enum DeviceType
    {
        CPU,
        CUDA,
        DirectML
    }

    public record Device
    {
        public DeviceType Type { get; init; }
        public int Index { get; init; }

        public static Device CPU => new() { Type = DeviceType.CPU, Index = 0 };
        public static Device CUDA(int index = 0) => new() { Type = DeviceType.CUDA, Index = index };
    }
}