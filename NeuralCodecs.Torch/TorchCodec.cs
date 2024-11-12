namespace NeuralCodecs.Torch;

internal abstract class TorchCodec : ICodec
{
    protected readonly torch.Device Device;
    protected readonly CodecConfig Config;
    protected bool disposedValue;

    protected TorchCodec(CodecConfig config)
    {
        Config = config ?? new CodecConfig();
        Device = ResolveDevice(config.Device);
    }

    public abstract string Name { get; }
    public abstract CodecType Type { get; }
    public abstract int SampleRate { get; }
    public bool IsDisposed => disposedValue;
    public IEncoder Encoder { get; }
    public IDecoder Decoder { get; }
    public IQuantizer Quantizer { get; }

    private static torch.Device ResolveDevice(Core.Models.DeviceType requestedDevice)
    {
        return requestedDevice switch
        {
            Core.Models.DeviceType.CPU => torch.CPU,
            Core.Models.DeviceType.CUDA when torch.cuda.is_available() => torch.CUDA,
            Core.Models.DeviceType.CUDA => throw new CodecException(
                "CUDA requested but not available",
                "Base",
                CodecOperation.Initialization),
            _ => torch.cuda.is_available() ? torch.CUDA : torch.CPU
        };
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                // Dispose managed state (managed objects)
            }
            disposedValue = true;
        }
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}