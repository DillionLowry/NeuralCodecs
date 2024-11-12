using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Models;
using NeuralCodecs.Core.Interfaces;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;

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

    public abstract T Encode<T,T2>(T2 audio);
    public abstract ValueTask<float[]> DecodeAsync(float[] codes, CancellationToken ct = default);

    public virtual ValueTask<float[][]> EncodeBatchAsync(float[][] audio, CancellationToken ct = default)
    {
        throw new NotImplementedException("Batch processing not supported by this codec");
    }

    public virtual ValueTask<float[][]> DecodeBatchAsync(float[][] codes, CancellationToken ct = default)
    {
        throw new NotImplementedException("Batch processing not supported by this codec");
    }
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
