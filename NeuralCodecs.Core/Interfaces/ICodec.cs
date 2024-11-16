namespace NeuralCodecs.Core.Interfaces
{
    internal interface ICodec : IDisposable
    {
        IEncoder Encoder { get; }
        IDecoder Decoder { get; }
        IQuantizer Quantizer { get; }

        //public abstract TEncoded Encode<TEncoded, TAudio>(TAudio audio);
        //public abstract TDecoded Decode<TDecoded, TEncoded>(TEncoded encodedAudio);
    }

    //string Name { get; }
    //CodecType Type { get; }
    //int SampleRate { get; }
    //bool IsDisposed { get; }
}