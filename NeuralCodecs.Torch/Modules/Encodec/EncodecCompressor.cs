using NeuralCodecs.Core.Utils;
using NeuralCodecs.Torch.Config.Encodec;
using NeuralCodecs.Torch.Utils;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// Audio compression using Encodec neural codec
/// </summary>
public static class EncodecCompressor
{
    private static readonly Dictionary<string, Func<Task<Models.Encodec>>> ModelFactories = new()
    {
        { "encodec_24khz", () => NeuralCodecs.CreateEncodecAsync("facebook/encodec_24khz", EncodecConfig.Encodec24Khz) },
        { "encodec_48khz", () => NeuralCodecs.CreateEncodecAsync("facebook/encodec_48khz", EncodecConfig.Encodec48Khz) }
    };

    /// <summary>
    /// Compress audio to bytes
    /// </summary>
    /// <param name="model"></param>
    /// <param name="wav"></param>
    /// <param name="useLm"></param>
    /// <returns></returns>
    public static async Task<byte[]> CompressAsync(
        Models.Encodec model,
        Tensor wav,
        bool useLm = false)
    {
        await using var ms = new MemoryStream();
        await CompressToStreamAsync(model, wav, ms, useLm);
        return ms.ToArray();
    }

    /// <summary>
    /// Compress audio to a .ecdc file
    /// </summary>
    /// <param name="model"></param>
    /// <param name="wav"></param>
    /// <param name="filePath"></param>
    /// <param name="useLm"></param>
    /// <returns></returns>
    public static async Task CompressToFileAsync(
        Models.Encodec model,
        Tensor wav,
        string filePath,
        bool useLm = true)
    {
        await using var fs = new FileStream(filePath, FileMode.Create);
        await CompressToStreamAsync(model, wav, fs, useLm);
    }

    /// <summary>
    /// Compress audio to a stream
    /// </summary>
    /// <param name="model"></param>
    /// <param name="wav"></param>
    /// <param name="stream"></param>
    /// <param name="useLm"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static async Task CompressToStreamAsync(
        Models.Encodec model,
        Tensor wav,
        Stream stream,
        bool useLm = true)
    {
        // Validate input
        if (wav.dim() != 2)
        {
            throw new ArgumentException("Only single waveform can be encoded (shape should be [C, L])");
        }

        // Ensure channels match the model
        int channels = (int)wav.shape[0];
        if (channels != model.Channels)
        {
            throw new ArgumentException($"Expected {model.Channels} channels, got {channels}");
        }

        if (!ModelFactories.ContainsKey(model.GetName()))
        {
            throw new ArgumentException($"Model {model.GetName()} not supported");
        }

        // Get language model if needed
        var lm = useLm ? await model.GetLanguageModel() : null;

        //if (wav.device)
        using (no_grad())
        {
            // Encode the audio in frames
            var frames = model.Encode(wav.unsqueeze(0));

            // Write metadata header
            var metadata = new Dictionary<string, object>
            {
                { "m", model.GetName() },
                { "al", wav.shape[^1] },
                { "nc", frames[0].Codes.shape[1] },
                { "lm", useLm },
                { "ch", channels },
                { "sr", model.SampleRate },
            };

            if (model.CurrentBandwidth.HasValue)
            {
                metadata["bw"] = model.CurrentBandwidth.Value;
            }

            await BinaryIO.WriteHeaderAsync(stream, metadata);

            // Process frames
            foreach (var (frame, scale) in frames)
            {
                // Write scale factor for normalized audio
                if (scale is not null)
                {
                    // Handle multi-channel scale factors
                    if (scale.numel() > 1)
                    {
                        // Write number of scale values - ensure consistent endianness
                        BinaryUtils.WriteInt32BigEndian(stream, (int)scale.numel());

                        // Write each scale value
                        for (int i = 0; i < scale.numel(); i++)
                        {
                            BinaryUtils.WriteSingleBigEndian(stream, scale[i].cpu().item<float>());
                        }
                    }
                    else
                    {
                        // For single-scale, write 1 as the count followed by the value
                        BinaryUtils.WriteInt32BigEndian(stream, 1);
                        BinaryUtils.WriteSingleBigEndian(stream, scale.cpu().item<float>());
                    }
                }

                var (_, K, T) = frame.GetBCTDimensions();

                if (useLm && lm is not null)
                {
                    using (torch.no_grad())
                    {
                        lm.to(wav.device);
                    }
                    using var coder = new ArithmeticCoder(stream);
                    var states = new List<Tensor>();
                    var offset = 0;
                    var input = zeros(1, K, 1, dtype: int64, device: wav.device);

                    for (var t = 0; t < T; t++)
                    {
                        Tensor probas;
                        using (no_grad())
                        {
                            (probas, states, offset) = lm.forward(input, states, offset);
                            input = frame.narrow(2, t, 1).add(1);  // Add 1 to avoid index 0 (reserved)
                        }

                        for (var k = 0; k < K; k++)
                        {
                            var value = frame[0, k, t].item<long>();
                            var qCdf = ArithmeticCodingUtils.BuildStableQuantizedCdf(
                                probas[0, .., k, 0],
                                coder.TotalRangeBits);
                            coder.Push((int)value, qCdf);
                        }
                    }
                    coder.Flush();
                }
                else
                {
                    // Use simple bit packing without language model
                    using var packer = new BitPacker(model.BitsPerCodebook, stream);
                    for (var t = 0; t < T; t++)
                    {
                        for (var k = 0; k < K; k++)
                        {
                            packer.Push((int)frame[0, k, t].item<long>());
                        }
                    }
                    packer.Flush();
                }
            }
        }
    }

    /// <summary>
    /// Decompress audio from bytes
    /// </summary>
    /// <param name="compressed"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    public static async Task<(Tensor waveform, int sampleRate)> DecompressAsync(
        byte[] compressed,
        Device? device = null)
    {
        await using var ms = new MemoryStream(compressed);
        return await DecompressFromStreamAsync(ms, device);
    }

    /// <summary>
    /// Decompress audio from a .ecdc file
    /// </summary>
    /// <param name="filePath"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    public static async Task<(Tensor waveform, int sampleRate)> DecompressFromFileAsync(
        string filePath,
        Device? device = null)
    {
        if (string.IsNullOrEmpty(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"File not found: {filePath}");
        }
        await using var fs = new FileStream(filePath, FileMode.Open);
        return await DecompressFromStreamAsync(fs, device);
    }

    /// <summary>
    /// Decompress audio from a stream
    /// </summary>
    /// <param name="stream"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    /// <exception cref="InvalidDataException"></exception>
    /// <exception cref="EndOfStreamException"></exception>
    public static async Task<(Tensor waveform, int sampleRate)> DecompressFromStreamAsync(
        Stream stream,
        Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanRead)
        {
            throw new ArgumentException("Stream must be readable", nameof(stream));
        }
        device ??= CPU;

        var metadata = await BinaryIO.ReadHeaderAsync(stream);

        var modelName = metadata["m"].ToString();
        var audioLength = Convert.ToInt32(metadata["al"].ToString());
        var numCodebooks = Convert.ToInt32(metadata["nc"].ToString());
        var useLm = Convert.ToBoolean(metadata["lm"].ToString());

        // Get channel count from metadata or default to mono
        var channels = metadata.TryGetValue("ch", out var ch)
            ? Convert.ToInt32(ch.ToString())
            : 1;

        // Get bandwidth if available
        float? bandwidth = metadata.TryGetValue("bw", out var bw)
            ? Convert.ToSingle(bw.ToString())
            : null;

        // Get sample rate from metadata or default based on model
        if (!metadata.TryGetValue("sr", out var rate) || !int.TryParse(rate.ToString(), out int sampleRate))
        {
            sampleRate = modelName?.Contains("48khz") == true ? 48000 : 24000;
        }

        // Determine model factory based on model name or sample rate
        if (string.IsNullOrWhiteSpace(modelName) || !ModelFactories.ContainsKey(modelName))
        {
            modelName = sampleRate switch
            {
                48000 => "encodec_48khz",
                24000 => "encodec_24khz",
                _ => throw new ArgumentException($"Model {modelName} not supported")
            };
        }

        var model = await ModelFactories[modelName]();
        using (torch.no_grad())
        {
            model = model.to(device);
        }
        if (bandwidth.HasValue)
        {
            model.SetTargetBandwidth(bandwidth.Value);
        }

        if (model.Channels != channels)
        {
            throw new InvalidDataException(
                $"Model has {model.Channels} channels but compressed data has {channels} channels");
        }

        var lm = useLm ? await model.GetLanguageModel() : null;

        var frames = new List<EncodedFrame>();
        var segmentLength = model.SegmentLength ?? audioLength;
        var segmentStride = model.SegmentStride ?? audioLength;

        // Process frames
        for (var offset = 0; offset < audioLength; offset += segmentStride)
        {
            var thisSegmentLength = Math.Min(audioLength - offset, segmentLength);
            var frameLength = (int)Math.Ceiling(
                thisSegmentLength * model.FrameRate / (double)model.SampleRate);

            Tensor? scale = null;
            if (model.Normalize)
            {
                // Read scale factor count - using consistent endianness
                int numScales = BinaryUtils.ReadInt32BigEndian(stream);

                if (numScales is <= 0 or > 1000) // Sanity check
                {
                    throw new InvalidDataException($"Invalid scale count: {numScales}");
                }

                if (numScales > 1)
                {
                    // Read multiple scale values
                    var scalesList = new List<float>();
                    for (int i = 0; i < numScales; i++)
                    {
                        float scaleValue = BinaryUtils.ReadSingleBigEndian(stream);
                        scalesList.Add(scaleValue);
                    }

                    scale = tensor(scalesList.ToArray(), device: device);
                }
                else
                {
                    // Read single scale factor
                    float scaleValue = BinaryUtils.ReadSingleBigEndian(stream);
                    scale = tensor(scaleValue, device: device).view(1);
                }
            }

            var frame = zeros(1, numCodebooks, frameLength, dtype: int64, device: device);

            if (useLm)
            {
                using var decoder = new ArithmeticDecoder(stream);
                var states = new List<Tensor>();
                var lmOffset = 0;
                var input = zeros(1, numCodebooks, 1, dtype: int64, device: device);

                for (var t = 0; t < frameLength; t++)
                {
                    Tensor probas;
                    using (no_grad())
                    {
                        (probas, states, lmOffset) = lm!.forward(input, states, lmOffset);
                    }

                    var codeList = new List<long>();
                    for (var k = 0; k < numCodebooks; k++)
                    {
                        var qCdf = ArithmeticCodingUtils.BuildStableQuantizedCdf(
                            probas[0, .., k, 0],
                            decoder.TotalRangeBits);
                        var code = decoder.Pull(qCdf);

                        if (!code.HasValue)
                        {
                            throw new EndOfStreamException("Stream ended too soon");
                        }

                        codeList.Add(code.Value);
                    }

                    var codes = tensor(codeList, dtype: int64, device: device);
                    frame[0, .., t] = codes;
                    input = frame.narrow(2, t, 1).add(1);  // Add 1 to avoid index 0 (reserved)
                }
            }
            else
            {
                var unpacker = new BitUnpacker(model.BitsPerCodebook, stream);
                for (var t = 0; t < frameLength; t++)
                {
                    for (var k = 0; k < numCodebooks; k++)
                    {
                        var code = unpacker.Pull();
                        if (!code.HasValue)
                        {
                            throw new EndOfStreamException("Stream ended too soon");
                        }

                        frame[0, k, t] = code.Value;
                    }
                }
            }

            frames.Add(new EncodedFrame(frame, scale));
        }

        // Decode final waveform
        using (no_grad())
        {
            var wav = model.Decode(frames);

            // Ensure we only return exactly audioLength samples
            if (wav.shape[^1] > audioLength)
            {
                wav = wav.narrow(-1, 0, audioLength);
            }
            return (wav[0], model.SampleRate);
        }
    }

    /// <summary>
    /// Register a new model factory for a custom model
    /// </summary>
    /// <param name="modelName">Unique model identifier</param>
    /// <param name="modelFactory">Function to create model instance</param>
    public static void RegisterModel(string modelName, Func<Task<Models.Encodec>> modelFactory)
    {
        if (string.IsNullOrEmpty(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
        }

        ArgumentNullException.ThrowIfNull(modelFactory);

        ModelFactories[modelName] = modelFactory;
    }
}