using NAudio.Utils;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Torch.AudioTools;
using NeuralCodecs.Torch.Config.DAC;
using NeuralCodecs.Torch.Config.Dia;
using NeuralCodecs.Torch.Config.Encodec;
using NeuralCodecs.Torch.Config.SNAC;
using NeuralCodecs.Torch.Modules.Encodec;
using ScottPlot.TickGenerators.TimeUnits;
using Spectre.Console;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Channels;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchaudio;
using NeuralCodecs.Core.Utils;
using NeuralCodecs.Torch.Utils;
using NeuralCodecs.Torch.Models;
using SkiaSharp;

namespace NeuralCodecs.Torch.Examples
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            while (true)
            {
                string outputAudioPath = $"output_{DateTime.Now:ddmmss}.wav";

                AnsiConsole.Clear();
                AnsiConsole.Write(new FigletText("Neural Codecs").Centered().Color(Spectre.Console.Color.Blue));
                var codec = await AnsiConsole.PromptAsync(
                    new SelectionPrompt<string>()
                        .Title("Choose your [green]codec[/]:")
                        .PageSize(5)
                        .AddChoices("SNAC", "DAC", "Encodec", "Dia TTS", "Exit"));

                if (codec == "Exit")
                    break;

                var sampleRate = codec == "Dia TTS" ? "44.1khz" : await AnsiConsole.PromptAsync(
                    new SelectionPrompt<string>()
                        .Title($"Select sample rate for [green]{codec}[/]:")
                        .PageSize(5)
                        .AddChoices(codec switch
                        {
                            "SNAC" => new[] { "24khz", "32khz", "44.1khz" },
                            "DAC" => new[] { "16khz", "24khz", "44.1khz", "44.1khz-16kbps" },
                            "Encodec" => new[] { "24khz", "48khz" },
                            _ => Array.Empty<string>()
                        }));

                var useCustomModel = await AnsiConsole.ConfirmAsync("Would you like to use a custom model path?", defaultValue: false);

                string modelPath;
                if (useCustomModel)
                {
                    modelPath = await AnsiConsole.PromptAsync(
                        new TextPrompt<string>("Enter the path to your [green]model[/]:")
                            .ValidationErrorMessage("[red]Please enter a valid model path[/]")
                            .Validate(path =>
                            {
                                if (string.IsNullOrEmpty(path))
                                    return ValidationResult.Error("[red]Path cannot be empty[/]");
                                if (!Directory.Exists(path) && !File.Exists(path))
                                    return ValidationResult.Error("[red]Model path does not exist[/]");
                                return ValidationResult.Success();
                            }));
                }
                else
                {
                    modelPath = (codec, sampleRate) switch
                    {
                        ("SNAC", "24khz") => "hubertsiuzdak/snac_24khz",
                        ("SNAC", "32khz") => "hubertsiuzdak/snac_32khz",
                        ("SNAC", "44.1khz") => "hubertsiuzdak/snac_44khz",
                        ("DAC", "16khz") => "descript/dac_16khz",
                        ("DAC", "24khz") => "descript/dac_24khz",
                        ("DAC", "44.1khz") => "descript/dac_44khz",
                        ("DAC", "44.1khz-16kbps") => @"https://github.com/descriptinc/descript-audio-codec/",
                        ("Encodec", "24khz") => "facebook/encodec_24khz",
                        ("Encodec", "48khz") => "facebook/encodec_48khz",
                        ("Dia TTS", _) => "nari-labs/Dia-1.6B",
                        _ => throw new InvalidDataException("Selection was invalid")
                    };
                }
                var config = (codec, sampleRate) switch
                {
                    ("SNAC", "24khz") => (IModelConfig)SNACConfig.SNAC24kHz,
                    ("SNAC", "32khz") => (IModelConfig)SNACConfig.SNAC32kHz,
                    ("SNAC", "44.1khz") => (IModelConfig)SNACConfig.SNAC44kHz,
                    ("DAC", "16khz") => (IModelConfig)DACConfig.DAC16kHz,
                    ("DAC", "24khz") => (IModelConfig)DACConfig.DAC24kHz,
                    ("DAC", "44.1khz") => (IModelConfig)DACConfig.DAC44kHz,
                    ("DAC", "44.1khz-16kbps") => (IModelConfig)DACConfig.DAC44kHz_16kbps,
                    ("Encodec", "24khz") => (IModelConfig)EncodecConfig.Encodec24Khz,
                    ("Encodec", "48khz") => (IModelConfig)EncodecConfig.Encodec48Khz,
                    ("Dia TTS", _) => (IModelConfig)new DiaConfig(),
                    _ => throw new InvalidDataException("Selection was invalid")
                }; 
                
                string filePath;
                string textInput = "";

                if (codec == "Dia TTS")
                {
                    AnsiConsole.WriteLine("Voice annotations available (Can produce unpredictable results): (laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)\n");
                    AnsiConsole.WriteLine("Input expects two speakers denoted by \"[S1]\" and \"[S2]\" with at least 5 seconds of expected audio output\n");
                    textInput = (await AnsiConsole.PromptAsync(
                        new TextPrompt<string>("Enter the [green]text[/] you want to convert to speech:")
                            .DefaultValue("[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face.".EscapeMarkup())
                            .ValidationErrorMessage("[red]Text cannot be empty[/]")
                            .Validate(text =>
                            {
                                if (string.IsNullOrWhiteSpace(text))
                                    return ValidationResult.Error("[red]Text cannot be empty[/]");
                                return ValidationResult.Success();
                            })));
                    // Use a placeholder path for output
                    filePath = "not_used.wav";
                }
                else
                {
                    filePath = (await AnsiConsole.PromptAsync(
                        new TextPrompt<string>("Enter the path to your [green]WAV file[/]:")
                            .ValidationErrorMessage("[red]Please enter a valid file path[/]")
                            .Validate(path =>
                            {
                                if (string.IsNullOrEmpty(path))
                                {
                                    return ValidationResult.Error("[red]Path cannot be empty[/]");
                                }
                                if (!File.Exists(path))
                                    return ValidationResult.Error("[red]File does not exist[/]");
                                if (!path.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
                                    return ValidationResult.Error("[red]File must be a WAV file[/]");
                                return ValidationResult.Success();
                            }))).EscapeMarkup();
                }
                AnsiConsole.MarkupLine($"Encoding [blue]{(codec == "Dia TTS" ? "text to speech" : Path.GetFileName(filePath))}[/] with [green]{codec}[/] at [green]{sampleRate}[/]");

                try
                {
                    await AnsiConsole.Status()
                        .StartAsync("Loading model...", async ctx =>
                        {
                            if (codec == "SNAC")
                            {
                                await SNACEncodeDecode(modelPath, filePath, outputAudioPath, (SNACConfig)config, ctx);
                            }
                            else if (codec == "DAC")
                            {
                                await DACEncodeDecode(modelPath, filePath, outputAudioPath, (DACConfig)config, ctx: ctx);
                            }
                            else if (codec == "Dia TTS")
                            {
                                await DiaTTS(modelPath, textInput, outputAudioPath, (DiaConfig)config, ctx);
                            }
                            else // Encodec
                            {
                                await EncodecEncodeDecode(modelPath, filePath, outputAudioPath, (EncodecConfig)config, ctx);
                            }
                        });
                    AnsiConsole.MarkupLine("[green]Encoding completed successfully[/]");
                    AnsiConsole.WriteLine();
                    if (codec != "Dia TTS" && AnsiConsole.Confirm("Would you like to visualize the audio?"))
                    {
                        AnsiConsole.WriteLine();

                        AnsiConsole.WriteLine("Opening image...");
                        var t = new Table();
                        t.AddColumn("Spectrogram Comparison");
                        t.AddRow("Top:    Original Audio");
                        t.AddRow("Middle: Encoded Audio");
                        t.AddRow("Bottom: Difference");
                        AnsiConsole.Write(t);

                        AudioVisualizer.CompareAudioSpectrograms(filePath, outputAudioPath, config.SampleRate, "spectrogram_comparison.png");
                        AnsiConsole.WriteLine();

                        OpenImage("spectrogram_comparison.png");
                    }
                }
                catch (Exception ex)
                {
                    AnsiConsole.MarkupLine($"[red]Error during encoding: {Markup.Escape(ex.Message)} {Markup.Escape(ex.InnerException?.Message ?? string.Empty)}[/]");
                }
                AnsiConsole.WriteLine();

                if (!AnsiConsole.Confirm("Would you like to encode another file?"))
                    break;
            }

            AnsiConsole.MarkupLine("[blue]Exiting...[/]");
        }

        public static async Task CompressDecompress(Encodec model, string inputPath, string addToFilenames = "", EncodecConfig? config = null)
        {
            config ??= EncodecConfig.Encodec48Khz;
            var khz = $"{config.SampleRate / 1000}khz";

            Tensor tensor = config.Channels switch
            {
                1 => LoadAudioTensor(inputPath, config.SampleRate, config.Channels).squeeze(0),
                2 => torch.tensor(LoadAndDeinterleaveWavMultidimensional(inputPath, config.SampleRate)),
                _ => throw new NotImplementedException("Only mono and stereo are supported")
            };

            tensor = tensor.to(model.Device);

            await EncodecCompressor.CompressToFileAsync(model, tensor,
                $"output_{khz}_{model.CurrentBandwidth}_compressed{(string.IsNullOrWhiteSpace(addToFilenames) ? "" : $"_{addToFilenames}")}.ecdc",
                true);

            Thread.Sleep(500);

            var (waveform, sampleRate) = await EncodecCompressor.DecompressFromFileAsync(
                $"output_{khz}_{model.CurrentBandwidth}_compressed{(string.IsNullOrWhiteSpace(addToFilenames) ? "" : $"_{addToFilenames}")}.ecdc",
                model.Device);

            SaveEncodecAudio(waveform, $"output_{khz}_{config.Bandwidth}_decompressed{(string.IsNullOrWhiteSpace(addToFilenames) ? "" : $"_{addToFilenames}")}.wav", sampleRate, rescale: false);
        }

        public static async Task SNACEncodeDecode(string modelPath, string inputPath, string outputPath, SNACConfig config, StatusContext? ctx = null)
        {
            using var scope = NewDisposeScope();
            if (cuda.is_available())
            {
                config.Device = DeviceConfiguration.CUDA();
            }

            Report("Creating Model...", ctx);
            var model = await NeuralCodecs.CreateSNACAsync(modelPath, config);

            Report("Loading audio...", ctx);
            var buffer = LoadAudio(inputPath, model.Config.SampleRate);

            Report("Encoding audio...", ctx);
            var codes = model.Encode(buffer);

            Report("Decoding codes...", ctx);
            var processedAudio = model.Decode(codes);

            Report("Saving Audio...", ctx);
            SaveAudio(outputPath, processedAudio, model.Config.SampleRate, channels: 1);
        }

        public static async Task DACEncodeDecode(string modelPath, string inputPath, string outputPath, DACConfig config, bool useAudioSignal = false, StatusContext? ctx = null)
        {
            using var scope = NewDisposeScope();
            if (cuda.is_available())
            {
                config.Device = DeviceConfiguration.CUDA();
            }

            Report("Creating Model...", ctx);
            var model = await NeuralCodecs.CreateDACAsync(modelPath, config, null);

            Report("Loading audio...", ctx);
            Tensor audioTensor;
            if (useAudioSignal)
            {
                var audio = new AudioSignal(inputPath, device: config.Device.Type.ToString());

                audioTensor = audio.AudioData;
                Report("Encoding audio...", ctx);
                (var tAudio, var _, var _, var _, var _) = model.Encode(audioTensor);

                Report("Decoding audio...", ctx);
                var decoded = model.Decode(tAudio);
                Report("Saving Audio...", ctx);
                SaveAudio(outputPath, decoded.cpu().detach().data<float>().ToArray(), model.Config.SampleRate, channels: 1);
            }
            else
            {
                var buffer = LoadAudio(inputPath, model.Config.SampleRate);

                Report("Encoding audio...", ctx);
                var encoded = model.Encode(buffer);

                Report("Decoding audio...", ctx);
                var decoded = model.Decode(encoded);

                Report("Saving Audio...", ctx);
                SaveAudio(outputPath, decoded, model.Config.SampleRate, channels: 1);
            }
        }

        public static async Task EncodecEncodeDecode(string modelPath, string inputPath, string outputPath, EncodecConfig config, StatusContext? ctx = null)
        {
            try
            {
                using var scope = NewDisposeScope();
                if (cuda.is_available())
                {
                    config.Device = DeviceConfiguration.CUDA();
                }

                Report("Creating Model...", ctx);
                var model = await NeuralCodecs.CreateEncodecAsync(modelPath, config);

                Report("Loading audio...", ctx);
                var audio = LoadEncodecAudio(inputPath, config.SampleRate, config.Channels);
                Report("Encoding audio...", ctx);
                List<EncodedFrame> encoded = model.Encode(audio);

                Report("Decoding audio...", ctx);
                var decoded = model.Decode(encoded);

                Report("Saving Audio...", ctx);
                SaveEncodecAudio(decoded, outputPath, model.Config.SampleRate, rescale: false);
            }
            catch (Exception e)
            {
                Debug.WriteLine(e);
                throw;
            }
        }

        public static async Task DiaTTS(string modelPath, string textInput, string outputPath, DiaConfig config, StatusContext? ctx = null)
        {
            try
            {
                using var scope = NewDisposeScope();
                if (cuda.is_available())
                {
                    config.Device = DeviceConfiguration.CUDA();
                }
                // Remove escape chars added for Spectre console
                textInput = textInput.Replace("[[", "[").Replace("]]", "]");

                Report("Creating Dia TTS Model...", ctx);
                var model = await NeuralCodecs.CreateDiaAsync(modelPath, config);

                Report($"Converting text to speech: \"{textInput.EscapeMarkup()}\"", ctx);
                var audio = model.Generate(textInput);

                Report("Saving generated audio...", ctx);
                Dia.SaveAudio(outputPath, audio);

                Report($"Text-to-speech completed successfully. Output saved to {outputPath}", ctx);
            }
            catch (Exception ex)
            {
                Report($"Error during TTS generation: {ex.Message}", ctx);
                Report($"Stack Trace: {ex.StackTrace}", ctx);
                throw;
            }
        }
        #region Audio Loading and Saving

        public static float[] LoadAndDeinterleaveWav(string filePath, int targetSampleRate, int bitDepth = 16)
        {
            if (bitDepth is not 16 and not 32)
            {
                throw new ArgumentException("Bit depth must be either 16 or 32");
            }

            using var reader = new WaveFileReader(filePath);
                if (reader.WaveFormat.Channels != 2)
                {
                    throw new ArgumentException("Input file must be stereo (2 channels)");
                }

                if (reader.WaveFormat.BitsPerSample != bitDepth)
                {
                    throw new ArgumentException($"Input file has {reader.WaveFormat.BitsPerSample} bits per sample, but {bitDepth} was requested");
                }
                if (reader.WaveFormat.SampleRate != targetSampleRate)
                {
                    throw new NotImplementedException("Encodec stereo resampling is not implemented");
                }

                int bytesPerSample = bitDepth / 8;
                long totalSamplesPerChannel = reader.Length / (bytesPerSample * 2);
                float[] result = new float[totalSamplesPerChannel * 2];

                byte[] audioData = new byte[reader.Length];

                if (reader.Read(audioData, 0, audioData.Length) == 0)
                {
                    throw new ArgumentException("Input file is empty");
                }

                if (bitDepth == 16)
                {
                    // convert value range -32768 to 32767 => -1.0 to 1.0
                    for (long i = 0; i < totalSamplesPerChannel; i++)
                    {
                        int leftOffset = (int)(i * 2 * bytesPerSample);
                        short leftSample = BitConverter.ToInt16(audioData, leftOffset);
                        result[i] = leftSample / 32768f;

                        // Get right channel sample and convert to float
                    int rightOffset = (int)((i * 2 * bytesPerSample) + bytesPerSample);
                        short rightSample = BitConverter.ToInt16(audioData, rightOffset);
                        result[i + totalSamplesPerChannel] = rightSample / 32768f;
                    }
                }
                else
                {
                    // Convert value range -2147483648 to 2147483647 => -1.0 to 1.0
                    for (long i = 0; i < totalSamplesPerChannel; i++)
                    {
                        int leftOffset = (int)(i * 2 * bytesPerSample);
                        int leftSample = BitConverter.ToInt32(audioData, leftOffset);
                        result[i] = leftSample / 2147483648f;

                    int rightOffset = (int)((i * 2 * bytesPerSample) + bytesPerSample);
                        int rightSample = BitConverter.ToInt32(audioData, rightOffset);
                        result[i + totalSamplesPerChannel] = rightSample / 2147483648f;
                    }
                }

                return result;
            }

        public static float[,] LoadAndDeinterleaveWavMultidimensional(string filePath, int targetSampleRate, int bitDepth = 16)
        {
            if (bitDepth is not 16 and not 32)
            {
                throw new ArgumentException("Bit depth must be either 16 or 32");
            }

            using var reader = new WaveFileReader(filePath);
                if (reader.WaveFormat.Channels != 2)
                {
                    throw new ArgumentException("Input file must be stereo (2 channels)");
                }

                if (reader.WaveFormat.BitsPerSample != bitDepth)
                {
                    throw new ArgumentException($"Input file has {reader.WaveFormat.BitsPerSample} bits per sample, but {bitDepth} was requested");
                }
                if (reader.WaveFormat.SampleRate != targetSampleRate)
                {
                    throw new NotImplementedException("Encodec stereo resampling is not implemented");
                }

                int bytesPerSample = bitDepth / 8;
                long totalSamplesPerChannel = reader.Length / (bytesPerSample * 2);
                float[,] result = new float[2, totalSamplesPerChannel];

                byte[] audioData = new byte[reader.Length];

                if (reader.Read(audioData, 0, audioData.Length) == 0)
                {
                    throw new ArgumentException("Input file is empty");
                }

                if (bitDepth == 16)
                {
                    // convert value range -32768 to 32767 => -1.0 to 1.0
                    for (long i = 0; i < totalSamplesPerChannel; i++)
                    {
                        int leftOffset = (int)(i * 2 * bytesPerSample);
                        short leftSample = BitConverter.ToInt16(audioData, leftOffset);
                        result[0, i] = leftSample / 32768f;

                        // Get right channel sample and convert to float
                    int rightOffset = (int)((i * 2 * bytesPerSample) + bytesPerSample);
                        short rightSample = BitConverter.ToInt16(audioData, rightOffset);
                        result[1, i] = rightSample / 32768f;
                    }
                }
                else
                {
                    // Convert value range -2147483648 to 2147483647 => -1.0 to 1.0
                    for (long i = 0; i < totalSamplesPerChannel; i++)
                    {
                        int leftOffset = (int)(i * 2 * bytesPerSample);
                        int leftSample = BitConverter.ToInt32(audioData, leftOffset);
                        result[0, i] = leftSample / 2147483648f;

                    int rightOffset = (int)((i * 2 * bytesPerSample) + bytesPerSample);
                        int rightSample = BitConverter.ToInt32(audioData, rightOffset);
                        result[1, i] = rightSample / 2147483648f;
                    }
                }

                return result;
            }

        public static void InterleaveAndSaveAudio(Tensor audioTensor, string outputPath, int channels, int sampleRate)
        {
            int samples = (int)audioTensor.size(-1);

            var format = WaveFormat.CreateCustomFormat(
                WaveFormatEncoding.Pcm,
                sampleRate,
                channels,
                sampleRate * channels * 2, // Average bytes per second
                channels * 2,              // Block align
                16);                       // Bits per sample

            using var writer = new WaveFileWriter(outputPath, format);
            float[][] channelData = new float[channels][];
            for (int ch = 0; ch < channels; ch++)
            {
                channelData[ch] = audioTensor[ch].data<float>().ToArray();
            }

            byte[] sampleBuffer = new byte[2]; // 16-bit samples
            for (int i = 0; i < samples; i++)
            {
                for (int ch = 0; ch < channels; ch++)
                {
                    // Convert float [-1,1] to short [−32,768, 32,767]
                    short value = (short)(channelData[ch][i] * 32767.0f);

                    // Convert to bytes (respecting endianness)
                    sampleBuffer[0] = (byte)(value & 0xFF); // Low byte
                    sampleBuffer[1] = (byte)((value >> 8) & 0xFF); // High byte

                    writer.Write(sampleBuffer, 0, sampleBuffer.Length);
                }
            }
        }

        public static float[] LoadEncodecAudio(string filePath, int sampleRate, int channels, int bitDepth = 16)
        {
            return channels switch
            {
                1 => LoadAudio(filePath, sampleRate, mono: true),
                2 => LoadAndDeinterleaveWav(filePath, sampleRate, bitDepth),
                _ => throw new ArgumentException("Unsupported number of channels"),
            };
        }

        public static void SaveEncodecAudio(Tensor audioTensor, string outputPath, int sampleRate, bool rescale = false)
        {
            if (audioTensor.dim() == 3)
            {
                audioTensor.squeeze_(0);
            }
            var audio = audioTensor.cpu().detach();
            int channels = (int)audio.size(-2);
            float limit = 0.99f;

            if (rescale)
            {
                var max = audio.abs().max().item<float>();
                if (max > 0)
                {
                    audio = audio.mul(Math.Min(limit / max, 1.0f));
                }
            }
            else
            {
                audio = audio.clamp(-limit, limit);
            }

            if (channels == 1)
            {
                SaveAudio(outputPath, audio.cpu().detach().data<float>().ToArray(), sampleRate, channels);
            }
            else if (channels == 2)
            {
                InterleaveAndSaveAudio(audio, outputPath, channels, sampleRate);
            }
            else
            {
                throw new ArgumentException($"Unsupported number of channels: {channels}");
            }
        }

        public static float[] LoadAudio(string path, int targetSampleRate, bool mono = true)
        {
            using var audioFile = new AudioFileReader(path);
            var buffer = new List<float>();
            var readBuffer = new float[audioFile.WaveFormat.SampleRate * 4];
            int samplesRead;
            while ((samplesRead = audioFile.Read(readBuffer, 0, readBuffer.Length)) > 0)
            {
                buffer.AddRange(readBuffer.Take(samplesRead));
            }
            // Convert to mono, resample if necessary
            if (mono && audioFile.WaveFormat.Channels > 1)
            {
                buffer = AudioUtils.ConvertToMono(buffer, audioFile.WaveFormat.Channels);
            }
            if (audioFile.WaveFormat.SampleRate != targetSampleRate)
            {
                buffer = AudioUtils.ResampleLinear(buffer.ToArray(), audioFile.WaveFormat.SampleRate, targetSampleRate).ToList();
            }
            return buffer.ToArray();
        }

        public static void SaveAudio(string path, float[] buffer, int sampleRate, int channels)
        {
            using var writer = new WaveFileWriter(
                path,
                new WaveFormat(sampleRate, channels)
            );
            writer.WriteSamples(buffer, 0, buffer.Length);
        }

        public static Tensor LoadAudioTensor(string path, int? targetSampleRate = null, int targetChannels = 2)
        {
            using var reader = new AudioFileReader(path);

            // Read all samples
            var allSamples = new List<float>();
            var readBuffer = new float[reader.WaveFormat.SampleRate * reader.WaveFormat.Channels];
            int samplesRead;

            while ((samplesRead = reader.Read(readBuffer, 0, readBuffer.Length)) > 0)
            {
                allSamples.AddRange(readBuffer.Take(samplesRead));
            }

            var samples = allSamples.ToArray();
            var inputChannels = reader.WaveFormat.Channels;
            var length = samples.Length / inputChannels;

            // Convert to tensor with shape [batch=1, channels, time]
            var audio = torch.tensor(samples).reshape(1, inputChannels, length);

            // Convert number of channels if needed
            if (targetChannels != inputChannels)
            {
                if (targetChannels == 1 && inputChannels == 2)
                {
                    // Stereo to mono
                    audio = audio.mean([1], keepdim: true);
                }
                else if (targetChannels == 2 && inputChannels == 1)
                {
                    // Mono to stereo
                    audio = audio.repeat(1, 2, 1);
                }
                else
                {
                    throw new ArgumentException($"Unsupported channel conversion {inputChannels}->{targetChannels}");
                }
            }

            // Resample if needed and requested
            if (targetSampleRate.HasValue && targetSampleRate.Value != reader.WaveFormat.SampleRate)
            {
                var scale = targetSampleRate.Value / (float)reader.WaveFormat.SampleRate;
                var newLength = (int)(audio.size(-1) * scale);
                audio = torch.nn.functional.interpolate(
                    audio,
                    size: new long[] { newLength },
                    mode: InterpolationMode.Linear,
                    align_corners: true);
            }

            return audio;
        }

        public static Tensor LoadAudio(string path, int? targetSampleRate = null, int? targetChannels = null)
        {
            using var reader = new AudioFileReader(path);
            var sampleRate = reader.WaveFormat.SampleRate;
            var channels = reader.WaveFormat.Channels;

            // Read all samples
            var buffer = new List<float>();
            var readBuffer = new float[reader.WaveFormat.SampleRate * channels];
            int samplesRead;
            while ((samplesRead = reader.Read(readBuffer, 0, readBuffer.Length)) > 0)
            {
                buffer.AddRange(readBuffer.Take(samplesRead));
            }

            // Convert to tensor
            var audio = torch.tensor(buffer.ToArray())
                .reshape(1, channels, -1);

            // Resample if needed
            if (targetSampleRate.HasValue && targetSampleRate.Value != sampleRate)
            {
                // Simple linear resampling - could be improved with better algorithm
                var scale = targetSampleRate.Value / (float)sampleRate;
                var newLength = (int)(audio.size(-1) * scale);
                audio = torch.nn.functional.interpolate(
                    audio,
                    size: new long[] { newLength },
                    mode: InterpolationMode.Linear);
            }

            // Convert channels if needed
            if (targetChannels.HasValue && targetChannels.Value != channels)
            {
                if (targetChannels.Value == 1)
                {
                    // Convert to mono by averaging
                    audio = audio.mean([1], keepdim: true);
                }
                else if (targetChannels.Value == 2 && channels == 1)
                {
                    // Duplicate mono to stereo
                    audio = audio.repeat(1, 2, 1);
                }
                else
                {
                    throw new ArgumentException($"Unsupported channel conversion {channels}->{targetChannels.Value}");
                }
            }

            return audio;
        }

        #endregion Audio Loading and Saving

        private static void Report(string output, StatusContext? ctx = null)
        {
            if (ctx is null)
            {
                Console.WriteLine(output);
            }
            else
            {
                ctx.Status(output);
            }
        }

        // Helper method to validate the audio content
        public static void PrintAudioStats(Tensor audio, string label = "Audio")
        {
            var min = audio.min().item<float>();
            var max = audio.max().item<float>();
            var mean = audio.mean().item<float>();
            var std = audio.std().item<float>();

            Console.WriteLine($"{label} stats:");
            Console.WriteLine($"Shape: [{string.Join(", ", audio.shape)}]");
            Console.WriteLine($"Min: {min:F6}");
            Console.WriteLine($"Max: {max:F6}");
            Console.WriteLine($"Mean: {mean:F6}");
            Console.WriteLine($"Std: {std:F6}");

            // Print per-channel stats for stereo
            if (audio.size(1) == 2)
            {
                for (int c = 0; c < 2; c++)
                {
                    var channel = audio[0, c];
                    Console.WriteLine($"Channel {c}:");
                    Console.WriteLine($"  Min: {channel.min().item<float>():F6}");
                    Console.WriteLine($"  Max: {channel.max().item<float>():F6}");
                    Console.WriteLine($"  Mean: {channel.mean().item<float>():F6}");
                }
            }
        }

        public static void OpenImage(string imagePath)
        {
            try
            {
                var psi = new ProcessStartInfo();
                psi.UseShellExecute = true;

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    psi.FileName = imagePath;
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    psi.FileName = "xdg-open";
                    psi.Arguments = imagePath;
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    psi.FileName = "open";
                    psi.Arguments = imagePath;
                }
                else
                {
                    throw new PlatformNotSupportedException("Current OS platform is not supported.");
                }

                Process.Start(psi);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to open image: {ex.Message}");
            }
        }
    }
}