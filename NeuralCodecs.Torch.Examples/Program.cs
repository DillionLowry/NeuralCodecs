using NAudio.Wave;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Torch.AudioTools;
using NeuralCodecs.Torch.Config.DAC;
using NeuralCodecs.Torch.Config.SNAC;
using NeuralCodecs.Torch.Utils;
using SkiaSharp;
using Spectre.Console;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Examples
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            string outputAudioPath = "output.wav";
            while (true)
            {
                AnsiConsole.Clear();
                AnsiConsole.Write(new FigletText("Neural Codecs").Centered().Color(Spectre.Console.Color.Blue));

                var codec = AnsiConsole.Prompt(
                    new SelectionPrompt<string>()
                        .Title("Choose your [green]codec[/]:")
                        .PageSize(5)
                        .AddChoices("SNAC", "DAC", "Exit"));

                if (codec == "Exit")
                    break;

                var sampleRate = AnsiConsole.Prompt(
                    new SelectionPrompt<string>()
                        .Title($"Select sample rate for [green]{codec}[/]:")
                        .PageSize(5)
                        .AddChoices(codec == "SNAC" 
                            ? new[] { "24khz", "32khz", "44.1khz" }
                            : new[] { "16khz", "24khz", "44.1khz", "44.1khz-16kbps" }));

                var useCustomModel = AnsiConsole.Confirm("Would you like to use a custom model path?", defaultValue:false);

                string modelPath;
                if (useCustomModel)
                {
                    modelPath = AnsiConsole.Prompt(
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
                        _ => throw new InvalidDataException("Selection was invalid")
                    };
                }

                var config = (codec, sampleRate) switch
                {
                    ("SNAC", "24khz") => (IModelConfig)SNACConfig.SNAC24Khz,
                    ("SNAC", "32khz") => (IModelConfig)SNACConfig.SNAC32Khz,
                    ("SNAC", "44.1khz") => (IModelConfig)SNACConfig.SNAC44Khz,
                    ("DAC", "16khz") => (IModelConfig)DACConfig.DAC16kHz,
                    ("DAC", "24khz") => (IModelConfig)DACConfig.DAC24kHz,
                    ("DAC", "44.1khz") => (IModelConfig)DACConfig.DAC44kHz,
                    ("DAC", "44.1khz-16kbps") => (IModelConfig)DACConfig.DAC44kHz_16kbps,
                    _ => throw new InvalidDataException("Selection was invalid")
                };

                var filePath = AnsiConsole.Prompt(
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
                        }));

                AnsiConsole.MarkupLine($"Encoding [blue]{Path.GetFileName(filePath)}[/] with [green]{codec}[/] at [green]{sampleRate}[/]");

                try
                {
                    await AnsiConsole.Status()
                        .StartAsync("Loading model...", async ctx =>
                        {
                            if (codec == "SNAC")
                            {
                                await SNACEncodeDecode(modelPath, filePath, outputAudioPath, (SNACConfig)config, ctx);
                            }
                            else
                            {
                                await DACEncodeDecode(modelPath, filePath, outputAudioPath, (DACConfig)config, ctx:ctx);
                            }
                        });
                    AnsiConsole.MarkupLine("[green]Encoding completed successfully[/]");
                    AnsiConsole.WriteLine();
                    if (AnsiConsole.Confirm("Would you like to visualize the audio?"))
                    {
                        AnsiConsole.WriteLine();


                        AnsiConsole.WriteLine("Opening image...");
                        var t = new Table();
                        t.AddColumn("Spectrogram Comparison");
                        t.AddRow("Top:    Original Audio");
                        t.AddRow("Middle: Encoded Audio");
                        t.AddRow("Bottom: Difference");
                        AnsiConsole.Write(t);

                        AudioVisualizer.CompareAudioSpectrograms(filePath, outputAudioPath, config.SamplingRate, "spectrogram_comparison.png");
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
        public static async Task SNACEncodeDecode(string modelPath, string inputPath, string outputPath, SNACConfig config, StatusContext? ctx = null)
        {
            Report("Creating Model...", ctx);
            var model = await NeuralCodecs.CreateSNACAsync(modelPath, config);

            Report("Loading audio...", ctx);
            var buffer = LoadAudio(inputPath, model.Config.SamplingRate);

            Report("Encoding audio...", ctx);
            var codes = model.Encode(buffer);

            Report("Decoding codes...", ctx);
            var processedAudio = model.Decode(codes);

            Report("Saving Audio...", ctx);
            SaveAudio(outputPath, processedAudio, model.Config.SamplingRate);


        }
        public static async Task DACEncodeDecode(string modelPath, string inputPath, string outputPath, DACConfig config, bool useAudioSignal = true, StatusContext? ctx = null)
        {
            Report("Creating Model...", ctx);

            if (cuda.is_available())
            {
                config.Device = DeviceConfiguration.CUDA();
            }

            var model = await NeuralCodecs.CreateDACAsync(modelPath, config, null);

            Report("Loading audio...", ctx);
            Tensor audioTensor;
            if (useAudioSignal)
            {
                var audio = new AudioSignal(inputPath, device: config.Device.Type.ToString());
                audio.Resample(config.SamplingRate);
                audio.ToMono();

                audioTensor = audio.AudioData;

                Report("Encoding audio...", ctx);
                (var tAudio, var _, _, _, _) = model.Encode(audioTensor);

                Report("Decoding audio...", ctx);
                var decoded = model.Decode(tAudio);
                Report("Saving Audio...", ctx);
                SaveAudio(outputPath, decoded.cpu().detach().data<float>().ToArray(), model.Config.SamplingRate);
            }
            else
            {
                var buffer = LoadAudio(inputPath, model.Config.SamplingRate);

                Report("Encoding audio...", ctx);
                var encoded = model.Encode(buffer);

                Report("Decoding audio...", ctx);
                var decoded = model.Decode(encoded);

                Report("Saving Audio...", ctx);
                SaveAudio(outputPath, decoded, model.Config.SamplingRate);
            }
        }

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

        public static float[] LoadAudio(string path, int targetSampleRate)
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
            if (audioFile.WaveFormat.Channels > 1)
            {
                buffer = ConvertToMono(buffer, audioFile.WaveFormat.Channels);
            }
            if (audioFile.WaveFormat.SampleRate != targetSampleRate)
            {
                buffer = Resample(buffer.ToArray(), audioFile.WaveFormat.SampleRate, targetSampleRate).ToList();
            }
            return buffer.ToArray();
        }

        public static void SaveAudio(string path, float[] buffer, int sampleRate)
        {
            using var writer = new WaveFileWriter(
                path,
                new WaveFormat(sampleRate, 1)
            );
            writer.WriteSamples(buffer, 0, buffer.Length);
        }

        public static List<float> ConvertToMono(List<float> input, int channels)
        {
            var monoBuffer = new List<float>();
            for (int i = 0; i < input.Count; i += channels)
            {
                float sum = 0;
                for (int ch = 0; ch < channels; ch++)
                {
                    sum += input[i + ch];
                }
                monoBuffer.Add(sum / channels);
            }
            return monoBuffer;
        }

        private static float[] Resample(float[] input, int sourceSampleRate, int targetSampleRate)
        {
            var ratio = (double)targetSampleRate / sourceSampleRate;
            var outputLength = (int)(input.Length * ratio);
            var output = new float[outputLength];

            for (int i = 0; i < outputLength; i++)
            {
                var position = i / ratio;
                var index = (int)position;
                var fraction = position - index;

                if (index >= input.Length - 1)
                {
                    output[i] = input[input.Length - 1];
                }
                else
                {
                    output[i] = (float)((1 - fraction) * input[index] +
                               (fraction * input[index + 1]));
                }
            }

            return output;
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