using NAudio.Wave;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Torch.Config.SNAC;
using Spectre.Console;

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
                        .AddChoices("SNAC", "Exit"));

                if (codec == "Exit")
                    break;

                var sampleRate = AnsiConsole.Prompt(
                    new SelectionPrompt<string>()
                        .Title($"Select sample rate for [green]{codec}[/]:")
                        .PageSize(3)
                        .AddChoices("24khz", "32khz", "44.1khz"));

                var filePath = AnsiConsole.Prompt(
                    new TextPrompt<string>("Enter the path to your [green]WAV file[/]:")
                        .ValidationErrorMessage("[red]Please enter a valid file path[/]")
                        .Validate(path =>
                        {
                            if (string.IsNullOrEmpty(path))
                                return ValidationResult.Error("[red]Path cannot be empty[/]");
                            if (!File.Exists(path))
                                return ValidationResult.Error("[red]File does not exist[/]");
                            if (!path.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
                                return ValidationResult.Error("[red]File must be a WAV file[/]");
                            return ValidationResult.Success();
                        }));

                (string modelPath, IModelConfig config) = (codec, sampleRate) switch
                {
                    ("SNAC", "24khz") => ("hubertsiuzdak/snac_24khz", SNACConfig.SNAC24Khz),
                    ("SNAC", "32khz") => ("hubertsiuzdak/snac_32khz", SNACConfig.SNAC32Khz),
                    ("SNAC", "44.1khz") => ("hubertsiuzdak/snac_44khz", SNACConfig.SNAC44Khz),

                    _ => throw new InvalidDataException("Selection was invalid")
                };

                AnsiConsole.MarkupLine($"Encoding [blue]{Path.GetFileName(filePath)}[/] with [green]{codec}[/] at [green]{sampleRate}[/]");

                try
                {
                    await AnsiConsole.Status()
                        .StartAsync("Encoding...", async ctx =>
                        {
                            await SNACEncodeDecode(modelPath, filePath, outputAudioPath, (SNACConfig)config, ctx);
                        });
                    AnsiConsole.MarkupLine("[green]Encoding completed successfully[/]");
                }
                catch (Exception ex)
                {
                    AnsiConsole.MarkupLine($"[red]Error during encoding: {ex.Message}[/]");
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
    }
}