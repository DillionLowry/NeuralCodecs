using NAudio.Wave;

using System.Globalization;
using TorchSharp;
using NeuralCodecs.Torch;
using static TorchSharp.torch;


namespace NeuralCodecs.Torch.Examples
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            string modelPath, inputAudioPath, outputAudioPath;
            if (args.Length < 3)
            {
                modelPath = @"T:\Models\SNAC\snac_24khz\pytorch_model.bin";
                inputAudioPath = @"C:\Users\Main\source\repos\SNACSharp\SNACSharp.Example\en_sample.wav";
                outputAudioPath = "output.wav";
            }
            else
            {
                modelPath = args[0];
                inputAudioPath = args[1];
                outputAudioPath = args[2];
            }

            try
            {
                await EncodeDecodeAudio(modelPath, inputAudioPath, outputAudioPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }

            // -------------------
            // Load from Hugging Face
            //var model = await NeuralCodec.LoadModelAsync("hubertsiuzdak/snac_24khz", Core.Models.InferenceBackend.Torch, new ModelLoadOptions
            //{
            //    //Device = new NeuralCodecs.Core.Models.Device(){ Type = "cuda"),
            //    ValidateModel = true
            //});

            //// Create new model
            //var config = new SNACConfig
            //{
            //    SamplingRate = 44100,
            //    EncoderDim = 64,
            //    // ... other settings
            //};

            //var model = NeuralCodec.CreateSNACModel(config);
        }

        public static async Task EncodeDecodeAudio(string modelPath, string inputPath, string outputPath)
        {
            // Initialize model
            using var scope = torch.NewDisposeScope();

            var model = await NeuralCodecs.CreateSNACAsync(modelPath, SNACConfig.SNAC24Khz);

            // Load and preprocess audio
            var audioData = LoadAudioFile(inputPath, targetSampleRate: 24000);

            var device = cuda.is_available() ? CUDA : CPU;
            model.to(device);
            audioData = audioData.to(device);

            // Encode and decode
            using (torch.inference_mode())
            {
                Console.WriteLine("Encoding audio...");
                var codes = model.Encode(audioData);

                Console.WriteLine("Decoding audio...");
                var reconstructed = model.Decode(codes).cpu().detach();
                var audioArray = reconstructed.data<float>().ToArray();

                // Save the output audio
                Console.WriteLine("Saving output audio...");
                SaveAudioFile(outputPath, audioArray, 24000);
            }
        }

        private static Tensor LoadAudioFile(string path, int targetSampleRate)
        {
            using var audioFile = new AudioFileReader(path);
            var buffer = new List<float>();
            var readBuffer = new float[audioFile.WaveFormat.SampleRate * 4]; // 4 seconds chunks
            int samplesRead;
            while ((samplesRead = audioFile.Read(readBuffer, 0, readBuffer.Length)) > 0)
            {
                buffer.AddRange(readBuffer.Take(samplesRead));
            }

            // Convert to mono if stereo
            if (audioFile.WaveFormat.Channels > 1)
            {
                var monoBuffer = new List<float>();
                for (int i = 0; i < buffer.Count; i += audioFile.WaveFormat.Channels)
                {
                    float sum = 0;
                    for (int ch = 0; ch < audioFile.WaveFormat.Channels; ch++)
                    {
                        sum += buffer[i + ch];
                    }
                    monoBuffer.Add(sum / audioFile.WaveFormat.Channels);
                }
                buffer = monoBuffer;
            }

            // Resample if needed
            if (audioFile.WaveFormat.SampleRate != targetSampleRate)
            {
                buffer = Resample(buffer.ToArray(), audioFile.WaveFormat.SampleRate, targetSampleRate).ToList();
            }

            // Convert to tensor with shape [1, 1, samples]
            return torch.tensor(buffer, dtype: float32).reshape(1, 1, -1);
        }

        private static void SaveAudioFile(string path, float[] audioData, int sampleRate)
        {
            using var writer = new WaveFileWriter(path, new WaveFormat(sampleRate, 1));
            writer.WriteSamples(audioData, 0, audioData.Length);
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

        public static void AnalyzeCompression(Tensor audio, List<Tensor> codes)
        {
            // Calculate original size (16-bit samples)
            long originalSize = audio.size(-1) * 2; // bytes

            // Calculate compressed size
            long compressedSize = 0;
            foreach (var code in codes)
            {
                // Each code is stored as an integer
                compressedSize += code.numel() * 4; // 4 bytes per int32
            }

            Console.WriteLine($"Original audio size: {originalSize / 1024.0:F2} KB");
            Console.WriteLine($"Compressed size: {compressedSize / 1024.0:F2} KB");
            Console.WriteLine($"Compression ratio: {(float)originalSize / compressedSize:F2}:1");
        }

        public static List<Tensor> LoadTensorsFromFile(string filePath)
        {
            var tensors = new List<Tensor>();
            var numberFormat = new NumberFormatInfo { NumberDecimalDigits = 30 };

            foreach (var line in File.ReadLines(filePath))
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue; // Skip empty lines
                }

                var values = line.Split(',', options: StringSplitOptions.RemoveEmptyEntries)
                                 .Select(value => float.Parse(value.Trim(), numberFormat))
                                 .ToArray();

                var tensor = torch.tensor(values, dtype: float32);
                tensors.Add(tensor);
            }

            return tensors;
        }
    }
}