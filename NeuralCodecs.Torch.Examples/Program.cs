using NAudio.Wave;
using NeuralCodecs.Torch.Config.SNAC;

namespace NeuralCodecs.Torch.Examples
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            string modelPath, inputAudioPath, outputAudioPath;
            if (args.Length < 3)
            {
                modelPath = "hubertsiuzdak/snac_24khz"; // Download from huggingface
                inputAudioPath = "en_sample.wav";
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
                var config = SNACConfig.SNAC24Khz;
                await SNACEncodeDecode(modelPath, inputAudioPath, outputAudioPath, config);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }

        public static async Task SNACEncodeDecode(string modelPath, string inputPath, string outputPath, SNACConfig config)
        {
            var model = await NeuralCodecs.CreateSNACAsync(modelPath, config);
            var buffer = LoadAudio(inputPath, model.Config.SamplingRate);

            Console.WriteLine("Encoding audio...");
            var codes = model.Encode(buffer);

            Console.WriteLine("Decoding codes...");
            var processedAudio = model.Decode(codes);

            Console.WriteLine("Saving output...");
            SaveAudio(outputPath, processedAudio, model.Config.SamplingRate);
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