using NAudio.Wave;

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
                outputAudioPath = "output_f.wav";
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
        }

        public static async Task EncodeDecodeAudio(string modelPath, string inputPath, string outputPath)
        {
            var model = await NeuralCodecs.CreateSNACAsync(modelPath, SNACConfig.SNAC24Khz);

            using var audioFile = new AudioFileReader(inputPath);
            var buffer = new List<float>();
            var readBuffer = new float[audioFile.WaveFormat.SampleRate * 4];
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

            //Process audio
            //Console.WriteLine("Processing audio...");
            //var processedAudio = model.ProcessAudio(
            //    [.. buffer],
            //    audioFile.WaveFormat.SampleRate
            //);


            Console.WriteLine("Encoding audio...");
            var codes = model.Encode(buffer.ToArray());

            Console.WriteLine("Decoding audio...");
            var processedAudio = model.Decode(codes);
            

            // Save output
            Console.WriteLine("Saving output...");
            await using var writer = new WaveFileWriter(
                outputPath,
                new WaveFormat(model.Config.SamplingRate, 1)
            );
            writer.WriteSamples(processedAudio, 0, processedAudio.Length);
        }
    }
}