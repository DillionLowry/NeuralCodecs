namespace NeuralCodecs.Core.Utils
{
    public static partial class TensorUtils
    {
        public static float[] Normalize(float[] input, float epsilon = 1e-5f)
        {
            float mean = 0;
            float variance = 0;

            // Calculate mean
            for (int i = 0; i < input.Length; i++)
                mean += input[i];
            mean /= input.Length;

            // Calculate variance
            for (int i = 0; i < input.Length; i++)
            {
                float diff = input[i] - mean;
                variance += diff * diff;
            }
            variance /= input.Length;

            // Normalize
            var output = new float[input.Length];
            float std = (float)Math.Sqrt(variance + epsilon);
            for (int i = 0; i < input.Length; i++)
                output[i] = (input[i] - mean) / std;

            return output;
        }

        public static float[] LayerNorm(float[] input, int channels, float[]? weight = null, float[]? bias = null)
        {
            if (input.Length % channels != 0)
                throw new ArgumentException("Input length must be divisible by number of channels");

            int timeSteps = input.Length / channels;
            var output = new float[input.Length];

            for (int t = 0; t < timeSteps; t++)
            {
                var slice = new Span<float>(input, t * channels, channels);
                var normalized = Normalize(slice.ToArray());

                for (int c = 0; c < channels; c++)
                {
                    float value = normalized[c];
                    if (weight != null)
                        value *= weight[c];
                    if (bias != null)
                        value += bias[c];
                    output[t * channels + c] = value;
                }
            }

            return output;
        }

        public static float[] Reshape(float[] input, params int[] shape)
        {
            int totalSize = 1;
            foreach (int dim in shape)
                totalSize *= dim;

            if (totalSize != input.Length)
                throw new ArgumentException("New shape must have same total size as input");

            return input.ToArray(); // Since we're working with flat arrays
        }
    }
}