using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;

namespace NeuralCodecs.Torch
{
    public class SNACConfigValidator : IModelValidator<SNACConfig>
    {
        public ValidationResult ValidateConfig(SNACConfig config)
        {
            var errors = new List<string>();

            if (config.SamplingRate <= 0)
                errors.Add($"Invalid sampling rate: {config.SamplingRate}");

            if (config.EncoderDim <= 0)
                errors.Add($"Invalid encoder dimension: {config.EncoderDim}");

            if (config.DecoderDim <= 0)
                errors.Add($"Invalid decoder dimension: {config.DecoderDim}");

            if (config.EncoderRates.IsNullOrEmpty())
                errors.Add("Missing encoder rates");
            else if (config.EncoderRates.Any(r => r <= 0))
                errors.Add("Invalid encoder rate values");

            if (config.DecoderRates.IsNullOrEmpty())
                errors.Add("Missing decoder rates");
            else if (config.DecoderRates.Any(r => r <= 0))
                errors.Add("Invalid decoder rate values");

            if (config.AttnWindowSize <= 0)
                errors.Add($"Invalid attention window size: {config.AttnWindowSize}");

            if (config.CodebookSize <= 0)
                errors.Add($"Invalid codebook size: {config.CodebookSize}");

            if (config.CodebookDim <= 0)
                errors.Add($"Invalid codebook dimension: {config.CodebookDim}");

            if (config.VQStrides.IsNullOrEmpty())
                errors.Add("Missing VQ strides");
            else if (config.VQStrides.Any(s => s <= 0))
                errors.Add("Invalid VQ stride values");

            return errors.Count > 0
                ? ValidationResult.Failed(errors.ToArray())
                : ValidationResult.Success();
        }

        public async Task<ValidationResult> ValidateModel(INeuralCodec model, SNACConfig config)
        {
            if (model is not SNAC snacModel)
            {
                return ValidationResult.Failed($"Expected SNAC model but got {model.GetType().Name}");
            }

            var errors = new List<string>();

            try
            {
                using var scope = torch.NewDisposeScope();
                // Create 100ms of audio at the model's sample rate for quick validation
                var sampleLength = (int)(config.SamplingRate * 0.1);
                var input = torch.randn(1, 1, sampleLength);

                if (config.Device?.Type is Core.Models.DeviceType.CUDA)
                {
                    input = input.cuda();
                }

                await Task.Run(() =>
                {
                    using (torch.inference_mode())
                    {
                        // Validate encoding
                        var codes = snacModel.Encode(input);
                        var cCount = codes.Count();
                        if (codes == null || cCount == 0)
                        {
                            errors.Add("Model encoding failed - no codes produced");
                            return;
                        }

                        // Validate number of codebooks matches config
                        if (cCount != config.VQStrides.Length)
                        {
                            errors.Add($"Model produced {cCount} codebooks but config specifies {config.VQStrides.Length}");
                            return;
                        }

                        // Validate decoding
                        var output = snacModel.Decode(codes);

                        // Check output dimensions
                        if (output.dim() != 3)
                        {
                            errors.Add($"Model output has incorrect number of dimensions: {output.dim()} (expected 3)");
                            return;
                        }

                        if (output.size(0) != 1 || output.size(1) != 1)
                        {
                            errors.Add($"Model output has incorrect batch/channel dimensions: {output.shape}");
                            return;
                        }

                        // Validate output length is reasonable
                        var expectedLength = sampleLength;
                        var actualLength = output.size(-1);
                        var lengthDiff = Math.Abs(expectedLength - actualLength);

                        // Allow for some padding/alignment differences but not too large
                        if (lengthDiff > config.SamplingRate * 0.01) // More than 1% difference
                        {
                            errors.Add($"Model output length {actualLength} differs significantly from expected {expectedLength}");
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                errors.Add($"Model validation failed: {ex.Message}");
            }

            return errors.Count > 0
                ? ValidationResult.Failed(errors.ToArray())
                : ValidationResult.Success();
        }
    }
}
