using System.Diagnostics;
using System.Text.RegularExpressions;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Config.DAC;

public class StateDictNameConverter
{
    private static readonly Regex BlockPattern = new Regex(@"(block|model)\.(\d+)");
    private static readonly Regex ResUnitPattern = new Regex(@"(res_unit)(\d+)");
    private static readonly Regex SnakePattern = new Regex(@"(snake)(\d+)");
    private static readonly Regex ConvPattern = new Regex(@"(conv)(?:_t)?(\d+)");

    public static Dictionary<string, Tensor> ConvertStateDict(
        Dictionary<string, Tensor> stateDict,
        bool fromSafetensor = false)
    {
        return fromSafetensor ?
            ConvertFromSafetensor(stateDict) :
            ConvertFromPth(stateDict);
    }

    private static Dictionary<string, Tensor> ConvertFromPth(Dictionary<string, Tensor> stateDict)
    {
        var converted = new Dictionary<string, Tensor>();

        foreach (var (key, tensor) in stateDict)
        {
            // Skip weight normalization tensors
            if (key.EndsWith("_g") || key.EndsWith("_v"))
                continue;

            var newKey = ConvertPthPath(key);
            converted[newKey] = tensor;
        }

        return converted;
    }

    private static Dictionary<string, Tensor> ConvertFromSafetensor(Dictionary<string, Tensor> stateDict)
    {
        var converted = new Dictionary<string, Tensor>();

        var keyMap = BuildKeyMap();
        foreach (var kvp in stateDict)
        {
            var newKey = TranslateKey(keyMap, kvp.Key);
            if (newKey.Contains(".weight_v"))
            {
                // Handle weight normalization conversion
                var weight = kvp.Value;
                var norm = weight.contiguous().pow(2)
                               .sum([1, 2], keepdim: true, ScalarType.Float32)
                               .sqrt();

                converted[newKey] = weight;  // weight_v
                converted[newKey.Replace("weight_v", "weight_g")] = norm;  // weight_g
            }
            else
            {
                converted[newKey] = kvp.Value;
            }
        }
        return converted;
    }

    public static Dictionary<string, Tensor> MapStateDict(Dictionary<string, Tensor> safeTensorDict)
    {
        var torchDict = new Dictionary<string, Tensor>();

        // Find number of blocks by analyzing dict structure
        var blockIndices = safeTensorDict.Keys
            .Where(k => k.StartsWith("decoder.block."))
            .Select(k => int.Parse(k.Split('.')[2]))
            .Distinct()
            .OrderBy(i => i)
            .ToList();

        // Get block sizes by analyzing conv_t1 weights
        var blockSizes = new List<int>();
        blockSizes.Add((int)safeTensorDict["decoder.conv1.weight"].shape[0]); // Input size

        foreach (var blockIdx in blockIndices)
        {
            var convKey = $"decoder.block.{blockIdx}.conv_t1.weight";
            if (safeTensorDict.ContainsKey(convKey))
            {
                blockSizes.Add((int)safeTensorDict[convKey].shape[1]); // Output size of conv_t1
            }
        }

        // Map initial and final convs
        MapSpecialLayers(safeTensorDict, torchDict);

        // For each decoder block
        for (int blockIdx = 0; blockIdx < blockIndices.Count; blockIdx++)
        {
            var inChannels = blockSizes[blockIdx];
            var outChannels = blockSizes[blockIdx + 1];

            // Map upsampling conv (conv_t1)
            var convT1Prefix = $"decoder.block.{blockIdx}.conv_t1";
            var torchConvPrefix = $"decoder.model.{blockIdx + 1}.block.1";
            MapConvLayer(safeTensorDict, torchDict, convT1Prefix, torchConvPrefix);

            // Map Snake alpha for the block
            var snakeKey = $"decoder.block.{blockIdx}.snake1.alpha";
            var torchSnakeKey = $"decoder.model.{blockIdx + 1}.block.0.alpha";
            if (safeTensorDict.ContainsKey(snakeKey))
                torchDict[torchSnakeKey] = safeTensorDict[snakeKey];

            // Count ResUnits for this block
            var resUnitCount = safeTensorDict.Keys
                .Count(k => k.StartsWith($"decoder.block.{blockIdx}.res_unit") && k.Contains(".conv1."));

            // Map ResUnits
            for (int unitIdx = 1; unitIdx <= resUnitCount; unitIdx++)
            {
                var resUnitPrefix = $"decoder.block.{blockIdx}.res_unit{unitIdx}";
                var torchResPrefix = $"decoder.model.{blockIdx + 1}.block.{unitIdx + 1}.block";

                MapResUnit(safeTensorDict, torchDict, resUnitPrefix, torchResPrefix, outChannels);
            }
        }

        return torchDict;
    }

    private static void MapResUnit(Dictionary<string, Tensor> safeTensorDict, Dictionary<string, Tensor> torchDict,
                       string resUnitPrefix, string torchPrefix, int channels)
    {
        // Map snake1 alpha
        var snakeKey = $"{resUnitPrefix}.snake1.alpha";
        var torchSnakeKey = $"{torchPrefix}.0.alpha";
        if (safeTensorDict.ContainsKey(snakeKey))
            torchDict[torchSnakeKey] = safeTensorDict[snakeKey];

        // Map conv1
        var conv1Prefix = $"{resUnitPrefix}.conv1";
        var torchConv1Prefix = $"{torchPrefix}.1";
        MapConvLayer(safeTensorDict, torchDict, conv1Prefix, torchConv1Prefix);

        // Map snake2 alpha
        snakeKey = $"{resUnitPrefix}.snake2.alpha";
        torchSnakeKey = $"{torchPrefix}.2.alpha";
        if (safeTensorDict.ContainsKey(snakeKey))
            torchDict[torchSnakeKey] = safeTensorDict[snakeKey];

        // Map conv2
        var conv2Prefix = $"{resUnitPrefix}.conv2";
        var torchConv2Prefix = $"{torchPrefix}.3";
        MapConvLayer(safeTensorDict, torchDict, conv2Prefix, torchConv2Prefix);
    }

    private static void MapConvLayer(Dictionary<string, Tensor> safeTensorDict, Dictionary<string, Tensor> torchDict,
                             string convPrefix, string torchPrefix)
    {
        var weightKey = $"{convPrefix}.weight";
        if (safeTensorDict.ContainsKey(weightKey))
        {
            var weight = safeTensorDict[weightKey];
            var norm = weight.contiguous().pow(2)
                            .sum([1, 2], keepdim: true, ScalarType.Float32)
                            .sqrt();

            torchDict[$"{torchPrefix}.weight_g"] = norm;
            torchDict[$"{torchPrefix}.weight_v"] = weight;
        }

        var biasKey = $"{convPrefix}.bias";
        if (safeTensorDict.ContainsKey(biasKey))
            torchDict[$"{torchPrefix}.bias"] = safeTensorDict[biasKey];
    }

    private static void MapSpecialLayers(Dictionary<string, Tensor> safeTensorDict, Dictionary<string, Tensor> torchDict)
    {
        // Map initial conv
        MapConvLayer(safeTensorDict, torchDict, "decoder.conv1", "decoder.model.0");

        // Map final snake + conv
        var snakeKey = "decoder.snake1.alpha";
        if (safeTensorDict.ContainsKey(snakeKey))
            torchDict["decoder.model.5.alpha"] = safeTensorDict[snakeKey];

        MapConvLayer(safeTensorDict, torchDict, "decoder.conv2", "decoder.model.6");
    }

    public static Dictionary<string, Tensor> TranslateStateDict(Dictionary<string, Tensor> stateDict)
    {
        var translatedDict = new Dictionary<string, Tensor>();
        var mappings = new Dictionary<string, string>();

        // Handle main conv layers
        mappings["decoder.conv1"] = "decoder.model.0";
        mappings["decoder.conv2"] = "decoder.model.6";
        mappings["decoder.snake1"] = "decoder.model.5";

        // Handle decoder blocks
        for (int blockIdx = 0; blockIdx < 4; blockIdx++)
        {
            var safeTensorPrefix = $"decoder.block.{blockIdx}";
            var torchPrefix = $"decoder.model.{blockIdx + 1}.block";

            // Map snake1
            mappings[$"{safeTensorPrefix}.snake1"] = $"{torchPrefix}.0";

            // Map conv_t1
            mappings[$"{safeTensorPrefix}.conv_t1"] = $"{torchPrefix}.1";

            // Map res units
            for (int unitIdx = 1; unitIdx <= 3; unitIdx++)
            {
                var stPrefix = $"{safeTensorPrefix}.res_unit{unitIdx}";
                var tPrefix = $"{torchPrefix}.{unitIdx + 1}.block";

                mappings[$"{stPrefix}.snake1"] = $"{tPrefix}.0";
                mappings[$"{stPrefix}.conv1"] = $"{tPrefix}.1";
                mappings[$"{stPrefix}.snake2"] = $"{tPrefix}.2";
                mappings[$"{stPrefix}.conv2"] = $"{tPrefix}.3";
            }
        }

        // Apply mappings and handle weight normalization
        foreach (var kvp in stateDict)
        {
            var oldKey = kvp.Key;
            var value = kvp.Value;

            // Find matching prefix
            var matchingPrefix = mappings.Keys
                .FirstOrDefault(prefix => oldKey.StartsWith(prefix));

            if (matchingPrefix != null)
            {
                var newPrefix = mappings[matchingPrefix];
                var suffix = oldKey.Substring(matchingPrefix.Length);

                // Handle weight norm conversion
                if (suffix.Contains("weight"))
                {
                    // Calculate g and v components
                    var norm = value.contiguous().pow(2)
                               .sum([1, 2], keepdim: true, ScalarType.Float32)
                               .sqrt();

                    translatedDict[$"{newPrefix}.weight_g"] = norm;
                    translatedDict[$"{newPrefix}.weight_v"] = value;
                }
                else
                {
                    translatedDict[$"{newPrefix}{suffix}"] = value;
                }
            }
        }

        return translatedDict;
    }

    private static string ConvertPthPath(string pthPath)
    {
        var parts = pthPath.Split('.');

        // Special handling for decoder
        if (parts[0] == "decoder" && parts[1] == "model")
        {
            var newParts = new List<string> { "decoder", "block" };
            newParts.AddRange(parts.Skip(2)); // Skip "decoder.model"
            return string.Join(".", newParts);
        }

        return pthPath; // For encoder and quantizer paths that don't need conversion
    }

    public static Dictionary<string, string> BuildKeyMap(int resunitCount = 3, int decoderBlockCount = 4, int encoderBlockCount = 4)
    {
        var keyMap = new Dictionary<string, string>();

        // Helper to build map for each decoder block
        void MapDecoderBlock(int blockIndex)
        {
            // Map the initial snake layer
            keyMap[$"decoder.block.{blockIndex}.snake1"] = $"decoder.model.{blockIndex + 1}.block.0";

            // Map the transpose conv
            keyMap[$"decoder.block.{blockIndex}.conv_t1"] = $"decoder.model.{blockIndex + 1}.block.1";

            // Map each residual unit
            for (int unitIdx = 1; unitIdx <= resunitCount; unitIdx++)
            {
                var safetensorPrefix = $"decoder.block.{blockIndex}.res_unit{unitIdx}";
                var torchPrefix = $"decoder.model.{blockIndex + 1}.block.{unitIdx + 1}";

                keyMap[$"{safetensorPrefix}.snake1"] = $"{torchPrefix}.block.0";
                keyMap[$"{safetensorPrefix}.conv1"] = $"{torchPrefix}.block.1";
                keyMap[$"{safetensorPrefix}.snake2"] = $"{torchPrefix}.block.2";
                keyMap[$"{safetensorPrefix}.conv2"] = $"{torchPrefix}.block.3";
            }
        }

        void MapEncoderBlock(int blockIndex)
        {
            // Map the initial snake layer
            keyMap[$"encoder.block.{blockIndex}.snake1"] = $"encoder.block.{blockIndex + 1}.block.3";

            // Map the transpose conv
            keyMap[$"encoder.block.{blockIndex}.conv1"] = $"encoder.block.{blockIndex + 1}.block.4";

            // Map each residual unit
            for (int unitIdx = 1; unitIdx <= resunitCount; unitIdx++)
            {
                var safetensorPrefix = $"encoder.block.{blockIndex}.res_unit{unitIdx}";
                var torchPrefix = $"encoder.block.{blockIndex + 1}.block.{unitIdx + -1}";

                keyMap[$"{safetensorPrefix}.snake1"] = $"{torchPrefix}.block.0";
                keyMap[$"{safetensorPrefix}.conv1"] = $"{torchPrefix}.block.1";
                keyMap[$"{safetensorPrefix}.snake2"] = $"{torchPrefix}.block.2";
                keyMap[$"{safetensorPrefix}.conv2"] = $"{torchPrefix}.block.3";
            }
        }

        // Map the first and last convolutions and snake layer
        keyMap["decoder.conv1"] = "decoder.model.0";
        keyMap["decoder.snake1"] = "decoder.model.5";
        keyMap["decoder.conv2"] = "decoder.model.6";

        for (int i = 0; i < decoderBlockCount; i++)
        {
            MapDecoderBlock(i);
        }

        keyMap["encoder.conv1"] = "encoder.block.0";
        keyMap["encoder.snake1"] = "encoder.block.5";
        keyMap["encoder.conv2"] = "encoder.block.6";

        for (int i = 0; i < encoderBlockCount; i++)
        {
            MapEncoderBlock(i);
        }
        return keyMap;
    }

    public static string TranslateKey(Dictionary<string, string> keyMap, string safetensorKey)
    {
        // Find the base key (without .weight/.bias/.alpha suffix)
        var baseParts = safetensorKey.Split(new[] { ".weight", ".bias", ".alpha" }, StringSplitOptions.None);
        var baseKey = baseParts[0];

        if (keyMap.TryGetValue(baseKey, out var torchKey))
        {
            // Reattach the appropriate suffix
            if (safetensorKey.Contains(".weight"))
            {
                // Convert single weight to weight_v/weight_g for conv layers
                if (safetensorKey.Contains(".conv1") || safetensorKey.Contains(".conv2") || safetensorKey.Contains(".conv_t1"))
                {
                    return torchKey + ".weight_v";  // handle weight_g separately
                }
                return torchKey + ".weight";
            }
            else if (safetensorKey.Contains(".bias"))
            {
                return torchKey + ".bias";
            }
            else if (safetensorKey.Contains(".alpha"))
            {
                return torchKey + ".alpha";
            }
        }
        else if (safetensorKey.Contains(".weight") && (baseKey.Contains(".in_proj") || baseKey.Contains("out_proj")))
        {
            // Convert in_proj/out_proj weights to weight_v/weight_g
            return baseKey + ".weight_v";
        }

        return safetensorKey; // Return original if no mapping found
    }

    public static string StripEndNumbers(string input)
    {
        return new string(input.TakeWhile(c => !char.IsDigit(c)).ToArray());
    }

    public static bool VerifyWeights(Dictionary<string, Tensor> stateDict)
    {
        // Check essential components
        var hasEncoder = stateDict.Keys.Any(k => k.StartsWith("encoder"));
        var hasDecoder = stateDict.Keys.Any(k => k.StartsWith("decoder"));
        var hasQuantizer = stateDict.Keys.Any(k => k.StartsWith("quantizer"));

        if (!hasEncoder || !hasDecoder || !hasQuantizer)
        {
            throw new InvalidOperationException(
                "State dict missing required components (encoder, decoder, or quantizer)");
        }

        // Check minimum required layers per component
        var decoderBlockCount = stateDict.Keys
            .Where(k => k.StartsWith("decoder"))
            .Select(k => k.Split('.')[2])
            .Where(k => int.TryParse(k, out _))
            .Distinct()
            .Count();

        var encoderBlockCount = stateDict.Keys
            .Where(k => k.StartsWith("encoder"))
            .Select(k => k.Split('.')[2])
            .Where(k => int.TryParse(k, out _))
            .Distinct()
            .Count();

        var quantizerCount = stateDict.Keys
            .Where(k => k.StartsWith("quantizer"))
            .Select(k => k.Split('.')[2])
            .Where(k => int.TryParse(k, out _))
            .Distinct()
            .Count();

        if (decoderBlockCount == 0 || encoderBlockCount == 0 || quantizerCount == 0)
            throw new InvalidOperationException("One or more components has no blocks");

        return true;
    }
}