// Copyright (c) Dillion Lowry
//
// This file contains a C# port of Dia, originally developed by Nari Labs.
// 
// Original work: Copyright (c) Nari Labs
// Original source: https://github.com/nari-labs/dia
// Original license: Apache License 2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NeuralCodecs.Core;
using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Core.Utils;
using NeuralCodecs.Torch.Config.DAC;
using NeuralCodecs.Torch.Config.Dia;
using NeuralCodecs.Torch.Modules.Dia;
using NeuralCodecs.Torch.Utils;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;
using AudioUtils = NeuralCodecs.Torch.Modules.Dia.AudioUtils;

namespace NeuralCodecs.Torch.Models;

/// <summary>
/// Dia model for text-to-speech generation.
/// </summary>
public class Dia : INeuralCodec
{
    #region Constants

    /// <summary>
    /// Default sample rate for audio processing
    /// </summary>
    private const int DefaultSampleRate = 44100;

    /// <summary>
    /// Maximum valid codebook index value
    /// </summary>
    private const long MaxValidCodebookIndex = 1023L;

    /// <summary>
    /// Minimum valid codebook index value
    /// </summary>
    private const long MinValidCodebookIndex = 0L;

    #endregion Constants

    private readonly ScalarType _computeDtype;
    private readonly DiaConfig _config;
    private readonly Device _device;
    private DiaModel _model;
    private DAC? _dacModel;
    private bool _disposed;
    private bool _verbose;

    /// <summary>
    /// Gets or sets the audio processing method to use for speed and pitch adjustments
    /// </summary>
    public AudioSpeedCorrectionMethod AudioCorrectionMethod
    {
        get => _config.SpeedCorrectionMethod;
        set => _config.SpeedCorrectionMethod = value;
    }

    IModelConfig INeuralCodec.Config => _config;

    /// <summary>
    /// Gets the device used for computation.
    /// </summary>
    public Device Device => _device;

    /// <summary>
    /// Gets the underlying DiaModel instance.
    /// </summary>
    public DiaModel Model => _model;

    #region Constructor and Factory Methods

    /// <summary>
    /// Initializes a new instance of the Dia class.
    /// </summary>
    /// <param name="config">Configuration settings for the Dia model</param>
    public Dia(DiaConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _computeDtype = config.ComputeDtype.ToTorchDtype();
        _device = TorchUtils.GetDevice(config.Device);
        _verbose = config.Verbose;
        if (config.Seed is not null)
        {
            SetSeed(config.Seed.Value);
        }

        _model = new DiaModel(_config, _computeDtype, _device);
        _model.to(_device);
        _model.eval();

        if (config.LoadDACModel)
        {
            LoadDacModel(_config.DACModelPath);
        }
    }

    /// <summary>
    /// Loads the DAC model for audio decoding.
    /// </summary>
    public void LoadDacModel(string dacPath)
        => LoadDacModelAsync(dacPath).GetAwaiter().GetResult();

    /// <summary>
    /// Loads the DAC model for audio decoding.
    /// </summary>
    public async Task LoadDacModelAsync(string dacPath, DACConfig? config = null)
    {
        try
        {
            _dacModel = await NeuralCodecs.CreateDACAsync(dacPath, config ?? DACConfig.DAC44kHz);
            _dacModel.to(_device);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Failed to load DAC model", ex);
        }
    }

    /// <summary>
    /// Loads weights from the specified path.
    /// </summary>
    /// <param name="path">Path to the weights file</param>
    public void LoadWeights(string path) => _model.LoadWeights(path);

#pragma warning disable IDE1006 // Naming Styles

    /// <summary>
    /// Sets the model to evaluation mode.
    /// </summary>
    public void eval() => _model.eval();

    /// <summary>
    /// Sets the model to training mode.
    /// </summary>

    public void train() => _model.train();

#pragma warning restore IDE1006 // Naming Styles

    #endregion Constructor and Factory Methods

    #region Text Processing Methods

    private static int CountOccurrences(ReadOnlySpan<byte> source, int sourceLength, ReadOnlySpan<byte> pattern)
    {
        int count = 0;
        int pos = 0;

        while (pos <= sourceLength - pattern.Length)
        {
            if (source.Slice(pos, pattern.Length).SequenceEqual(pattern))
            {
                count++;
                pos += pattern.Length;
            }
            else
            {
                pos++;
            }
        }

        return count;
    }

    private static int ReplaceInPlace(ReadOnlySpan<byte> source, int sourceLength,
    ReadOnlySpan<byte> pattern, ReadOnlySpan<byte> replacement, out byte[] output)
    {
        output = null!;

        if (pattern.Length == replacement.Length)
        {
            int pos = 0;
            while (pos <= sourceLength - pattern.Length)
            {
                if (source.Slice(pos, pattern.Length).SequenceEqual(pattern))
                {
                    if (output == null)
                    {
                        output = new byte[sourceLength];
                        source.Slice(0, pos).CopyTo(output);
                    }
                    replacement.CopyTo(output.AsSpan(pos));
                    pos += pattern.Length;
                }
                else
                {
                    if (output != null)
                    {
                        output[pos] = source[pos];
                    }
                    pos++;
                }
            }

            if (output != null && pos < sourceLength)
            {
                source.Slice(pos).CopyTo(output.AsSpan(pos));
            }

            return sourceLength;
        }

        int count = CountOccurrences(source, sourceLength, pattern);

        if (count == 0)
        {
            return sourceLength;
        }

        int newSize = sourceLength - (count * pattern.Length) + (count * replacement.Length);
        output = new byte[newSize];

        int writePos = 0;
        int readPos = 0;

        while (readPos < sourceLength)
        {
            if (readPos <= sourceLength - pattern.Length &&
                source.Slice(readPos, pattern.Length).SequenceEqual(pattern))
            {
                replacement.CopyTo(output.AsSpan(writePos));
                writePos += replacement.Length;
                readPos += pattern.Length;
            }
            else
            {
                output[writePos++] = source[readPos++];
            }
        }

        return newSize;
    }

    private Tensor EncodeText(string text)
    {
        return torch.WrappedTensorDisposeScope(() =>
        {
            int maxLen = _config.Data.TextLength;
            int byteCount = Encoding.UTF8.GetByteCount(text);

            Span<byte> buffer = stackalloc byte[256];
            byte[]? array = byteCount > 256 ? new byte[byteCount] : null;
            Span<byte> sourceSpan = array ?? buffer.Slice(0, byteCount);

            Encoding.UTF8.GetBytes(text, sourceSpan);

            ReadOnlySpan<byte> s1Pattern = "[S1]"u8;
            ReadOnlySpan<byte> s2Pattern = "[S2]"u8;
            ReadOnlySpan<byte> s1Replacement = stackalloc byte[] { 0x01 };
            ReadOnlySpan<byte> s2Replacement = stackalloc byte[] { 0x02 };

            int resultLength = ReplaceInPlace(sourceSpan, byteCount, s1Pattern, s1Replacement, out byte[] replaced);
            resultLength = ReplaceInPlace(replaced ?? sourceSpan, resultLength, s2Pattern, s2Replacement, out byte[] final);

            byte[] resultBytes = final ?? replaced ?? array ?? sourceSpan.ToArray();

            int tokenCount = Math.Min(maxLen, resultLength);
            long[] tokens = new long[tokenCount];

            for (int i = 0; i < tokenCount; i++)
            {
                tokens[i] = resultBytes[i];
            }

            return tensor(tokens, dtype: ScalarType.Int64, device: Device);
        });
    }

    private Tensor PadTextInput(Tensor[] textTokens)
    {
        return torch.WrappedTensorDisposeScope(() =>
        {
            int textPadValue = _config.Data.TextPadValue;
            int maxLen = _config.Data.TextLength;
            int batchSize = textTokens.Length;

            Tensor srcTokens = torch.full(
                new long[] { batchSize, 1, maxLen },
                textPadValue,
                dtype: ScalarType.Int64,
                device: Device
            );

            for (int i = 0; i < batchSize; i++)
            {
                int currentLen = (int)textTokens[i].shape[0];
                if (currentLen > 0)
                {
                    int copyLen = Math.Min(currentLen, maxLen);
                    srcTokens[i, 0, ..copyLen] = textTokens[i][..copyLen];
                }
            }

            return srcTokens;
        });
    }

    #endregion Text Processing Methods

    #region Audio Prompt Processing

    /// <summary>
    /// Prepares the audio prompts for the model.
    /// </summary>
    /// <param name="batchSize">Requested number of prompts, used to set prefill steps. Supports batch size greater than the number of audio prompts.</param>
    /// <param name="audioPrompts">Optional list of audio prompts as tensors. Can be null or empty.</param>
    /// <returns>Tuple of (delayed audio batch tensor, array of prefill steps)</returns>
    private (Tensor, int[]) PrepareAudioPrompt(int batchSize, List<Tensor>? audioPrompts = null)
    {
        using var scope = torch.NewDisposeScope();

        int numChannels = _config.Data.Channels;
        int audioBosValue = _config.Data.AudioBosValue;

        int[] delayPattern = _config.Data.DelayPattern;
        int maxDelayPattern = delayPattern.Max();
        int batch;
        long maxlength;

        if (audioPrompts is null || audioPrompts.Count == 0)
        {
            batch = batchSize;
            maxlength = maxDelayPattern;
        }
        else
        {
            batch = Math.Max(batchSize, audioPrompts.Count);
            maxlength = audioPrompts.Select(p => p?.shape[0] ?? 0).Max() + maxDelayPattern;
        }

        var prefillSteps = new List<int>();

        Tensor prefill = torch.full(
                size: new long[] { batch, maxlength, numChannels },
                value: -1,
                dtype: torch.int32,
                device: Device);

        prefill[TensorIndex.Colon, 0, TensorIndex.Colon] = audioBosValue;

        if (audioPrompts is null || audioPrompts.Count == 0)
        {
            for (int i = 0; i < batch; i++)
            {
                prefillSteps.Add(1);
            }
        }
        else
        {
            for (int i = 0; i < batch; i++)
            {
                var prompt = audioPrompts.ElementAtOrDefault(i);
                if (prompt is not null)
                {
                    prompt = prompt.to(torch.int32, device: Device);
                    prefill[i, 1..(int)(prompt.shape[0] + 1), TensorIndex.Colon] = prompt;
                    prefillSteps.Add((int)prompt.shape[0] + 1);
                }
                else
                {
                    prefillSteps.Add(1);
                }
            }
        }

        (Tensor t_idx_BxTxC, Tensor indices_BTCx3) delayPrecomp = AudioUtils.BuildDelayIndices(
                B: batch,
                T: (int)maxlength,
                C: numChannels,
                delayPattern: delayPattern);

        var delayedBatch = AudioUtils.ApplyAudioDelay(
            audio_BxTxC: prefill,
            padValue: -1,
            bosValue: audioBosValue,
            precomp: delayPrecomp);

        return (delayedBatch.MoveToOuterDisposeScope(), prefillSteps.ToArray());
    }

    #endregion Audio Prompt Processing

    /// <summary>
    /// Sets the seed value for random number generation to ensure voice consistency across runs.
    /// </summary>
    /// <param name="seed">The seed value to use for random number generation. Must be a non-negative integer.</param>
    public static void SetSeed(int seed)
    {
        torch.manual_seed(seed);
        torch.cuda.manual_seed_all(seed);
        torch.random.manual_seed(seed);
    }

    /// <summary>
    /// Samples the next token from logits.
    /// </summary>
    /// <param name="logits">Token logits</param>
    /// <param name="temperature">Sampling temperature - controls randomness (higher = more random)</param>
    /// <param name="topP">Top-p (nucleus) sampling probability threshold</param>
    /// <param name="topK">Top-k filter parameter - limits sampling to top k most likely tokens</param>
    /// <param name="audioEosValue">End-of-sequence token value for special handling</param>
    /// <returns>Sampled token indices</returns>
    public static Tensor SampleNextToken(
        Tensor logits,
        float temperature,
        float topP,
        int? topK = null,
        int? audioEosValue = null)
    {
        using var scope = torch.NewDisposeScope();

        // Greedy sampling for temperature=0
        if (temperature < 1e-5f)
        {
            return argmax(logits, dim: -1).MoveToOuterDisposeScope();
        }

        if (audioEosValue is >= 0)
        {
            // Mask out EOS token for non-first channels
            var topLogitIndices_BCxV = argmax(logits, dim: -1);
            var eosNotHighestMask_BCxV = topLogitIndices_BCxV.ne(audioEosValue);
            var maskEosUnlessHighest_BCxV = zeros_like(logits, dtype: ScalarType.Bool);

            maskEosUnlessHighest_BCxV[eosNotHighestMask_BCxV, audioEosValue] = true;
            logits = logits.masked_fill(maskEosUnlessHighest_BCxV, float.NegativeInfinity);
        }

        // logits: Shape [C, V] -> scaled by temperature
        logits = logits.div(temperature);

        // Apply top-k filtering if specified
        if (topK.HasValue)
        {
            (_, Tensor indices) = topk(logits, k: topK.Value, dim: -1);

            var mask = ones_like(logits, dtype: ScalarType.Bool);
            var falseValues = zeros_like(indices, dtype: ScalarType.Bool);

            // Scatter false values at top-k indices
            mask.scatter_(-1, indices, falseValues);

            // Mask out non-top-k values with -inf
            logits.masked_fill_(mask, float.NegativeInfinity);
        }

        if (topP < 1.0f)
        {
            // probs: Shape [C, V] - probability distribution
            var probs = softmax(logits, dim: -1);

            // Sort probabilities in descending order
            // sortedProbs: Shape [C, V] - sorted probabilities
            // sortedIndices: Shape [C, V] - original indices
            var (sortedProbs, sortedIndices) = sort(probs, dim: -1, descending: true);

            // Calculate cumulative probabilities
            // cumulativeProbs: Shape [C, V] - cumulative sum
            var cumulativeProbs = cumsum(sortedProbs, dim: -1);

            // Create mask for tokens exceeding top-p threshold
            var sortedIndicesToRemove = cumulativeProbs.gt(topP);
            sortedIndicesToRemove = torch.roll(sortedIndicesToRemove, shifts: 1, dims: -1);
            sortedIndicesToRemove[TensorIndex.Ellipsis, 0] = torch.zeros_like(sortedIndicesToRemove[TensorIndex.Ellipsis, 0]);

            // Scatter mask back to original indices
            // indicesToRemove: Shape [C, V] - remapped mask
            var indicesToRemove = zeros_like(sortedIndicesToRemove);
            indicesToRemove.scatter_(dim: -1, sortedIndices, sortedIndicesToRemove);

            // Mask out tokens beyond top-p cumulative probability
            logits.masked_fill_(indicesToRemove, float.NegativeInfinity).MoveToOuterDisposeScope();
        }

        // Calculate final probabilities and sample
        var finalProbs = softmax(logits, dim: -1);
        var sampledIndices = multinomial(finalProbs, num_samples: 1);
        var sampledsqueezed = sampledIndices.squeeze_(1);
        return sampledsqueezed.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Performs a single step of decoder inference.
    /// </summary>
    /// <param name="tokensBx1xC">Current token tensor with shape [Batch*2, 1, Channels]</param>
    /// <param name="decState">Decoder inference state</param>
    /// <param name="cfgScale">Classifier-free guidance scale factor</param>
    /// <param name="temperature">Sampling temperature</param>
    /// <param name="topP">Top-p (nucleus) sampling threshold</param>
    /// <param name="topK">Top-k sampling threshold</param>
    /// <param name="currentIdx">Current decoding step index</param>
    /// <returns>Next token predictions with shape [Batch, Channels]</returns>
    public Tensor DecoderStep(
        Tensor tokensBx1xC,
        DecoderInferenceState decState,
        float cfgScale,
        float temperature,
        float topP,
        int topK,
        int currentIdx)
    {
        using var scope = NewDisposeScope();

        var B = tokensBx1xC.shape[0] / 2;
        var audioEosValue = _config.Data.AudioEosValue;

        // Get logits from decoder
        var logitsBx1xCxV = _model.Decoder.DecodeStep(tokensBx1xC, decState, currentIdx);
        var logitsLast2BxCxV = logitsBx1xCxV.index(TensorIndex.Colon, -1);
        var logitsLastBx2xCxV = logitsLast2BxCxV.view(B, 2, -1, logitsLast2BxCxV.shape[^1]);

        // Split conditional and unconditional paths
        var uncondLogitsBxCxV = logitsLastBx2xCxV.index(TensorIndex.Colon, 0, TensorIndex.Colon, TensorIndex.Colon);
        var condLogitsBxCxV = logitsLastBx2xCxV.index(TensorIndex.Colon, 1, TensorIndex.Colon, TensorIndex.Colon);

        // Apply classifier-free guidance
        var logitsBxCxV = condLogitsBxCxV + (cfgScale * (condLogitsBxCxV - uncondLogitsBxCxV));

        // Fixed indexing bug: mask invalid tokens properly
        logitsBxCxV.index(TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(audioEosValue + 1))
            .fill_(float.NegativeInfinity);

        logitsBxCxV.index(TensorIndex.Colon, TensorIndex.Slice(1), TensorIndex.Slice(audioEosValue))
            .fill_(float.NegativeInfinity);

        // Reduce EOS probability for first channel
        logitsBxCxV.index(TensorIndex.Colon, 0, audioEosValue).mul_(0.8f);

        // Sample next token
        var flatLogitsBCxV = logitsBxCxV.view(B * _config.Data.Channels, -1);
        var predBC = SampleNextToken(
            flatLogitsBCxV.to(ScalarType.Float32),
            temperature,
            topP,
            topK,
            audioEosValue);

        var predBxC = predBC.view(B, _config.Data.Channels);
        return predBxC.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Disposes resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Generates audio from a single text dialog prompt.
    /// </summary>
    /// <param name="text">Input text prompt</param>
    /// <param name="maxTokens">Maximum number of audio tokens to generate</param>
    /// <param name="cfgScale">Scale factor for classifier-free guidance</param>
    /// <param name="temperature">Sampling temperature</param>
    /// <param name="topP">Cumulative probability threshold for nucleus sampling</param>
    /// <param name="cfgFilterTopK">Number of top logits to consider for CFG filtering</param>
    /// <param name="audioPrompt">Optional audio prompt tensor for conditioning</param>
    /// <param name="audioPromptPath">Optional path to audio file for prompt</param>
    /// <param name="verbose">Whether to print progress information</param>
    /// <returns>The audio waveform as a float array</returns>
    public float[] Generate(
        string text,
        int? maxTokens = null,
        float cfgScale = 3.0f,
        float temperature = 1.2f,
        float topP = 0.95f,
        int cfgFilterTopK = 45,
        Tensor? audioPrompt = null,
        string? audioPromptPath = null,
        bool? verbose = null)
    {
        return Generate([text], maxTokens, cfgScale, temperature, topP, cfgFilterTopK,
                            audioPrompt is null ? null : [audioPrompt],
                            audioPromptPath is null ? null : [audioPromptPath],
                            verbose)[0];
    }

    /// <summary>
    /// Generates audio corresponding to the input list of text dialogs.
    /// </summary>
    /// <param name="text">List of input text prompts for batch generation</param>
    /// <param name="maxTokens">Maximum number of audio tokens to generate per prompt</param>
    /// <param name="cfgScale">Scale factor for classifier-free guidance</param>
    /// <param name="temperature">Temperature for sampling</param>
    /// <param name="topP">Cumulative probability threshold for nucleus sampling</param>
    /// <param name="cfgFilterTopK">Number of top logits to consider for CFG filtering</param>
    /// <param name="audioPrompt">Optional list of audio prompt tensors for conditioning</param>
    /// <param name="audioPromptPath">Optional list of paths to audio files for prompts</param>
    /// <param name="verbose">Whether to print progress information</param>
    /// <returns>List of generated audio waveforms as float arrays</returns>
    public List<float[]> Generate(
        List<string> text,
        int? maxTokens = null,
        float cfgScale = 3.0f,
        float temperature = 1.2f,
        float topP = 0.95f,
        int cfgFilterTopK = 45,
        List<Tensor>? audioPrompt = null,
        List<string>? audioPromptPath = null,
        bool? verbose = null)
    {
        if (verbose is not null)
        {
            _verbose = verbose.Value;
        }
        using var inference = torch.inference_mode();
        using var scope = torch.NewDisposeScope();

        // Note: Need to ensure values put into tensors are either created or cast to int64
        // because TorchSharp does not implicitly convert types for CUDA tensors.

        // Initialize parameters
        int batchSize = text.Count;
        var audioEosValue = _config.Data.AudioEosValue;
        var audioPadValue = _config.Data.AudioPadValue;
        var delayPattern = _config.Data.DelayPattern;
        maxTokens ??= _config.Data.AudioLength;
        var maxDelayPattern = (long)delayPattern.Max();
        var delayPatternCx = torch.tensor(delayPattern, dtype: torch.int64, device: Device);

        // Set model to eval mode
        _model.eval();

        Stopwatch? totalTimer = null;
        if (_verbose)
        {
            totalTimer = Stopwatch.StartNew();
            Console.WriteLine("generate: starting generation");
        }

        var prompts = audioPrompt is null ? LoadAudioPrompts(audioPromptPath)
                    : [.. audioPrompt, .. LoadAudioPrompts(audioPromptPath)];

        // Encode text and capture text lengths for speed factor calculation
        var encodedTexts = text.Select(t => EncodeText(t)).ToArray();
        var textLengths = encodedTexts.Select(t => (int)t.shape[0]).ToArray();

        var processedText = PadTextInput(encodedTexts);
        var (decState, decOutput) = PrepareGeneration(processedText, prompts, maxTokens);
        int decStep = decOutput.PrefillSteps.Min() - 1;

        int currentIdx = decStep;
        Tensor eosDetectedBx = torch.zeros(batchSize, dtype: ScalarType.Bool, device: Device);
        Tensor eosCountdownBx = torch.full(batchSize, value: -1L, dtype: torch.int64, device: Device);
        Tensor finishedStepBx = torch.full(batchSize, value: -1L, dtype: torch.int64, device: Device);

        var bosOver = false;
        Stopwatch? stepTimer = null;

        if (_verbose)
        {
            Console.WriteLine("generate: starting generation");
            stepTimer = Stopwatch.StartNew();
        }

        // Generation Loop
        while (decStep < maxTokens)
        {
            if (eosCountdownBx.eq(0).all().item<bool>())
            {
                break;
            }

            var currentStepIdx = decStep + 1;
            decState.PrepareStep(decStep);

            // Get current tokens and expand for CFG
            var tokensBx1xC = decOutput.GetTokensAt(decStep).repeat_interleave(2, dim: 0);

            // Generate next tokens
            var predBxC = DecoderStep(
                tokensBx1xC,
                decState,
                cfgScale,
                temperature,
                topP,
                cfgFilterTopK,
                currentIdx);

            currentIdx += 1;

            // Handle EOS detection and propagation
            using (var activeMaskBx = eosCountdownBx.ne(0))
            using (var eosTriggerBx = torch.zeros_like(activeMaskBx))
            {
                if (activeMaskBx.any().item<bool>())
                {
                    var isEosToken = eosDetectedBx[activeMaskBx].logical_not()
                        .logical_and(predBxC.index(activeMaskBx, 0).eq(audioEosValue));
                    var isMaxLen = currentStepIdx >= maxTokens - maxDelayPattern;
                    eosTriggerBx[activeMaskBx] = isEosToken.logical_or(isMaxLen);
                }

                eosDetectedBx = eosDetectedBx.logical_or(eosTriggerBx);
                var startCountdownMaskBx = eosTriggerBx.logical_and(eosCountdownBx.lt(0));

                if (startCountdownMaskBx.any().item<bool>())
                {
                    eosCountdownBx[startCountdownMaskBx] = maxDelayPattern;
                    finishedStepBx[startCountdownMaskBx] = (long)currentStepIdx;
                }
            }

            // Handle padding after EOS
            using (var paddingMaskBx = eosCountdownBx.gt(0))
            {
                if (paddingMaskBx.any().item<bool>())
                {
                    var predActiveBxC = predBxC.index(paddingMaskBx).clone();
                    var countdownActiveBx = eosCountdownBx.index(paddingMaskBx);
                    var stepAfterEosBx = maxDelayPattern - countdownActiveBx;
                    var stepAfterEosBx_ = stepAfterEosBx.unsqueeze(1);
                    var delayPatternCx_ = delayPatternCx.unsqueeze(0);

                    var eosMaskNxC = stepAfterEosBx_.eq(delayPatternCx_);
                    var padMaskNxC = stepAfterEosBx_.gt(delayPatternCx_);
                    predActiveBxC[eosMaskNxC] = (long)audioEosValue;
                    predActiveBxC[padMaskNxC] = (long)audioPadValue;
                    predBxC[paddingMaskBx] = predActiveBxC;
                    eosCountdownBx[paddingMaskBx] -= 1;
                }
            }

            // Update BOS state
            if (!bosOver)
            {
                bosOver = decOutput.PrefillSteps.All(prefillStep =>
                        decStep - prefillStep > maxDelayPattern);
            }

            decOutput.UpdateOne(predBxC, currentStepIdx, !bosOver);
            decStep++;

            if (_verbose && decStep % 86 == 0)
            {
                stepTimer?.Stop();
                var duration = stepTimer?.Elapsed.TotalSeconds ?? 1;
                Console.WriteLine($"duration time {duration:F3}s");

                if (duration > 0)
                {
                    double speed = 86 * batchSize / duration;
                    double realtimeFactor = batchSize / duration;
                    Console.WriteLine($"generate step {decStep}: speed={speed:F3} tokens/s, realtime factor={realtimeFactor:F3}x");
                }
                stepTimer?.Restart();
            }
        }

        // Finalize generation
        var finalStep = decStep + 1;
        finishedStepBx[finishedStepBx == -1] = finalStep - maxDelayPattern;
        var prefillStepsTensor = torch.tensor(decOutput.PrefillSteps, device: _device);

        var lengthsBx = finishedStepBx - prefillStepsTensor;
        torch.clamp_(lengthsBx, min: 0);

        var maxLen = lengthsBx.max().item<long>() + maxDelayPattern;
        if (maxLen > 0)
        {
            int numChannels = _config.Data.Channels;
            var generatedCodes = torch.full(
                size: new[] { batchSize, maxLen, numChannels },
                value: audioPadValue,
                dtype: torch.int64,
                device: _device);

            for (int i = 0; i < batchSize; i++)
            {
                int startStep = decOutput.PrefillSteps[i];
                var actualLen = lengthsBx[i].item<long>() + maxDelayPattern;
                Console.WriteLine($"Length of lengthsBx for batch {i} after decode: {actualLen}");
                if (actualLen > 0)
                {
                    Tensor tokensToCopy = decOutput.GeneratedTokens[i, TensorIndex.Slice(startStep, startStep + actualLen), TensorIndex.Colon];
                    generatedCodes[i, TensorIndex.Slice(stop: actualLen), TensorIndex.Colon] = tokensToCopy;
                }
            }

            if (_verbose && totalTimer is not null)
            {
                totalTimer.Stop();
                var seconds = totalTimer.Elapsed.TotalSeconds;
                var realTimeSeconds = finalStep / 86.0; // 86 steps = 1 second realtime

                Console.WriteLine($"generate: total duration={seconds:F3}s, total steps={finalStep}");
                Console.WriteLine($"total generate speed={finalStep / seconds:F3} tokens/s, total realtime factor={batchSize * realTimeSeconds / seconds:F3}x");
            }

            decState.Dispose();
            return GenerateOutput(generatedCodes.MoveToOuterDisposeScope(), lengthsBx.MoveToOuterDisposeScope(), textLengths);
        }

        Console.WriteLine("Warning: Nothing was generated.");
        return [];
    }

    /// <summary>
    /// Loads audio from a file and encodes it using the DAC model.
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <returns>Encoded audio tensor</returns>
    public Tensor? LoadAudio(string audioPath)
    {
        if (!File.Exists(audioPath))
        {
            return null;
        }

        float[] audioData;
        int sampleRate;
        int channels;

        using (var reader = new AudioFileReader(audioPath))
        {
            sampleRate = reader.WaveFormat.SampleRate;
            channels = reader.WaveFormat.Channels;
            var length = (int)reader.Length;
            var bytesPerSample = reader.WaveFormat.BitsPerSample / 8;
            var samplesPerChannel = length / (bytesPerSample * channels);

            // Read samples
            var buffer = new float[samplesPerChannel * channels];
            reader.Read(buffer, 0, buffer.Length);

            // Convert to mono if needed
            if (channels > 1)
            {
                var monoData = new float[samplesPerChannel];
                for (int i = 0; i < samplesPerChannel; i++)
                {
                    float sum = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        sum += buffer[(i * channels) + c];
                    }
                    monoData[i] = sum / channels;
                }
                audioData = monoData;
            }
            else
            {
                audioData = buffer;
            }
        }
        if (sampleRate != _config.SampleRate)
        {
            Console.WriteLine($"Warning: Sample rate mismatch ({sampleRate} != {_config.SampleRate})");
            Core.Utils.AudioUtils.ResampleLinear(audioData, sampleRate, _config.SampleRate);
        }

        return Encode(tensor(audioData).reshape(1, -1).to(_device));
    }

    /// <summary>
    /// Saves audio to a WAV file.
    /// </summary>
    /// <param name="path">Output file path</param>
    /// <param name="audio">Audio data tensor to save</param>
    public static void SaveAudio(string path, Tensor audio)
    {
        using var scope = NewDisposeScope();
        using var cpuAudio = audio.detach().to(ScalarType.Float32);

        // Get the actual number of audio samples
        var numSamples = cpuAudio.numel();
        var audioData = new float[numSamples];

        using var flatAudio = cpuAudio.view(-1);
        var dataSpan = flatAudio.data<float>();
        dataSpan.CopyTo(audioData);

        SaveAudio(path, audioData);
    }

    public static void SaveAudio(string path, float[] audio)
    {
        // Normalize the audio
        var absMax = audio.Select(Math.Abs).Max();
        if (absMax > 10.0f)
        {
            Console.WriteLine($"Warning: Audio values out of expected range. Normalizing from {absMax} to [-1, 1]");
            for (int i = 0; i < audio.Length; i++)
            {
                audio[i] /= absMax;
            }
        }

        using var writer = new WaveFileWriter(path, new WaveFormat(DefaultSampleRate, 16, 1));

        // Convert float array to Int16 samples for 16-bit WAV format
        // NAudio expects integer samples for WaveFileWriter
        var intSamples = new short[audio.Length];
        for (int i = 0; i < audio.Length; i++)
        {
            // Clamp to [-1, 1] range before conversion
            var clampedSample = Math.Max(-1.0f, Math.Min(1.0f, audio[i]));
            intSamples[i] = (short)(clampedSample * 32767);
        }

        writer.WriteSamples(intSamples, 0, intSamples.Length);
    }

    /// <summary>
    /// Disposes resources.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _model?.Dispose();
                _dacModel?.Dispose();
            }
            _model = null;
            _dacModel = null;
            _disposed = true;
        }
    }

    private static Tensor AdjustSpeed(Tensor audio, float speedFactor)
    {
        if (Math.Abs(speedFactor - 1.0f) < 1e-5f)
        {
            return audio.alias();
        }

        var originalLen = audio.size(-1);
        var targetLen = (int)(originalLen / speedFactor);
        if (targetLen <= 0 || targetLen == originalLen)
        {
            return audio.alias();
        }
        using var scope = NewDisposeScope();
        var xOriginal = torch.arange(originalLen, device: audio.device);
        var xResampled = torch.linspace(0, originalLen - 1, targetLen, device: audio.device);
        // Use linear interpolation for speed adjustment
        var audioTensorSped = TorchUtils.Interp(xResampled, xOriginal, audio);
        return audioTensorSped.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Decodes DAC codebook indices to audio waveform
    /// </summary>
    /// <param name="audioCodes">Input code tensor [T, C]</param>
    /// <returns>Audio waveform tensor [1, T]</returns>
    private Tensor Decode(Tensor audioCodes)
    {
        using var scope = torch.NewDisposeScope();
        // Reshape for DAC: [1, C, T]
        using var audioValues = _dacModel!.FromCodes(audioCodes.unsqueeze(0).transpose(1, 2));

        // Decode through DAC model, Remove batch dimension
        return _dacModel.Decode(audioValues).squeeze_().MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Encodes audio waveform into DAC codebook indices
    /// </summary>
    /// <param name="audio">Input audio tensor [C, T]</param>
    /// <returns>Encoded frame tensor [T, C]</returns>
    private Tensor Encode(Tensor audio)
    {
        using var scope = torch.NewDisposeScope();

        // Add batch dimension [1, C, T]
        audio = audio.unsqueeze(0);
        var (_, encodedFrame, _, _, _) = _dacModel.Encode(audio, sampleRate: _config.SampleRate);

        // Remove batch dim and transpose to [T, C]
        return encodedFrame
            .squeeze(0)
            .transpose(0, 1)
            .MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Converts generated delayed codes into audio waveforms.
    /// </summary>
    /// <param name="generatedCodes">Generated audio codes with delays [B, T_gen, C]</param>
    /// <param name="lengthsBx">Valid length of codes per batch item [B]</param>
    /// <param name="textLengths">Original text prompt lengths for speed factor calculation [B]</param>
    /// <returns>List of audio waveforms as numpy arrays</returns>
    private List<float[]> GenerateOutput(Tensor generatedCodes, Tensor lengthsBx, int[] textLengths)
    {
        int numChannels = _config.Data.Channels;
        int batchSize = (int)generatedCodes.shape[0];
        int sequenceLength = (int)generatedCodes.shape[1];
        var delayPattern = _config.Data.DelayPattern;
        int audioPadValue = _config.Data.AudioPadValue;
        int maxDelayPattern = delayPattern.Max();
        Tensor codebook;

        using (var scope = torch.NewDisposeScope())
        {
            // Build indices to undo the channel delays
            var revertPrecomp = AudioUtils.BuildRevertIndices(
                B: batchSize,
                T: sequenceLength,
                C: numChannels,
                delayPattern: delayPattern
            );

            // Apply the revert operation to undo delays
            var unmaskedCodebook = AudioUtils.RevertAudioDelay(
                audio_BxTxC: generatedCodes,
                padValue: audioPadValue,
                precomp: revertPrecomp,
                T: sequenceLength
            );

            // Remove the delay pattern padding
            codebook = unmaskedCodebook[TensorIndex.Colon, TensorIndex.Slice(stop: -maxDelayPattern), TensorIndex.Colon];
            // Clamp to valid codebook range [0, 1023]
            var minValidIndex = MinValidCodebookIndex;
            var maxValidIndex = MaxValidCodebookIndex;
            var invalidMask = codebook.lt(minValidIndex).logical_or(codebook.gt(maxValidIndex));
            codebook[invalidMask] = 0L;
            codebook.MoveToOuterDisposeScope();
        }

        var audios = new List<float[]>();

        // Speed factor calculation using configuration settings
        var speedfactor = 1.0f;
        // Process each item in batch
        if (_dacModel != null)
        {
            for (int i = 0; i < batchSize; i++)
            {
                var length = (int)lengthsBx[i].item<long>();
                var audioTensor = Decode(codebook[i, TensorIndex.Slice(stop: length), TensorIndex.Colon]);

                if (_config.SlowdownMode == AudioSlowdownMode.Dynamic)
                {
                    // Use text length instead of audio code length for speed factor calculation
                    var textLength = textLengths[i];

                    speedfactor = textLength > _config.DynamicSlowdownStartLength ?
                        1f - (_config.DynamicSlowdownMaxPercent * Math.Min(1f,
                            (textLength - _config.DynamicSlowdownStartLength) /
                            (_config.DynamicSlowdownMaxLength - _config.DynamicSlowdownStartLength))) :
                        1f;
                }
                else
                {
                    speedfactor = _config.StaticSlowdownFactor;
                }

                if (AudioCorrectionMethod == AudioSpeedCorrectionMethod.None ||
                    (_config.SlowdownMode == AudioSlowdownMode.Dynamic && Math.Abs(speedfactor - 1.0f) < 1e-6f))
                {
                    // No audio processing needed, just return raw audio
                    using var cpuTensor = audioTensor.cpu();
                    using var detachedTensor = cpuTensor.detach();
                    audios.Add(detachedTensor.data<float>().ToArray());
                    continue;
                }

                if (AudioCorrectionMethod is AudioSpeedCorrectionMethod.TorchSharp or
                    AudioSpeedCorrectionMethod.All)
                {
                    var slowedAudio = AdjustSpeed(audioTensor, speedfactor);
                    using var cpuTensor = slowedAudio.cpu();
                    using var detachedTensor = cpuTensor.detach();
                    audios.Add([.. detachedTensor.data<float>()]);
                }

                if (AudioCorrectionMethod is AudioSpeedCorrectionMethod.Hybrid or
                    AudioSpeedCorrectionMethod.All)
                {
                    try
                    {
                        var halfSlowedAudio = AdjustSpeed(audioTensor, (speedfactor + 1) / 2);

                        using var cpuTensor = halfSlowedAudio.cpu();
                        using var detachedTensor = cpuTensor.detach();
                        var audioFloats = detachedTensor.data<float>().ToArray();
                        var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(DefaultSampleRate, 1);
                        var sampleProvider = new NAudioUtils.FloatArraySampleProvider(waveFormat, audioFloats);

                        // Apply half resampling for speed adjustment
                        var targetSampleRate = (int)(DefaultSampleRate * (1 + ((1 - speedfactor) / 2)));
                        var highQualityResampler = new WdlResamplingSampleProvider(sampleProvider, targetSampleRate);

                        // Read resampled audio
                        var outputLength = (int)(audioFloats.Length / speedfactor);
                        var outputBuffer = new float[outputLength];
                        var samplesRead = highQualityResampler.Read(outputBuffer, 0, outputLength);

                        var processedAudio = new float[samplesRead];
                        Array.Copy(outputBuffer, processedAudio, samplesRead);
                        audios.Add(processedAudio);
                    }
                    catch (Exception e)
                    {
                        if (AudioCorrectionMethod != AudioSpeedCorrectionMethod.All)
                        {
                            throw new InvalidOperationException($"Hybrid slow/resample failed: {e.Message}", e);
                        }
                        else if (_verbose)
                        {
                            Debug.WriteLine($"Hybrid slow/resample failed: {e.Message}");
                        }
                    }
                }

                if (AudioCorrectionMethod is AudioSpeedCorrectionMethod.NAudioResampling or
                    AudioSpeedCorrectionMethod.All)
                {
                    try
                    {
                        using var cpuTensor = audioTensor.cpu();
                        using var detachedTensor = cpuTensor.detach();
                        var audioFloats = detachedTensor.data<float>().ToArray();
                        var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(DefaultSampleRate, 1);
                        var sampleProvider = new NAudioUtils.FloatArraySampleProvider(waveFormat, audioFloats);

                        // Apply full resampling for speed adjustment
                        var targetSampleRate = (int)(DefaultSampleRate * (1 + (1 - speedfactor)));
                        var highQualityResampler = new WdlResamplingSampleProvider(sampleProvider, targetSampleRate);

                        // Read resampled audio
                        var outputLength = (int)(audioFloats.Length / speedfactor);
                        var outputBuffer = new float[outputLength];
                        var samplesRead = highQualityResampler.Read(outputBuffer, 0, outputLength);

                        var processedAudio = new float[samplesRead];
                        Array.Copy(outputBuffer, processedAudio, samplesRead);
                        audios.Add(processedAudio);
                    }
                    catch (Exception e)
                    {
                        if (AudioCorrectionMethod != AudioSpeedCorrectionMethod.All)
                        {
                            throw new InvalidOperationException($"NAudio resampling failed: {e.Message}", e);
                        }
                        else if (_verbose)
                        {
                            Debug.WriteLine($"NAudio resampling failed: {e.Message}");
                        }
                    }
                }
            }
        }
        else
        {
            // Return raw codebook indices if DAC is not loaded
            for (int i = 0; i < batchSize; i++)
            {
                var length = (int)lengthsBx[i].item<long>();
                using var codebookSlice = codebook[i, TensorIndex.Slice(stop: length), TensorIndex.Colon];
                using var cpuTensor = codebookSlice.cpu();
                using var detachedTensor = cpuTensor.detach();
                audios.Add(detachedTensor.to(float32).data<float>().ToArray());
            }
        }

        return audios;
    }

    private List<Tensor> LoadAudioPrompts(List<string>? audioPromptPaths)
    {
        if (audioPromptPaths is null || audioPromptPaths.Count == 0)
        {
            return [];
        }

        var processedPrompts = new List<Tensor>();
        foreach (var path in audioPromptPaths)
        {
            if (LoadAudio(path) is Tensor audio)
            {
                processedPrompts.Add(audio);
            }
        }

        return processedPrompts;
    }

    /// <summary>
    /// Initializes the model state for generation.
    /// </summary>
    /// <param name="text">Input text tensor</param>
    /// <param name="audioPrompts">List of audio prompts (optional)</param>
    /// <param name="maxTokens">Maximum generation length (optional)</param>
    /// <returns>Tuple of decoder state and output</returns>
    private (DecoderInferenceState, DecoderOutput) PrepareGeneration(
        Tensor text,
        List<Tensor>? audioPrompts,
        int? maxTokens = null)
    {
        var batchSize = (int)text.shape[0];

        // Prepare conditional and unconditional inputs
        using var encInputUncond = torch.zeros_like(text);
        using var stackedInputs = torch.stack([encInputUncond, text], dim: 1);
        using var encInput = stackedInputs.view(2 * batchSize, -1);

        // Initialize encoder state and get output
        var encState = EncoderInferenceState.New(_config, text);
        var encoderOut = _model.forward(encInput, encState);

        // Prepare decoder state
        List<KVCache> decCrossAttnCache = _model.Decoder.PrecomputeCrossAttnCache(
            encoderOut, encState.Positions, encState.PaddingMask);

        var decState = DecoderInferenceState.New(
            _config, encState, encoderOut, decCrossAttnCache, _computeDtype, maxTokens);

        // Prepare audio prompt
        var (prefill, prefillSteps) = PrepareAudioPrompt(batchSize, audioPrompts);

        // Initialize decoder output
        var decOutput = DecoderOutput.New(batchSize, _config, Device);
        decOutput.Prefill(prefill, prefillSteps);

        // Handle prefill steps if needed
        var decStep = prefillSteps.Min() - 1;
        if (decStep > 0)
        {
            decState.PrepareStep(0, decStep);
            using var tokens = decOutput.GetTokensAt(0, decStep);
            using var expandedTokens = tokens.repeat_interleave(2, dim: 0);
            _model.Decoder.forward(expandedTokens, decState);
        }

        return (decState.MoveToOuterDisposeScope(), decOutput.MoveToOuterDisposeScope());
    }
}