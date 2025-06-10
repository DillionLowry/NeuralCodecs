using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch.Config.Dia;

/// <summary>
/// Configuration for data loading and preprocessing.
/// </summary>
public class DataConfig
{
    /// <summary>
    /// Maximum length of text sequences (must be multiple of 128).
    /// </summary>
    [JsonPropertyName("text_length")]
    public int TextLength { get; set; } = 1024;

    /// <summary>
    /// Maximum length of audio sequences (must be multiple of 128).
    /// </summary>
    [JsonPropertyName("audio_length")]
    public int AudioLength { get; set; } = 3072;

    /// <summary>
    /// Number of audio channels.
    /// </summary>
    [JsonPropertyName("channels")]
    public int Channels { get; set; } = 9;

    /// <summary>
    /// Value used for padding text sequences.
    /// </summary>
    [JsonPropertyName("text_pad_value")]
    public int TextPadValue { get; set; } = 0;

    /// <summary>
    /// Value representing the end of audio sequences.
    /// </summary>
    [JsonPropertyName("audio_eos_value")]
    public int AudioEosValue { get; set; } = 1024;

    /// <summary>
    /// Value representing the beginning of audio sequences.
    /// </summary>
    [JsonPropertyName("audio_bos_value")]
    public int AudioBosValue { get; set; } = 1026;

    /// <summary>
    /// Value used for padding audio sequences.
    /// </summary>
    [JsonPropertyName("audio_pad_value")]
    public int AudioPadValue { get; set; } = 1025;

    /// <summary>
    /// List of delay values for each audio channel.
    /// </summary>
    [JsonPropertyName("delay_pattern")]
    public int[] DelayPattern { get; set; } = new[] { 0, 8, 9, 10, 11, 12, 13, 14, 15 };

    /// <summary>
    /// Creates a new DataConfig instance.
    /// </summary>
    /// <param name="textLength">Maximum length of text sequences (must be multiple of 128).</param>
    /// <param name="audioLength">Maximum length of audio sequences (must be multiple of 128).</param>
    /// <param name="channels">Number of audio channels.</param>
    /// <param name="textPadValue">Value used for padding text sequences.</param>
    /// <param name="audioEosValue">Value representing the end of audio sequences.</param>
    /// <param name="audioBosValue">Value representing the beginning of audio sequences.</param>
    /// <param name="audioPadValue">Value used for padding audio sequences.</param>
    /// <param name="delayPattern">List of delay values for each audio channel.</param>
    public DataConfig(
        int textLength = 1024,
        int audioLength = 3072,
        int channels = 9,
        int textPadValue = 0,
        int audioEosValue = 1024,
        int audioPadValue = 1025,
        int audioBosValue = 1026,
        int[] delayPattern = null)
    {
        // Validate and adjust textLength to be a multiple of 128
        int adjustedTextLength = (textLength + 127) / 128 * 128;
        if (textLength <= 0)
        {
            throw new ArgumentException("TextLength must be positive", nameof(textLength));
        }

        // Validate and adjust audioLength to be a multiple of 128
        int adjustedAudioLength = (audioLength + 127) / 128 * 128;
        if (audioLength <= 0)
        {
            throw new ArgumentException("AudioLength must be positive", nameof(audioLength));
        }

        // Validate channels
        if (channels <= 0)
        {
            throw new ArgumentException("Channels must be positive", nameof(channels));
        }

        TextLength = adjustedTextLength;
        AudioLength = adjustedAudioLength;
        Channels = channels;
        TextPadValue = textPadValue;
        AudioEosValue = audioEosValue;
        AudioPadValue = audioPadValue;
        AudioBosValue = audioBosValue;
        DelayPattern = delayPattern ?? new[] { 0, 8, 9, 10, 11, 12, 13, 14, 15 };

        if (Channels != DelayPattern.Length)
        {
            throw new ArgumentException($"Number of channels ({Channels}) must match delay pattern length ({DelayPattern.Length})");
        }
    }
}