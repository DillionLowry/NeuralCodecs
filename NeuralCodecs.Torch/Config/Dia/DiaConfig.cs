using NeuralCodecs.Core.Configuration;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NeuralCodecs.Torch.Config.Dia;

/// <summary>
/// The main configuration for Dia TTS, including model, data processing, and device settings.
/// </summary>
public class DiaConfig : IModelConfig
{
    /// <summary>
    /// Data loading and processing configuration.
    /// </summary>
    [JsonPropertyName("data")]
    public DataConfig Data { get; set; } = new DataConfig();

    /// <summary>
    /// Model architecture configuration.
    /// </summary>
    [JsonPropertyName("model")]
    public DiaModelConfig Model { get; set; } = new DiaModelConfig();

    /// <summary>
    /// Configuration version string.
    /// </summary>
    [JsonPropertyName("version")]
    public string Version { get; set; } = "0.1";

    /// <summary>
    /// Path to the Dia model.
    /// </summary>
    [JsonIgnore]
    public string ModelPath { get; set; } = "nari-labs/Dia-1.6B";

    /// <summary>
    /// Path to the DAC model for audio decoding.
    /// </summary>
    [JsonIgnore]
    public string DACModelPath { get; set; } = "descript/dac_44khz";

    /// <summary>
    /// CFG (Classifier-Free Guidance) scale for generation.
    /// </summary>
    [JsonIgnore]
    public float CfgScale { get; set; } = 3.0f;

    /// <summary>
    /// Temperature for sampling during generation.
    /// </summary>
    [JsonIgnore]
    public float Temperature { get; set; } = 1.3f;

    /// <summary>
    /// Top-p (nucleus) sampling parameter.
    /// </summary>
    [JsonIgnore]
    public float TopP { get; set; } = 0.95f;

    /// <summary>
    /// Scale factor for classifier-free guidance.
    /// </summary>
    [JsonIgnore]
    public int CfgFilterTopK { get; set; } = 45;

    /// <summary>
    /// Enable verbose logging during inference.
    /// </summary>
    [JsonIgnore]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Compute data type for model operations.
    /// </summary>
    [JsonIgnore]
    public ComputeDtype ComputeDtype { get; set; } = ComputeDtype.Float32;

    /// <summary>
    /// Device configuration for model execution.
    /// </summary>
    [JsonIgnore]
    public DeviceConfiguration Device { get; set; } = DeviceConfiguration.CUDA();

    /// <summary>
    /// DAC sample rate for audio synthesis.
    /// </summary>
    [JsonIgnore]
    public int SampleRate { get; set; } = 44100;

    /// <summary>
    /// Whether to load the DAC model for decoding.
    /// </summary>
    [JsonIgnore]
    public bool LoadDACModel { get; set; } = true;

    /// <summary>
    /// Model architecture identifier.
    /// </summary>
    [JsonIgnore]
    public string Architecture { get; set; } = string.Empty;

    /// <summary>
    /// Additional metadata for the configuration.
    /// </summary>
    [JsonIgnore]
    public IDictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

    /// <summary>
    /// Random seed for reproducible generation.
    /// </summary>
    [JsonIgnore]
    public int? Seed { get; set; } = null;

    /// <summary>
    /// Audio slowdown mode configuration.
    /// </summary>
    [JsonIgnore]
    public AudioSlowdownMode SlowdownMode { get; set; } = AudioSlowdownMode.Dynamic;

    /// <summary>
    /// Static slowdown factor when using static mode (e.g., 0.95 for 5% slowdown).
    /// </summary>
    [JsonIgnore]
    public float StaticSlowdownFactor { get; set; } = 0.95f;

    /// <summary>
    /// Audio processing method for speed adjustment.
    /// </summary>
    [JsonIgnore]
    public AudioSpeedCorrectionMethod SpeedCorrectionMethod { get; set; } = AudioSpeedCorrectionMethod.Hybrid;

    /// <summary>
    /// Text length threshold for dynamic slowdown to start (in characters).
    /// </summary>
    [JsonIgnore]
    public float DynamicSlowdownStartLength { get; set; } = 400f;

    /// <summary>
    /// Text length where maximum dynamic slowdown is reached (in characters).
    /// </summary>
    [JsonIgnore]
    public float DynamicSlowdownMaxLength { get; set; } = 750f;

    /// <summary>
    /// Maximum slowdown percentage for dynamic mode (0.20 = 20% slowdown).
    /// </summary>
    [JsonIgnore]
    public float DynamicSlowdownMaxPercent { get; set; } = 0.20f;

    /// <summary>
    /// Initializes a new instance of the <see cref="DiaConfig"/> class.
    /// </summary>
    /// <param name="model">Model architecture configuration.</param>
    /// <param name="data">Data loading and processing configuration.</param>
    /// <param name="version">Configuration version string.</param>
    public DiaConfig(
        DiaModelConfig model,
        DataConfig data,
        string version = "0.1")
    {
        Model = model ?? new DiaModelConfig();
        Data = data ?? new DataConfig();
        Version = version;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DiaConfig"/> class.
    /// </summary>
    /// <remarks>This constructor initializes the <see cref="Model"/> and <see cref="Data"/> properties with
    /// their default configurations.</remarks>
    public DiaConfig()
    {
        Model = new DiaModelConfig();
        Data = new DataConfig();
    }

    /// <summary>
    /// Save the current configuration instance to a JSON file.
    /// </summary>
    /// <param name="path">The target file path to save the configuration.</param>
    public void Save(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }
        if (Path.GetDirectoryName(path) is not string outPath)
        {
            throw new ArgumentException("Directory does not exist", nameof(path));
        }
        Directory.CreateDirectory(outPath);
        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };
        string json = JsonSerializer.Serialize(this, options);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Load and validate a Dia configuration from a JSON file.
    /// </summary>
    /// <param name="path">The path to the configuration file.</param>
    /// <returns>A validated DiaConfig instance if the file exists and is valid, otherwise null.</returns>
    public static DiaConfig Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Configuration file not found: {path}");
        }

        if (!Path.GetExtension(path).Equals(".json", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Configuration must be a .json file: {path}");
        }

        try
        {
            string json = File.ReadAllText(path);
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
            return JsonSerializer.Deserialize<DiaConfig>(json, options)!;
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException($"Failed to parse configuration file: {ex.Message}", ex);
        }
    }
}