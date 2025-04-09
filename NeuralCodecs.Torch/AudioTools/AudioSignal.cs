using NAudio.Wave;
using NeuralCodecs.Core.Utils;
using System.Numerics;
using System.Threading.Channels;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.AudioTools;

/// <summary>
/// Represents an audio signal with support for batch processing, STFT analysis, and various audio transformations.
/// Provides functionality for loading, manipulating, and analyzing audio data using TorchSharp tensors.
/// Based on Descript's audiotools python library.
/// </summary>
public class AudioSignal : IDisposable
{
    #region Fields

    private static readonly Dictionary<(int NumMfcc, int NumMels), Tensor> _dctMatrixCache = new();
    private static readonly Dictionary<(int SampleRate, int NfftSize, int NumMels, float FreqMin, float FreqMax), float[,]> _melFilterbankCache = new();
    private static readonly float _minLoudnessDb = -70.0f;
    private static readonly Dictionary<(string WindowType, int WindowSize, string DeviceType), Tensor> _windowCache = new();
    private Tensor _audioData;
    private bool _disposed = false;
    private Tensor _loudness;
    private int _originalSignalLength;
    private Tensor _stftData;
    private STFTParams _stftParams;

    #endregion Fields

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the AudioSignal class.
    /// </summary>
    /// <param name="audioData">Optional tensor containing the audio data in format (batch, channels, samples)</param>
    /// <param name="sampleRate">Sample rate of the audio in Hz</param>
    /// <param name="stftParams">Optional STFT parameters for spectral analysis</param>
    /// <param name="offset">Time offset in seconds for audio loading</param>
    /// <param name="duration">Duration in seconds for audio loading</param>
    /// <param name="device">Target device for tensor operations (cpu/cuda)</param>
    public AudioSignal(
        Tensor audioData = null,
        int sampleRate = 44100,
        STFTParams stftParams = null,
        float offset = 0,
        float? duration = null,
        string device = null)
    {
        Metadata = new Dictionary<string, object>
            {
                {"offset", offset},
                {"duration", duration}
            };

        if (audioData is not null)
        {
            LoadFromArray(audioData, sampleRate, device);
        }

        _stftParams = stftParams;
    }

    public AudioSignal(string audioPath, float offset = 0, float? duration = null, string device = null)
    {
        using var reader = new AudioFileReader(audioPath);
        var sampleRate = reader.WaveFormat.SampleRate;
        var channels = reader.WaveFormat.Channels;
        Console.WriteLine($"Loading {audioPath} with offset {offset} and duration {duration?.ToString() ?? "null"}");
        // Calculate start and length in samples
        var startSample = (int)(offset * sampleRate);
        var lengthInSamples = duration.HasValue
            ? (int)(duration.Value * sampleRate)
            : (int)((reader.Length / reader.WaveFormat.BlockAlign) - startSample);
        Console.WriteLine($"Reading {lengthInSamples} samples from {startSample} to {startSample + lengthInSamples}");

        // Seek to offset position
        reader.Position = startSample * reader.WaveFormat.BlockAlign;

        // Read specified duration into buffer
        var buffer = new List<float>();
        var readBuffer = new float[reader.WaveFormat.SampleRate * channels];

        var remainingSamples = lengthInSamples;
        while (remainingSamples > 0)
        {
            var samplesToRead = Math.Min(readBuffer.Length, remainingSamples);
            // Adjust to ensure complete blocks
            samplesToRead = samplesToRead - (samplesToRead % reader.WaveFormat.BlockAlign);

            var samplesRead = reader.Read(readBuffer, 0, samplesToRead);
            if (samplesRead == 0)
            {
                break;
            }

            buffer.AddRange(readBuffer.Take(samplesRead));
            remainingSamples -= samplesRead;
        }

        var samples = buffer.ToArray();

        // Convert to tensor and reshape for channels
        var tensor = torch.tensor(samples)
            .reshape(1, channels, -1);

        // Move to specified device if provided
        if (!string.IsNullOrEmpty(device))
        {
            tensor = tensor.to(device);
        }

        AudioData = tensor;
        SampleRate = sampleRate;
        PathToFile = audioPath;
    }

    #endregion Constructors

    #region Properties

    /// <summary>
    /// Gets or sets the audio data tensor in format (batch, channels, samples).
    /// Setting this property will reset any cached loudness calculations.
    /// </summary>
    public Tensor AudioData
    {
        get => _audioData;
        set
        {
            if (value is not null)
            {
                if (!is_tensor(value))
                {
                    throw new ArgumentException("Audio data must be a torch Tensor");
                }
                if (value.ndim != 3)
                {
                    throw new ArgumentException("Audio data must be 3-dimensional (B,C,T)");
                }
            }
            _audioData = value;
            _loudness = null;
        }
    }

    /// <summary>
    /// Gets the number of samples in the batch dimension
    /// </summary>
    public int BatchSize => (int)_audioData.size(0);

    /// <summary>
    /// Gets the device (CPU/CUDA) where the audio data is stored
    /// </summary>
    public string Device => _audioData?.device.type.ToString() ?? "unknown";

    /// <summary>
    /// Gets the magnitude spectrum of the STFT data.
    /// Performs STFT if not already computed.
    /// </summary>
    public Tensor Magnitude
    {
        get
        {
            if (_stftData is null)
            {
                STFT();
            }
            return abs(_stftData);
        }
        set
        {
            _stftData = value * exp(Complex.ImaginaryOne * Phase);
        }
    }

    /// <summary>
    /// Gets or sets metadata associated with the audio signal
    /// </summary>
    public Dictionary<string, object> Metadata { get; private set; }

    /// <summary>
    /// Gets the number of audio channels
    /// </summary>
    public int NumChannels => (int)_audioData.size(1);

    /// <summary>
    /// Gets or sets the path to the source audio file
    /// </summary>
    public string PathToFile { get; private set; }

    public Tensor Phase
    {
        get
        {
            if (_stftData is null)
            {
                STFT();
            }
            return angle(_stftData);
        }
        set
        {
            _stftData = Magnitude * exp(Complex.ImaginaryOne * value);
        }
    }

    /// <summary>
    /// Gets the sample rate in Hz
    /// </summary>
    public int SampleRate { get; private set; }

    /// <summary>
    /// Gets the duration of the signal in seconds
    /// </summary>
    public float SignalDuration => SignalLength / (float)SampleRate;

    /// <summary>
    /// Gets the length of the signal in samples
    /// </summary>
    public int SignalLength => (int)_audioData.size(-1);

    public Tensor StftData
    {
        get => _stftData;
        set
        {
            if (value is not null)
            {
                if (!is_tensor(value) || !value.is_complex())
                {
                    throw new ArgumentException("STFT data must be a complex torch Tensor");
                }
                if (_stftData is not null && !_stftData.shape.SequenceEqual(value.shape))
                {
                    Console.WriteLine("Warning: STFT data changed shape");
                }
            }
            _stftData = value;
        }
    }

    /// <summary>
    /// Gets or sets the STFT parameters used for spectral analysis.
    /// </summary>
    public STFTParams StftParams
    {
        get => _stftParams;
        set
        {
            int defaultWinLen = NextPowerOfTwo((int)(0.032f * SampleRate));
            int defaultHopLen = defaultWinLen / 4;
            string defaultWinType = "hann";
            bool defaultMatchStride = false;
            PaddingModes defaultPaddingType = PaddingModes.Reflect;

            var defaultParams = new STFTParams(
                windowLength: defaultWinLen,
                hopLength: defaultHopLen,
                windowType: defaultWinType,
                matchStride: defaultMatchStride,
                paddingMode: defaultPaddingType
            );

            if (value == null)
            {
                _stftParams = defaultParams;
            }
            else
            {
                // Fill in any null values with defaults
                _stftParams = new STFTParams(
                    windowLength: value.WindowLength == 0 ? defaultParams.WindowLength : value.WindowLength,
                    hopLength: value.HopLength == 0 ? defaultParams.HopLength : value.HopLength,
                    windowType: value.WindowType ?? defaultParams.WindowType,
                    matchStride: value.MatchStride,
                    paddingMode: value.PaddingMode
                );
            }

            // Reset STFT data since parameters changed
            _stftData = null;
        }
    }

    #endregion Properties

    #region Equality Members

    public static bool operator !=(AudioSignal left, AudioSignal right)
    {
        return !Equals(left, right);
    }

    public static bool operator ==(AudioSignal left, AudioSignal right)
    {
        return Equals(left, right);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (obj is AudioSignal other)
        {
            return _audioData.Equals(other._audioData) &&
                   _loudness.Equals(other._loudness) &&
                   _originalSignalLength == other._originalSignalLength &&
                   _stftData.Equals(other._stftData) &&
                   _stftParams.Equals(other._stftParams);
        }
        return false;
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return HashCode.Combine(_originalSignalLength, _stftParams);
    }

    #endregion Equality Members

    #region Methods

    // Additional indexing support
    public AudioSignal this[params int[] indices]
    {
        get
        {
            var audioData = AudioData.index(indices);
            var loudness = _loudness?.index(indices);
            var stftData = _stftData?.index(indices);

            var copy = new AudioSignal(audioData, SampleRate, _stftParams);
            copy._loudness = loudness;
            copy._stftData = stftData;
            copy.PathToFile = PathToFile;
            return copy;
        }
        set
        {
            if (value.AudioData is not null && AudioData is not null)
            {
                AudioData.index_put_(indices, value.AudioData);
            }
            if (_loudness is not null && value._loudness is not null)
            {
                _loudness.index_put_(indices, value._loudness);
            }
            if (_stftData is not null && value._stftData is not null)
            {
                _stftData.index_put_(indices, value._stftData);
            }
        }
    }

    /// <summary>
    /// Creates a batch from multiple audio signals.
    /// </summary>
    /// <param name="audioSignals">List of audio signals to batch</param>
    /// <param name="padSignals">Whether to pad shorter signals to match the longest</param>
    /// <param name="truncateSignals">Whether to truncate longer signals to match the shortest</param>
    /// <param name="resample">Whether to resample signals to match the first signal's sample rate</param>
    /// <param name="dim">Dimension along which to concatenate the signals</param>
    /// <returns>A new AudioSignal containing the batched data</returns>
    public static AudioSignal Batch(
        List<AudioSignal> audioSignals,
        bool padSignals = false,
        bool truncateSignals = false,
        bool resample = false,
        int dim = 0)
    {
        var signalLengths = audioSignals.Select(x => x.SignalLength).ToList();
        var sampleRates = audioSignals.Select(x => x.SampleRate).ToList();

        if (sampleRates.Distinct().Count() != 1)
        {
            if (resample)
            {
                foreach (var x in audioSignals)
                {
                    x.Resample(sampleRates[0]);
                }
            }
            else
            {
                throw new InvalidOperationException(
                    $"Not all signals had the same sample rate! Got {string.Join(", ", sampleRates)}. " +
                    "All signals must have the same sample rate, or resample must be True.");
            }
        }

        if (signalLengths.Distinct().Count() != 1)
        {
            if (padSignals)
            {
                int maxLength = signalLengths.Max();
                foreach (var x in audioSignals)
                {
                    int padLen = maxLength - x.SignalLength;
                    x.ZeroPad(0, padLen);
                }
            }
            else if (truncateSignals)
            {
                int minLength = signalLengths.Min();
                foreach (var x in audioSignals)
                {
                    x.TruncateSamples(minLength);
                }
            }
            else
            {
                throw new InvalidOperationException(
                    $"Not all signals had the same length! Got {string.Join(", ", signalLengths)}. " +
                    "All signals must be the same length, or pad_signals/truncate_signals must be True.");
            }
        }

        var audioData = cat(audioSignals.Select(x => x.AudioData).ToList(), dim: dim);
        var audioPaths = audioSignals.Select(x => x.PathToFile).ToList();

        var batchedSignal = new AudioSignal(
            audioData,
            sampleRate: audioSignals[0].SampleRate);
        batchedSignal.PathToFile = audioPaths[0];

        return batchedSignal;
    }

    /// <summary>
    /// Creates a signal containing concatenated audio from multiple signals
    /// </summary>
    /// <param name="signals">List of signals to concatenate</param>
    /// <returns>A new AudioSignal containing the concatenated data</returns>
    public static AudioSignal ConcatBatch(List<AudioSignal> signals)
    {
        if (!signals.Any())
        {
            throw new ArgumentException("No signals provided for concatenation");
        }

        var firstSignal = signals[0];
        var tensorList = signals.Select(s => s.AudioData).ToList();
        var concatenated = cat(tensorList, dim: 0);

        return new AudioSignal(concatenated, firstSignal.SampleRate, firstSignal._stftParams);
    }

    /// <summary>
    /// Creates an audio excerpt from a file with optional offset and duration
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <param name="offset">Start time in seconds</param>
    /// <param name="duration">Duration in seconds</param>
    /// <param name="state">Random number generator for random offset selection</param>
    /// <returns>An AudioSignal containing the excerpt</returns>
    public static AudioSignal Excerpt(
       string audioPath,
       float? offset = null,
       float? duration = null,
       Random state = null)
    {
        var info = GetAudioInfo(audioPath);
        float totalDuration = info.Duration;

        state ??= new Random();
        float lowerBound = offset ?? 0;
        float upperBound = Math.Max(totalDuration - (duration ?? 0), 0);
        float actualOffset = lowerBound + (float)(state.NextDouble() * (upperBound - lowerBound));

        var signal = new AudioSignal(audioPath, offset: actualOffset, duration: duration);
        signal.Metadata["offset"] = actualOffset;
        signal.Metadata["duration"] = duration;

        return signal;
    }

    public static Tensor GetWindow(string windowType, int windowLength, string device)
    {
        var key = (windowType, windowLength, device);
        if (!_windowCache.ContainsKey(key))
        {
            Tensor window;
            if (windowType == "average")
            {
                window = ones(windowLength) / windowLength;
            }
            else if (windowType == "sqrt_hann")
            {
                window = sqrt(hann_window(windowLength));
            }
            else if (windowType == "hann")
            {
                window = hann_window(windowLength);
            }
            else
            {
                throw new ArgumentException($"Unsupported window type: {windowType}");
            }

            window = window.to(device).@float();
            _windowCache[key] = window;
        }
        return _windowCache[key];
    }

    public static AudioSignal operator -(AudioSignal a, AudioSignal b)
    {
        var result = a.Clone();
        result._audioData -= b._audioData;
        return result;
    }

    public static AudioSignal operator *(AudioSignal a, float scalar)
    {
        var result = a.Clone();
        result._audioData *= scalar;
        return result;
    }

    public static AudioSignal operator +(AudioSignal a, AudioSignal b)
    {
        var result = a.Clone();
        result._audioData += b._audioData;
        return result;
    }

    /// <summary>
    /// Creates a salient excerpt based on loudness criteria
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <param name="loudnessCutoff">Minimum loudness threshold in dB</param>
    /// <param name="numTries">Maximum number of attempts to find a suitable excerpt</param>
    /// <param name="state">Random number generator</param>
    /// <param name="kwargs">Additional parameters</param>
    /// <returns>An AudioSignal containing a salient excerpt</returns>
    public static AudioSignal SalientExcerpt(
        string audioPath,
        float? loudnessCutoff = null,
        int numTries = 8,
        Random state = null,
        Dictionary<string, object> kwargs = null)
    {
        state ??= new Random();
        AudioSignal excerpt;

        if (!loudnessCutoff.HasValue)
        {
            excerpt = Excerpt(audioPath, state: state);
        }
        else
        {
            float loudness;
            int numTry = 0;
            do
            {
                excerpt = Excerpt(audioPath, state: state);
                loudness = excerpt.Loudness().item<float>();
                numTry++;
                if (numTry >= numTries)
                {
                    break;
                }
            } while (loudness <= loudnessCutoff.Value);
        }
        return excerpt;
    }

    /// <summary>
    /// Generates a synthetic waveform
    /// </summary>
    /// <param name="frequency">Frequency in Hz</param>
    /// <param name="duration">Duration in seconds</param>
    /// <param name="sampleRate">Sample rate in Hz</param>
    /// <param name="numChannels">Number of channels</param>
    /// <param name="shape">Waveform shape (sine, square, sawtooth, triangle)</param>
    /// <returns>An AudioSignal containing the synthetic waveform</returns>
    public static AudioSignal Wave(
        float frequency,
        float duration,
        int sampleRate,
        int numChannels = 1,
        string shape = "sine")
    {
        int nSamples = (int)(duration * sampleRate);
        var t = linspace(0, duration, nSamples);
        Tensor waveData = shape.ToLower() switch
        {
            "sawtooth" => 2 * ((t * frequency) - floor((t * frequency) + 0.5f)),
            "square" => sign(sin(2 * (float)Math.PI * frequency * t)),
            "triangle" => (2 * abs(2 * ((t * frequency) - floor((t * frequency) + 0.5f)))) - 1,
            _ => sin(2 * (float)Math.PI * frequency * t),
        };
        waveData = waveData.unsqueeze(0).unsqueeze(0).repeat(1, numChannels, 1);
        return new AudioSignal(waveData, sampleRate);
    }

    /// <summary>
    /// Creates a signal filled with zeros
    /// </summary>
    public static AudioSignal Zeros(
        float duration,
        int sampleRate,
        int numChannels = 1,
        int batchSize = 1)
    {
        int nSamples = (int)(duration * sampleRate);
        return new AudioSignal(
            zeros(batchSize, numChannels, nSamples),
            sampleRate);
    }

    /// <summary>
    /// Creates a deep copy of the audio signal
    /// </summary>
    /// <returns>A new AudioSignal instance with cloned data</returns>
    public AudioSignal Clone()
    {
        var clone = new AudioSignal(
            _audioData?.clone(),
            SampleRate,
            _stftParams);

        if (_stftData is not null)
        {
            clone._stftData = _stftData.clone();
        }
        if (_loudness is not null)
        {
            clone._loudness = _loudness.clone();
        }

        clone.PathToFile = PathToFile;
        clone.Metadata = new Dictionary<string, object>(Metadata);

        return clone;
    }

    public (int rightPad, int pad) ComputeStftPadding(int windowLength, int hopLength, bool matchStride)
    {
        var length = SignalLength;

        if (matchStride)
        {
            if (hopLength != windowLength / 4)
            {
                throw new ArgumentException("For match_stride, hop must equal window_length // 4");
            }
            var rightPad = ((int)Math.Ceiling((double)length / hopLength) * hopLength) - length;
            var pad = (windowLength - hopLength) / 2;
            return (rightPad, pad);
        }

        return (0, 0);
    }

    /// <summary>
    /// Moves the audio data to CPU memory
    /// </summary>
    public AudioSignal Cpu()
    {
        return To("cpu");
    }

    /// <summary>
    /// Moves the audio data to CUDA device
    /// </summary>
    public AudioSignal Cuda()
    {
        return To("cuda");
    }

    /// <summary>
    /// Detaches the audio data from the computational graph
    /// </summary>
    public AudioSignal Detach()
    {
        if (_loudness is not null)
        {
            _loudness = _loudness.detach();
        }
        if (_stftData is not null)
        {
            _stftData = _stftData.detach();
        }
        _audioData = _audioData.detach();
        return this;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Converts the audio data to 32-bit floating point format
    /// </summary>
    public AudioSignal Float()
    {
        _audioData = _audioData.@float();
        return this;
    }

    public AudioSignal InverseSTFT(
        int? windowLength = null,
        int? hopLength = null,
        string windowType = null,
        bool? matchStride = null,
        int? length = null)
    {
        if (_stftData is null)
        {
            throw new InvalidOperationException("Cannot perform inverse STFT without STFT data!");
        }

        windowLength ??= _stftParams.WindowLength;
        hopLength ??= _stftParams.HopLength;
        windowType ??= _stftParams.WindowType;
        matchStride ??= _stftParams.MatchStride;

        var window = GetWindow(windowType, windowLength.Value, _stftData.device.type.ToString() ?? "unknown");

        var (nb, nch, nf, nt) = (_stftData.size(0), _stftData.size(1), _stftData.size(2), _stftData.size(3));
        var stftData = _stftData.reshape(nb * nch, nf, nt);

        var (rightPad, pad) = ComputeStftPadding(windowLength.Value, hopLength.Value, matchStride.Value);

        if (length == null)
        {
            length = _originalSignalLength;
            length += (2 * pad) + rightPad;
        }

        if (matchStride.Value)
        {
            // Add back the frames that were dropped in STFT
            stftData = nn.functional.pad(stftData, (2, 2));
        }

        var audioData = torch.istft(
            stftData,
            n_fft: windowLength.Value,
            hop_length: hopLength.Value,
            window: window,
            length: length.Value,
            center: true
        );

        audioData = audioData.reshape(nb, nch, -1);

        if (matchStride.Value)
        {
            audioData = audioData.slice(2, pad, -(pad + rightPad), 1);
        }

        _audioData = audioData;
        return this;
    }

    public void LoadFromArray(Tensor audioArray, int sampleRate, string device = null)
    {
        if (!is_tensor(audioArray))
        {
            throw new ArgumentException("Audio data must be a torch Tensor");
        }

        if (audioArray.ndim < 3)
        {
            // Add batch and channel dimensions if needed
            if (audioArray.ndim < 2)
            {
                audioArray = audioArray.unsqueeze(0);
            }
            audioArray = audioArray.unsqueeze(0);
        }

        _audioData = audioArray;
        _originalSignalLength = SignalLength;
        SampleRate = sampleRate;

        if (!string.IsNullOrEmpty(device))
        {
            To(device);
        }
    }

    public void LoadFromFile(string audioPath, float offset = 0, float? duration = null, string device = null)
    {
        using var audioFile = new AudioFileReader(audioPath);
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

        var tensorData = tensor(buffer.ToArray()).reshape(1, 1, -1);

        if (duration.HasValue)
        {
            var endSample = (int)(offset + duration.Value) * SampleRate;
            tensorData = tensorData.narrow(-1, (int)(offset * SampleRate), endSample);
        }

        LoadFromArray(tensorData, audioFile.WaveFormat.SampleRate, device);
        PathToFile = audioPath;
    }

    public Tensor LogMagnitude(float refValue = 1.0f, float amin = 1e-5f, float? topDb = 80.0f)
    {
        var magnitude = Magnitude;
        float aminSq = amin * amin;

        var logSpec = 10.0f * log10(magnitude.pow(2).clamp(min: aminSq));
        logSpec -= 10.0f * (float)Math.Log10(Math.Max(aminSq, refValue));

        if (topDb.HasValue)
        {
            var maxVal = logSpec.max();
            logSpec = maximum(logSpec, maxVal - topDb.Value);
        }

        return logSpec;
    }

    public Tensor Loudness(
      string filterClass = "K-weighting",
      float blockSize = 0.400f,
      Dictionary<string, object> kwargs = null)
    {
        if (_loudness is not null)
        {
            return _loudness.to(Device);
        }

        int originalLength = SignalLength;
        if (SignalDuration < 0.5f)
        {
            int padLen = (int)((0.5f - SignalDuration) * SampleRate);
            ZeroPad(0, padLen);
        }

        using var meter = new LoudnessMeter(
            SampleRate,
            filterClass: filterClass,
            blockSize: blockSize);

        var loudness = meter.IntegratedLoudness(this);

        // Restore original length
        TruncateSamples(originalLength);

        var minLoudness = ones_like(loudness, device: loudness.device) * _minLoudnessDb;
        _loudness = maximum(loudness, minLoudness);

        return _loudness.to(Device);
    }

    /// <summary>
    /// Computes the mel-spectrogram of the audio signal.
    /// </summary>
    /// <param name="nMels">Number of mel bands</param>
    /// <param name="melFmin">Minimum frequency</param>
    /// <param name="melFmax">Maximum frequency</param>
    /// <param name="stftParams">Optional STFT parameters</param>
    /// <returns>Mel-spectrogram tensor</returns>
    public Tensor MelSpectrogram(
        int nMels = 80,
        float melFmin = 0.0f,
        float? melFmax = null,
        Dictionary<string, object> stftParams = null)
    {
        var stft = stftParams != null ? STFT() : _stftData ?? STFT();
        var magnitude = abs(stft);

        int nf = (int)magnitude.size(2);
        var melBasis = MelFilterbank(
            SampleRate,
            2 * (nf - 1),
            nMels,
            melFmin,
            melFmax ?? SampleRate / 2f
        );

        var melSpec = matmul(magnitude.transpose(2, -1), melBasis.t());
        return melSpec.transpose(-1, 2);
    }

    public Tensor MFCC(
            int nMfcc = 40,
            int nMels = 80,
            float logOffset = 1e-6f,
            Dictionary<string, object> kwargs = null)
    {
        var melSpectrogram = MelSpectrogram(nMels, 0.0f, null, kwargs);
        melSpectrogram = log(melSpectrogram + logOffset);

        // Create DCT matrix
        var dctMatrix = DCTMatrix(nMfcc, nMels);
        var mfcc = matmul(melSpectrogram.transpose(-1, -2), dctMatrix);
        return mfcc.transpose(-1, -2);
    }

    /// <summary>
    /// Normalizes the audio signal to a target loudness level
    /// </summary>
    /// <param name="targetDb">Target loudness in dB</param>
    public AudioSignal Normalize(float targetDb = -24.0f)
    {
        var db = tensor(targetDb).to(Device);
        var refDb = Loudness();
        var gain = db - refDb;
        gain = exp(gain * LoudnessMeter.GAIN_FACTOR);

        AudioData *= gain.unsqueeze(-1).unsqueeze(-1);
        return this;
    }

    /// <summary>
    /// Resamples the audio to a new sample rate using linear interpolation
    /// </summary>
    /// <param name="sampleRate">Target sample rate in Hz</param>
    public void Resample(int sampleRate)
    {
        if (sampleRate == SampleRate)
        {
            return;
        }

        AudioData = DSP.ResampleLinear(
            _audioData,
            SampleRate,
            sampleRate);
        SampleRate = sampleRate;
    }

    /// <summary>
    /// Resamples the audio using a high-quality fractional resampling algorithm
    /// </summary>
    /// <param name="targetSampleRate">Target sample rate in Hz</param>
    public void ResampleFrac(int targetSampleRate)
    {
        if (targetSampleRate == SampleRate)
        {
            return;
        }

        long greatestCommonDivisor = MathUtils.GCD(SampleRate, targetSampleRate);
        int upsampleFactor = targetSampleRate / (int)greatestCommonDivisor;
        int downsampleFactor = SampleRate / (int)greatestCommonDivisor;

        // Upsample first
        var audioSignal = _audioData;
        if (upsampleFactor > 1)
        {
            // Insert zeros
            var signalShape = audioSignal.shape;
            signalShape[^1] *= upsampleFactor;
            var upsampledSignal = zeros(signalShape, device: audioSignal.device, dtype: audioSignal.dtype);
            upsampledSignal.index_copy_(-1,
                arange(0, upsampledSignal.size(-1), upsampleFactor, device: audioSignal.device),
                audioSignal);

            // Apply Low Pass Filter
            var filterCutoff = 1.0f / upsampleFactor;
            var filterHalfWidth = 64;
            var timeAxis = arange(-filterHalfWidth, filterHalfWidth + 1, device: audioSignal.device);
            var filterKernel = sinc(filterCutoff * timeAxis) * filterCutoff * upsampleFactor;
            filterKernel *= hamming_window((2 * filterHalfWidth) + 1, device: audioSignal.device);

            // Normalize filter
            filterKernel /= filterKernel.sum();

            // Reshape for batch convolution
            filterKernel = filterKernel.reshape(1, 1, -1);
            var paddingSize = filterKernel.size(-1) - 1;
            upsampledSignal = nn.functional.pad(upsampledSignal, [paddingSize, paddingSize],
                mode: PaddingModes.Reflect);

            // Apply filter
            audioSignal = nn.functional.conv1d(upsampledSignal, filterKernel);
        }

        // Downsample
        if (downsampleFactor > 1)
        {
            // Apply Low Pass Filter first
            var filterCutoff = 1.0f / downsampleFactor;
            var filterHalfWidth = 64;
            var timeAxis = arange(-filterHalfWidth, filterHalfWidth + 1, device: audioSignal.device);
            var filterKernel = sinc(filterCutoff * timeAxis) * filterCutoff;
            filterKernel *= hamming_window((2 * filterHalfWidth) + 1, device: audioSignal.device);
            filterKernel /= filterKernel.sum();

            filterKernel = filterKernel.reshape(1, 1, -1);
            var paddingSize = filterKernel.size(-1) - 1;
            audioSignal = nn.functional.pad(audioSignal, [paddingSize, paddingSize],
                mode: PaddingModes.Reflect);
            audioSignal = nn.functional.conv1d(audioSignal, filterKernel);

            // Downsample by taking every downsampleFactor'th sample
            audioSignal = audioSignal.index(
                TensorIndex.Slice(),
                TensorIndex.Slice(),
                arange(0, audioSignal.size(-1), downsampleFactor, device: audioSignal.device)
            );
        }

        _audioData = audioSignal;
        SampleRate = targetSampleRate;
    }

    public void Reset()
    {
        _stftData?.Dispose();
        _stftData = null;
    }

    /// <summary>
    /// Performs Short-Time Fourier Transform on the audio data.
    /// </summary>
    /// <param name="windowLength">Length of the STFT window</param>
    /// <param name="hopLength">Number of samples between successive frames</param>
    /// <param name="windowType">Type of window function to use</param>
    /// <param name="matchStride">Whether to adjust padding to match stride</param>
    /// <param name="paddingType">Type of padding to apply</param>
    /// <returns>Complex STFT tensor</returns>
    public Tensor STFT(
        int? windowLength = null,
        int? hopLength = null,
        string windowType = null,
        bool? matchStride = null,
        PaddingModes? paddingType = null)
    {
        windowLength ??= _stftParams.WindowLength;
        hopLength ??= _stftParams.HopLength;
        windowType ??= _stftParams.WindowType;
        matchStride ??= _stftParams.MatchStride;
        paddingType ??= _stftParams.PaddingMode;

        var window = GetWindow(windowType, windowLength.Value, _audioData.device.type.ToString()); // TODO: device conversion
        window = window.to(_audioData.device);

        var audioData = _audioData;
        var (rightPad, pad) = ComputeStftPadding(windowLength.Value, hopLength.Value, matchStride.Value);

        // Apply padding
        audioData = nn.functional.pad(
            audioData,
            new long[] { pad, pad + rightPad },
            mode: paddingType!.Value); // TODO: padding mode conversion

        // Reshape for STFT
        var flatAudio = audioData.reshape(-1, audioData.size(-1));

        // Perform STFT
        var stftData = stft(
            flatAudio,
            n_fft: windowLength.Value,
            hop_length: hopLength.Value,
            window: window,
            return_complex: true,
            center: true
        );

        var (_, nf, nt) = (stftData.size(0), stftData.size(1), stftData.size(2));
        stftData = stftData.reshape(BatchSize, NumChannels, nf, nt);

        if (matchStride.Value)
        {
            // Drop first two and last two frames
            stftData = stftData.slice(3, 2, -2, 1);
        }

        _stftData = stftData;
        return stftData;
    }

    public AudioSignal To(string device)
    {
        if (_stftData is not null)
        {
            _stftData = _stftData.to(device);
        }
        if (_audioData is not null)
        {
            _audioData = _audioData.to(device);
        }
        return this;
    }

    /// <summary>
    /// Converts stereo audio to mono by averaging channels
    /// </summary>
    /// <remarks>
    /// Note: AudioSignal does not currently handle interleaving.
    /// </remarks>
    public void ToMono()
    {
        if (NumChannels == 1)
        {
            return;
        }
        _audioData = DSP.ConvertToMono(_audioData);
    }
    public override string ToString()
    {
        var info = new Dictionary<string, object>
            {
                {"duration", $"{SignalDuration:0.3f} seconds"},
                {"batch_size", BatchSize},
                {"path", PathToFile ?? "path unknown"},
                {"sample_rate", SampleRate},
                {"num_channels", NumChannels},
                {"audio_data.shape", string.Join("x", _audioData.shape)},
                {"stft_params", _stftParams},
                {"device", Device}
            };

        return string.Join("\n", info.Select(kvp => $"{kvp.Key}: {kvp.Value}"));
    }

    /// <summary>
    /// Trims samples from the beginning and/or end of the signal
    /// </summary>
    /// <param name="before">Number of samples to trim from start</param>
    /// <param name="after">Number of samples to trim from end</param>
    public void Trim(int before, int after)
    {
        if (after == 0)
        {
            _audioData = _audioData[.., before..];
        }
        else
        {
            _audioData = _audioData[.., before..^after];
        }
    }

    public void TruncateSamples(int lengthInSamples)
    {
        _audioData = _audioData[.., ..lengthInSamples];
    }

    // Additional utility methods
    public void ZeroPad(int before, int after)
    {
        _audioData = nn.functional.pad(_audioData, (before, after));
    }

    public void ZeroPadTo(int length, string mode = "after")
    {
        if (mode == "before")
        {
            ZeroPad(Math.Max(length - SignalLength, 0), 0);
        }
        else
        {
            ZeroPad(0, Math.Max(length - SignalLength, 0));
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _audioData?.Dispose();
            }

            _stftData?.Dispose();
            _loudness?.Dispose();
        }
        _disposed = true;
    }

    private static AudioInfo GetAudioInfo(string audioPath)
    {
        using var reader = new AudioFileReader(audioPath);
        return new AudioInfo
        {
            Duration = (float)reader.TotalTime.TotalSeconds,
            NumFrames = (int)(reader.Length / (reader.WaveFormat.BitsPerSample / 8)),
            SampleRate = reader.WaveFormat.SampleRate
        };
    }

    private static Tensor MelFilterbank(
            int nMels,
        int nFft,
        int sampleRate,
        float fMin,
        float fMax)
    {
        // Convert Hz to mel scale
        float HzToMel(float hz) =>
            2595 * (float)Math.Log10(1 + (hz / 700.0f));

        float MelToHz(float mel) =>
            700 * ((float)Math.Pow(10, mel / 2595.0f) - 1);

        var minMel = HzToMel(fMin);
        var maxMel = HzToMel(fMax);

        // Create equally spaced points in mel scale
        var melPoints = linspace(minMel, maxMel, nMels + 2);
        var hzPoints = tensor(melPoints.data<float>().Select(mel => MelToHz(mel)).ToArray());

        // Convert to FFT bins
        var bins = (nFft - 1) * hzPoints / sampleRate;

        // Create filterbank matrix
        var fbank = zeros(new long[] { nMels, nFft });

        for (int i = 0; i < nMels; i++)
        {
            var left = bins[i];
            var center = bins[i + 1];
            var right = bins[i + 2];

            for (int j = 0; j < nFft; j++)
            {
                float weight = 0;

                if (j > left.item<float>() && j < right.item<float>())
                {
                    if (j <= center.item<float>())
                    {
                        weight = (j - left.item<float>()) / (center.item<float>() - left.item<float>());
                    }
                    else
                    {
                        weight = (right.item<float>() - j) / (right.item<float>() - center.item<float>());
                    }
                }

                fbank[i, j] = weight;
            }
        }

        return fbank;
    }

    private static int NextPowerOfTwo(int x)
    {
        return (int)Math.Pow(2, Math.Ceiling(Math.Log2(x)));
    }

    private Tensor DCTMatrix(int numMfcc, int numMels)
    {
        var cacheKey = (numMfcc, numMels);
        if (!_dctMatrixCache.ContainsKey(cacheKey))
        {
            var melIndices = arange(numMels, device: Device);
            var mfccIndices = arange(numMfcc, device: Device).unsqueeze(-1);
            var dctMatrix = cos(mfccIndices * ((2 * melIndices) + 1) * Math.PI / (2 * numMels));

            // Normalization
            dctMatrix *= sqrt(2.0f / numMels);
            dctMatrix.select(0, 0).mul_(1.0f / sqrt(2.0f));

            _dctMatrixCache[cacheKey] = dctMatrix;
        }
        return _dctMatrixCache[cacheKey];
    }

    private float[,] MelFilterbankArray(int sampleRate, int nfftSize, int numMels, float freqMin, float freqMax)
    {
        var cacheKey = (sampleRate, nfftSize, numMels, freqMin, freqMax);
        if (!_melFilterbankCache.TryGetValue(cacheKey, out float[,]? cachedValue))
        {
            // Create mel filterbank using librosa-style triangular filters
            float[] fftFrequencies = Enumerable.Range(0, (nfftSize / 2) + 1)
                .Select(i => i * sampleRate / (float)nfftSize)
                .ToArray();

            // Convert Hz to mel
            float melMinimum = 1127 * (float)Math.Log(1 + (freqMin / 700));
            float melMaximum = 1127 * (float)Math.Log(1 + (freqMax / 700));

            // Create mel points evenly spaced in mel scale
            float[] melPoints = Enumerable.Range(0, numMels + 2)
                .Select(i => melMinimum + (i * (melMaximum - melMinimum) / (numMels + 1)))
                .ToArray();

            // Convert mel points back to Hz
            float[] hzPoints = melPoints
                .Select(melPoint => 700 * ((float)Math.Exp(melPoint / 1127) - 1))
                .ToArray();

            var filterbank = new float[numMels, (nfftSize / 2) + 1];

            for (int melBand = 0; melBand < numMels; melBand++)
            {
                float freqLeft = hzPoints[melBand];
                float freqCenter = hzPoints[melBand + 1];
                float freqRight = hzPoints[melBand + 2];

                for (int freqBin = 0; freqBin < fftFrequencies.Length; freqBin++)
                {
                    float frequency = fftFrequencies[freqBin];

                    // Left slope
                    if (frequency > freqLeft && frequency < freqCenter)
                    {
                        filterbank[melBand, freqBin] = (frequency - freqLeft) / (freqCenter - freqLeft);
                    }
                    // Right slope
                    else if (frequency >= freqCenter && frequency < freqRight)
                    {
                        filterbank[melBand, freqBin] = (freqRight - frequency) / (freqRight - freqCenter);
                    }
                }
            }

            cachedValue = filterbank;
            _melFilterbankCache[cacheKey] = cachedValue;
        }

        return cachedValue;
    }

    #endregion Methods
}