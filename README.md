# <img src="https://github.com/DillionLowry/NeuralCodecs/blob/main/nc_logo.png" width="50" height="50">  NeuralCodecs [![NuGet Version](https://img.shields.io/nuget/v/NeuralCodecs?style=flat)](https://www.nuget.org/packages/NeuralCodecs)

NeuralCodecs is a .NET library for neural audio codec implementations, designed for efficient audio compression and reconstruction.

## Features
- **SNAC**: [Multi-**S**cale **N**eural **A**udio **C**odec](https://github.com/hubertsiuzdak/snac)
  - Support for multiple sampling rates: 24kHz, 32kHz, and 44.1kHz
  - Attention mechanisms with adjustable window sizes for improved quality
  - Automatic resampling for input flexibility
- **DAC**: [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
  - Supports multiple sampling rates: 16kHz, 24kHz, and 44.1kHz
  - Configurable encoder/decoder architecture with variable rates
  - Flexible bitrate configurations from 8kbps to 16kbps
- **Encodec**: [Meta's Encodec neural audio compression](https://github.com/facebookresearch/encodec)
  - Supports stereo audio at 24kHz and 48kHz sample rates
  - Variable bitrate compression (1.5-24 kbps)
  - Neural language model for enhanced compression quality
  - Direct file compression to .ecdc format
- **AudioTools**: Advanced audio processing utilities
  - Based on Descript's [audiotools](https://github.com/descriptinc/audiotools) Python package
  - Extended with .NET-specific optimizations and additional features
  - Audio filtering, transformation, and effects processing
  - Works with Descript's AudioSignal or Tensors
- **Audio Visualization**: Example project includes spectrogram generation and comparison tools

## Requirements
- .NET 8.0 or later
- TorchSharp or libTorch compatible with your platform
- NAudio (for audio processing)
- SkiaSharp (for visualization features)

## Usage

### Creating/loading the model

There are several ways to load a model:

1. #### Using static factory method:
```csharp
// Load SNAC model with static method provided for built-in models
var model = await NeuralCodecs.CreateSNACAsync("model.pt");
```

2. #### Using  premade config:
    SnacConfig provides premade configurations for 24kHz, 32kHz, and 44kHz sampling rates.
```csharp
var model = await NeuralCodecs.CreateSNACAsync(modelPath, SNACConfig.SNAC24Khz);
```

3. #### Using IModelLoader instance with default config:
    Allows the use of custom loader implementations
```csharp
// Load model with default config from IModelLoader instance
var torchLoader = NeuralCodecs.CreateTorchLoader();
var model = await torchLoader.LoadModelAsync<SNAC, SNACConfig>("model.pt");
```

4. #### Using IModelLoader instance with custom config:
```csharp
// For Encodec with custom bandwidth and settings
var encodecConfig = new EncodecConfig { 
    SampleRate = 48000,
    Bandwidth = 12.0f,
    Channels = 2,  // Stereo audio
    Normalize = true
};
var encodecModel = await torchLoader.LoadModelAsync<Encodec, EncodecConfig>("encodec_model.pt", encodecConfig);
```

5. #### Using factory method for custom models:
      Allows the use of custom model implementations with built-in or custom loaders
```csharp
// Load custom model with factory method
var model = await torchLoader.LoadModelAsync<CustomModel, CustomConfig>(
    "model.pt",
    config => new CustomModel(config, ...),
    config);
```

Models can be loaded in Pytorch or Safetensors format.

### AudioTools Features

The AudioTools namespace provides extensive audio processing capabilities:

```csharp
var audio = new Tensor(...); // Load or create audio tensor

// Apply effects
var processedAudio = AudioEffects.ApplyCompressor(
    audio, 
    sampleRate: 48000,
    threshold: -20f,
    ratio: 4.0f);

// Compute spectrograms and transforms
var spectrogram = DSP.MelSpectrogram(audio, sampleRate);
var stft = DSP.STFT(audio, windowSize: 1024, hopSize: 512, windowType: "hann");
```

### Encoding and Decoding Audio

There are two main ways to process audio:

1. Using the simplified ProcessAudio method:
```csharp
// Compress audio in one step
var processedAudio = model.ProcessAudio(audioData, sampleRate);
```

2. Using separate encode and decode steps:
```csharp
// Encode audio to compressed format
var codes = model.Encode(buffer);

// Decode back to audio
var processedAudio = model.Decode(codes);
```

3. Saving the processed audio
    
    Use your preferred method to save WAV files
```csharp
// using NAudio
await using var writer = new WaveFileWriter(
    outputPath,
    new WaveFormat(model.Config.SamplingRate, channels: model.Channels)
);
writer.WriteSamples(processedAudio, 0, processedAudio.Length);
```
### Encodec-Specific Features

Encodec provides additional capabilities:

```csharp
// Set target bandwidth for compression (supported values depend on model)
encodecModel.SetTargetBandwidth(12.0f); // 12 kbps

// Get available bandwidth options
var availableBandwidths = encodecModel.TargetBandwidths; // e.g. [1.5, 3, 6, 12, 24]

// Use language model for enhanced compression quality
var lm = await encodecModel.GetLanguageModel();
// Apply LM during encoding/decoding for better quality

// Direct file compression
await EncodecCompressor.CompressToFileAsync(encodecModel, audioTensor, "audio.ecdc", useLm: true);

// Decompress from file
var (decompressedAudio, sampleRate) = await EncodecCompressor.DecompressFromFileAsync("audio.ecdc");

```

## Example

Check out the Example project for a complete implementation, including:
- Model loading and configuration
- Audio processing workflows
- Command-line interface implementation
- Audio Visualization

The example includes tools for visualizing and comparing audio spectrograms:

DAC Codec 24kHz before and after compression
<img src="Docs/Images/spectrogram_DAC_24k.png" width="500" height="300">

## Acknowledgments
- [SNAC](https://github.com/hubertsiuzdak/snac) - Original SNAC implementation
- [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) - DAC reference
- [Encodec](https://github.com/facebookresearch/encodec) - Meta's neural audio codec

## Contributing
Suggestions and contributions are welcome! Feel free to submit a pull request.

## License
This project is licensed under the MIT License.
