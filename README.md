# <img src="https://github.com/DillionLowry/NeuralCodecs/blob/main/nc_logo.png" width="50" height="50">  NeuralCodecs [![NuGet Version](https://img.shields.io/nuget/v/NeuralCodecs?style=flat)](https://www.nuget.org/packages/NeuralCodecs)

NeuralCodecs is a .NET library for neural audio codec implementations, designed for efficient audio compression and reconstruction.

## Features
- **SNAC**: [Multi-**S**cale **N**eural **A**udio **C**odec](https://github.com/hubertsiuzdak/snac)

## Work In Progress
- **DAC**: [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)

## Requirements
- Torchsharp or libTorch targeting your desired platform

## Usage

### Creating/loading the model

There are several ways to load a model:

1. Using static factory method:
```csharp
// Load SNAC model with static method provided for built-in models
var model = await NeuralCodecs.CreateSNACAsync("model.pt");
```

2. #### Using  premade config:
    SnacConfig provides premade configurations for 24kHz, 32kHz, and 44kHz sampling rates.
```csharp
var model = await NeuralCodecs.CreateSNACAsync(modelPath, SNACConfig.SNAC24Khz);
```

3. #### Using IModelLoader instance with default config 
    Allows the use of custom loader implementations
```csharp
// Load model with default config from IModelLoader instance
var torchLoader = NeuralCodecs.CreateTorchLoader();
var model = await torchLoader.LoadModelAsync<SNAC, SNACConfig>("model.pt");
```

4. #### Using IModelLoader instance with custom config:
```csharp
var config = new SNACConfig { /* ... */ };
var model = await torchLoader.LoadModelAsync<SNAC, SNACConfig>("model.pt", config);
```

5. #### Using factory method for custom models:
      Allows the use of custom model implementations with built-in loaders
```csharp
// Load custom model with factory method
var model = await torchLoader.LoadModelAsync<CustomModel, CustomConfig>(
    "model.pt",
    config => new CustomModel(config, ...),
    config);
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
    new WaveFormat(model.Config.SamplingRate, 1)
);
writer.WriteSamples(processedAudio, 0, processedAudio.Length);
```

## Acknowledgments
- [SNAC](https://github.com/hubertsiuzdak/snac) - Original SNAC implementation
- [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) - DAC reference

## Contributing
Suggestions and contributions are welcome! Feel free to submit a pull request.

## License
This project is licensed under the MIT License.
