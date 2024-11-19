# NeuralCodecs

NeuralCodecs is a .NET library for neural audio codec implementations, designed for efficient audio compression and reconstruction.

## Features

- **SNAC**: [Multi-**S**cale **N**eural **A**udio **C**odec](https://github.com/hubertsiuzdak/snac)

## Work In Progress
- **DAC**: [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)


## Usage
### Encoding and Decoding Audio

```
// Load SNAC model with static method provided for built-in models
var model = await NeuralCodecs.CreateSNACAsync("model.pt");

// Load model with default config from IModelLoader instance
var torchLoader = NeuralCodecs.CreateTorchLoader();
var model = await torchLoader.LoadModelAsync<SNAC, SNACConfig>("model.pt");

// Load model with custom config
var config = new SNACConfig { /* ... */ };
var model = await torchLoader.LoadModelAsync<SNAC, SNACConfig>("model.pt", config);

// or premade config
var model = await torchLoader.LoadModelAsync<SNAC, SNACConfig>(modelPath, SNACConfig.SNAC24Khz);

// Load custom model with factory method
var model = await torchLoader.LoadModelAsync<CustomModel, CustomConfig>(
    "model.pt",
    config => new CustomModel(config, device),
    config);
```
## Contributing
Contributions are welcome!

## License
This project is licensed under the MIT License.