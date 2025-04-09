using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Modules.Encodec;
/// <summary>
/// Represents an encoded audio frame with quantized codes and optional scale factor.
/// </summary>
/// <param name="Codes">The quantized codes representing the audio content.</param>
/// <param name="Scale">Optional scale factor applied during normalization.</param>
public record EncodedFrame(Tensor Codes, Tensor? Scale);