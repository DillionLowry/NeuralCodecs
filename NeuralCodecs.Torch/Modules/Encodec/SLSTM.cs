using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NeuralCodecs.Torch.Modules.Encodec;

/// <summary>
/// LSTM implementation that operates on convolutional layout tensors.
/// Handles the permutation between convolutional and sequence layouts transparently.
/// </summary>
public class SLSTM : Module<Tensor, Tensor>
{
    private readonly int _dimension;
    private readonly int _numLayers;
    private readonly bool _skip;
    private readonly LSTM lstm;

    /// <summary>
    /// Initialize SLSTM with the given parameters
    /// </summary>
    /// <param name="dimension">Hidden dimension size</param>
    /// <param name="numLayers">Number of LSTM layers</param>
    /// <param name="skip">Whether to use skip connection</param>
    public SLSTM(int dimension, int numLayers = 2, bool skip = true) : base("SLSTM")
    {
        ValidateParameters(dimension, numLayers);

        _skip = skip;
        _dimension = dimension;
        _numLayers = numLayers;
        lstm = LSTM(dimension, dimension, numLayers);
        RegisterComponents();
    }

    /// <summary>
    /// Forward pass applying LSTM to the input
    /// </summary>
    /// <param name="x">Input tensor in convolutional layout [batch, channels, time]</param>
    /// <returns>Output tensor in convolutional layout [batch, channels, time]</returns>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();

        ValidateInputShape(x);

        // Permute from [B, C, T] to [T, B, C]
        var permuted = x.permute(2, 0, 1).contiguous();
        var (output, _, _) = lstm.forward(permuted);

        if (_skip)
        {
            output = output.add(permuted);
        }

        // Permute back to [B, C, T]
        return output.permute(1, 2, 0).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Forward pass with explicitly provided initial hidden state
    /// </summary>
    /// <param name="x">Input tensor in convolutional layout [batch, channels, time]</param>
    /// <param name="initialHidden">Tuple of (hiddenState, cellState) or null for zero initialization</param>
    /// <returns>Tuple of (output, hiddenState, cellState)</returns>
    public (Tensor output, (Tensor h, Tensor c) hiddenState) forward(
        Tensor x, (Tensor h, Tensor c)? initialHidden)
    {
        using var scope = NewDisposeScope();

        // Validate input dimensions
        ValidateInputShape(x);

        // Create initial hidden states with appropriate device
        var device = x.device;
        var batchSize = x.size(0);

        // Initialize hidden state if not provided
        var h0 = initialHidden?.h ?? zeros(_numLayers, batchSize, _dimension, device: device);
        var c0 = initialHidden?.c ?? zeros(_numLayers, batchSize, _dimension, device: device);

        // Permute from [B, C, T] (convolutional) to [T, B, C] (sequence) layout
        var permuted = x.permute(2, 0, 1).contiguous();

        // Apply LSTM
        var (output, hN, cN) = lstm.forward(permuted, (h0, c0));

        // Apply skip connection if needed
        if (_skip)
        {
            output = output.add(permuted);
        }

        // Permute back to [B, C, T] (convolutional) layout
        var result = output.permute(1, 2, 0).contiguous();
        return (result.MoveToOuterDisposeScope(),
            (hN.MoveToOuterDisposeScope(), cN.MoveToOuterDisposeScope()));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            lstm?.Dispose();
        }
        base.Dispose(disposing);
    }

    private static void ValidateParameters(int dimension, int numLayers)
    {
        if (dimension <= 0)
        {
            throw new ArgumentException($"Dimension must be positive, got {dimension}");
        }

        if (numLayers <= 0)
        {
            throw new ArgumentException($"Number of layers must be positive, got {numLayers}");
        }
    }

    private void ValidateInputShape(Tensor x)
    {
        if (x.dim() != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input tensor [B, C, T], got shape [{string.Join(", ", x.shape)}]");
        }
    }
}