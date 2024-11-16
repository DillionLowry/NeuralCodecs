namespace NeuralCodecs.Core.Interfaces
{
    internal interface IModelProvider : IDisposable
    {
        string ModelPath { get; }
        bool IsLoaded { get; }

        ValueTask LoadModelAsync(string path, CancellationToken ct = default);

        ValueTask UnloadModelAsync(CancellationToken ct = default);
    }
}