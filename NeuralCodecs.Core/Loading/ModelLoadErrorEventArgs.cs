namespace NeuralCodecs.Core.Loading
{
    public class ModelLoadErrorEventArgs : EventArgs
    {
        public string Source { get; }
        public Exception Error { get; }

        public ModelLoadErrorEventArgs(string source, Exception error)
        {
            Source = source;
            Error = error;
        }
    }
}