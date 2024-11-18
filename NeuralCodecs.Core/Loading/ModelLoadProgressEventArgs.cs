namespace NeuralCodecs.Core.Loading
{
    public class ModelLoadProgressEventArgs : EventArgs
    {
        public string Source { get; }
        public double Progress { get; }

        public ModelLoadProgressEventArgs(string source, double progress)
        {
            Source = source;
            Progress = progress;
        }
    }
}