namespace NeuralCodecs.Core.Loading
{
    public class ModelLoadWarningEventArgs : EventArgs
    {
        public string Message { get; }

        public ModelLoadWarningEventArgs(string message)
        {
            Message = message;
        }
    }
}