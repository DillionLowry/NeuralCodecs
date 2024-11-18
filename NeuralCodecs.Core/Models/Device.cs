namespace NeuralCodecs.Core.Models //TODO THIS NEEDS WORK
{
    /// <summary>
    /// Generic device abstraction
    /// </summary>
    public abstract class Device
    {
        public abstract string Type { get; }
        public abstract int Index { get; }
    }
}