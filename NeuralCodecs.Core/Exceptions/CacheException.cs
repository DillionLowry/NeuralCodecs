using System.Runtime.Serialization;

namespace NeuralCodecs.Core.Exceptions
{
    /// <summary>
    /// Exception thrown when cache operations fail.
    /// </summary>
    public class CacheException : Exception
    {
        public string? CachePath { get; }

        public CacheException() : base() { }

        public CacheException(string message) : base(message) { }

        public CacheException(string message, Exception innerException)
            : base(message, innerException) { }

        public CacheException(string message, string cachePath)
            : base(message)
        {
            CachePath = cachePath;
        }

        public CacheException(string message, string cachePath, Exception innerException)
            : base(message, innerException)
        {
            CachePath = cachePath;
        }
    }
}