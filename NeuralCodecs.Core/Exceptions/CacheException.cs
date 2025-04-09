using NeuralCodecs.Core.Operations;
using System.Runtime.Serialization;

namespace NeuralCodecs.Core.Exceptions
{


        /// <summary>
        /// Exception thrown when a cache operation fails
        /// </summary>
        public class CacheException : NeuralCodecException
        {
            /// <summary>
            /// Gets the path to the cache
            /// </summary>
            public string? CachePath { get; }

            /// <summary>
            /// Gets the model ID
            /// </summary>
            public string? ModelId { get; }

            /// <summary>
            /// Creates a new cache exception
            /// </summary>
            public CacheException() : base() { }

            /// <summary>
            /// Creates a new cache exception with a message
            /// </summary>
            public CacheException(string message) : base(message) { }

            /// <summary>
            /// Creates a new cache exception with a message and inner exception
            /// </summary>
            public CacheException(string message, Exception innerException)
                : base(message, innerException) { }

            /// <summary>
            /// Creates a new cache exception with context
            /// </summary>
            public CacheException(string message, string cachePath, string? modelId = null)
                : base(message)
            {
                CachePath = cachePath;
                ModelId = modelId;

                if (cachePath != null) WithDiagnostic("CachePath", cachePath);
                if (modelId != null) WithDiagnostic("ModelId", modelId);
            }

            /// <summary>
            /// Creates a new cache exception with context and inner exception
            /// </summary>
            public CacheException(string message, string cachePath, Exception innerException, string? modelId = null)
                : base(message, innerException)
            {
                CachePath = cachePath;
                ModelId = modelId;

                if (cachePath != null) WithDiagnostic("CachePath", cachePath);
                if (modelId != null) WithDiagnostic("ModelId", modelId);
            }

        }
    }

