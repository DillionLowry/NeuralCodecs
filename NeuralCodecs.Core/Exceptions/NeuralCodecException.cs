using NeuralCodecs.Core.Operations;
using System.Runtime.Serialization;

namespace NeuralCodecs.Core.Exceptions
{
    /// <summary>
        /// Base exception class for all NeuralCodecs exceptions
        /// </summary>
        [Serializable]
        public class NeuralCodecException : Exception
        {
            /// <summary>
            /// Gets additional diagnostic information about the exception
            /// </summary>
            public IDictionary<string, string> Diagnostics { get; } = new Dictionary<string, string>();

            /// <summary>
            /// Creates a new codec exception
            /// </summary>
            public NeuralCodecException() : base() { }

            /// <summary>
            /// Creates a new codec exception with a message
            /// </summary>
            public NeuralCodecException(string message) : base(message) { }

            /// <summary>
            /// Creates a new codec exception with a message and inner exception
            /// </summary>
            public NeuralCodecException(string? message, Exception? innerException)
                : base(message, innerException) { }


            /// <summary>
            /// Adds diagnostic information to the exception
            /// </summary>
            public NeuralCodecException WithDiagnostic(string key, string value)
            {
                Diagnostics[key] = value;
                return this;
            }

            /// <summary>
            /// Adds multiple diagnostic information items to the exception
            /// </summary>
            public NeuralCodecException WithDiagnostics(IDictionary<string, string> diagnostics)
            {
                foreach (var kvp in diagnostics)
                {
                    Diagnostics[kvp.Key] = kvp.Value;
                }
                return this;
            }

            /// <summary>
            /// Gets the exception message with diagnostics
            /// </summary>
            public override string ToString()
            {
                if (Diagnostics.Count == 0)
                {
                    return base.ToString();
                }

                var baseMessage = base.ToString();
                var diagnosticsMessage = string.Join(Environment.NewLine,
                    Array.ConvertAll(
                        new Dictionary<string, string>(Diagnostics).ToArray(),
                        kvp => $"- {kvp.Key}: {kvp.Value}"));

                return $"{baseMessage}{Environment.NewLine}Diagnostics:{Environment.NewLine}{diagnosticsMessage}";
            }
        }
}