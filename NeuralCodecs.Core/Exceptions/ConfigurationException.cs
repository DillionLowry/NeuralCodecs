namespace NeuralCodecs.Core.Exceptions
{
    /// <summary>
    /// Exception thrown when configuration validation or processing fails.
    /// </summary>
    public class ConfigurationException : NeuralCodecException
    {
        /// <summary>
        /// Gets the name of the config property that caused the error
        /// </summary>
        public string? PropertyName { get; }

        /// <summary>
        /// Gets the value of the config property that caused the error
        /// </summary>
        public string? PropertyValue { get; }

        /// <summary>
        /// Gets the type of the configuration
        /// </summary>
        public string? ConfigType { get; }

        /// <summary>
        /// Creates a new configuration exception
        /// </summary>
        public ConfigurationException(string message) : base(message) { }

        /// <summary>
        /// Creates a new configuration exception with a message and inner exception
        /// </summary>
        public ConfigurationException(string message, Exception innerException)
            : base(message, innerException) { }

        /// <summary>
        /// Creates a new configuration exception with context
        /// </summary>
        public ConfigurationException(string message, string? propertyName,
            string? propertyValue = null, string? configType = null)
            : base(message)
        {
            PropertyName = propertyName;
            PropertyValue = propertyValue;
            ConfigType = configType;

            if (propertyName != null) WithDiagnostic("PropertyName", propertyName);
            if (propertyValue != null) WithDiagnostic("PropertyValue", propertyValue);
            if (configType != null) WithDiagnostic("ConfigType", configType);
        }

        /// <summary>
        /// Creates a new configuration exception with context and inner exception
        /// </summary>
        public ConfigurationException(string message, Exception innerException,
            string? propertyName, string? propertyValue = null, string? configType = null)
            : base(message, innerException)
        {
            PropertyName = propertyName;
            PropertyValue = propertyValue;
            ConfigType = configType;

            if (propertyName != null) WithDiagnostic("PropertyName", propertyName);
            if (propertyValue != null) WithDiagnostic("PropertyValue", propertyValue);
            if (configType != null) WithDiagnostic("ConfigType", configType);
        }
    }
}