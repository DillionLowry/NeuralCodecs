using System.Text.Json;
using System.Text.Json.Serialization;

namespace NeuralCodecs.Core.Configuration
{
    /// <summary>
    /// Provides JSON conversion functionality for model configuration classes that implement IModelConfig.
    /// </summary>
    /// <typeparam name="TModelConfig">The type of model configuration to convert.</typeparam>
    public class ModelConfigJsonConverter<TModelConfig> : JsonConverter<TModelConfig> where TModelConfig : IModelConfig
    {
        /// <summary>
        /// Reads and converts the JSON to an instance of TModelConfig.
        /// </summary>
        /// <param name="reader">The reader to read JSON from.</param>
        /// <param name="typeToConvert">The type of object to convert.</param>
        /// <param name="options">Options to control the behavior during parsing.</param>
        /// <returns>The converted TModelConfig value.</returns>
        public override TModelConfig? Read(
            ref Utf8JsonReader reader,
            Type typeToConvert,
            JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
            {
                throw new JsonException("Expected start of object");
            }

            using var jsonDoc = JsonDocument.ParseValue(ref reader);
            var root = jsonDoc.RootElement;

            // Create new options without this converter to avoid infinite recursion
            var newOptions = new JsonSerializerOptions(options);
            newOptions.Converters.Remove(this);

            // Deserialize to the concrete type
            return JsonSerializer.Deserialize<TModelConfig>(
                root.GetRawText(),
                newOptions);
        }

        /// <summary>
        /// Writes a specified value as JSON.
        /// </summary>
        /// <param name="writer">The writer to write JSON to.</param>
        /// <param name="value">The value to convert to JSON.</param>
        /// <param name="options">Options to control serialization behavior.</param>
        public override void Write(
            Utf8JsonWriter writer,
            TModelConfig value,
            JsonSerializerOptions options)
        {
            JsonSerializer.Serialize(writer, value, options);
        }
    }
}