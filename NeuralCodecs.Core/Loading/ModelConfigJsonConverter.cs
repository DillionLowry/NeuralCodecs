using NeuralCodecs.Core.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Text.Json;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Loading
{
    public class ModelConfigJsonConverter<TModelConfig> : JsonConverter<TModelConfig> where TModelConfig : IModelConfig
    {
        public override TModelConfig Read(
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

        public override void Write(
            Utf8JsonWriter writer,
            TModelConfig value,
            JsonSerializerOptions options)
        {
            JsonSerializer.Serialize<TModelConfig>(writer, value, options);
        }
    }
}
