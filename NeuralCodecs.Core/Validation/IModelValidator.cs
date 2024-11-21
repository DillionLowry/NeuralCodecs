using NeuralCodecs.Core.Configuration;

namespace NeuralCodecs.Core.Validation
{
    /// <summary>
    /// Provides validation functionality for neural codec models and their configurations.
    /// </summary>
    /// <typeparam name="T">The type of model configuration to validate, must implement IModelConfig.</typeparam>
    public interface IModelValidator<T> where T : IModelConfig
    {
        /// <summary>
        /// Validates the provided model configuration.
        /// </summary>
        /// <param name="config">The configuration to validate.</param>
        /// <returns>A ValidationResult indicating whether the configuration is valid.</returns>
        ValidationResult ValidateConfig(T config);

        /// <summary>
        /// Validates a neural codec model against its configuration.
        /// </summary>
        /// <param name="model">The neural codec model to validate.</param>
        /// <param name="config">The configuration to validate against.</param>
        /// <returns>A ValidationResult indicating whether the model is valid.</returns>
        Task<ValidationResult> ValidateModel(INeuralCodec model, T config);
    }
}