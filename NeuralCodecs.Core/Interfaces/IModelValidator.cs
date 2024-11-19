

using NeuralCodecs.Core.Loading;

namespace NeuralCodecs.Core.Interfaces
{

    public interface IModelValidator<T> where T : IModelConfig
    {
        // Config validation
        ValidationResult ValidateConfig(T config);

        // Runtime model validation
        Task<ValidationResult> ValidateModel(INeuralCodec model, T config);
    }
}
