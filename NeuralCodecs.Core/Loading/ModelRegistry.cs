using NeuralCodecs.Core.Configuration;
using NeuralCodecs.Core.Exceptions;

namespace NeuralCodecs.Core.Loading
{
    /// <summary>
    /// Registry for managing neural codec model factories and creation.
    /// </summary>
    public class ModelRegistry
    {
        private readonly Dictionary<Type, Func<IModelConfig, INeuralCodec>> _factories = new();

        /// <summary>
        /// Registers a factory function for creating models of a specific type.
        /// </summary>
        /// <typeparam name="TModel">The type of model to register.</typeparam>
        /// <typeparam name="TConfig">The type of configuration required by the model.</typeparam>
        /// <param name="factory">The factory function that creates model instances.</param>
        public void RegisterModel<TModel, TConfig>(Func<TConfig, TModel> factory)
            where TModel : class, INeuralCodec
            where TConfig : class, IModelConfig
        {
            var modelType = typeof(TModel);

            // Store the factory with type conversion
            _factories[modelType] = (config) => factory((TConfig)config);
        }

        /// <summary>
        /// Creates a model instance using the registered factory for the specified type.
        /// </summary>
        /// <typeparam name="TModel">The type of model to create.</typeparam>
        /// <typeparam name="TConfig">The type of configuration required by the model.</typeparam>
        /// <param name="config">The configuration to use for model creation.</param>
        /// <returns>A new instance of the specified model type.</returns>
        /// <exception cref="LoadException">Thrown when no factory is registered for the specified model type.</exception>
        public TModel CreateModel<TModel, TConfig>(TConfig config)
            where TModel : class, INeuralCodec
            where TConfig : class, IModelConfig
        {
            var mType = typeof(TModel);

            if (!_factories.TryGetValue(mType, out var factory))
            {
                throw new LoadException(
                    $"No factory registered for type: {mType.Name}");
            }

            return (TModel)factory(config);
        }
    }
}