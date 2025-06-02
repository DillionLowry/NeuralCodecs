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
            var modelType = typeof(TModel);

            if (_factories.TryGetValue(modelType, out var factory))
            {
                return (TModel)factory(config);
            }

            try
            {
                // // If no factory registered, attempt default factory pattern TModel(TConfig config)
                var constructorParams = new object[] { config };

                if (Activator.CreateInstance(modelType, constructorParams) is TModel model)
                {
                    // Cache the successful constructor pattern for future use
                    _factories[modelType] = (c) => (TModel)Activator.CreateInstance(modelType, c)!;
                    return model;
                }

                throw new LoadException(
                    $"Failed to create instance of {modelType.Name} using default constructor pattern");
            }
            catch (MissingMethodException ex)
            {
                throw new LoadException(
                    $"No factory registered for {modelType.Name} and no suitable constructor found. " +
                    $"Expected constructor: {modelType.Name}({typeof(TConfig).Name} config)", ex);
            }
            catch (Exception ex) when (
                ex is not LoadException &&
                ex is not ArgumentException)
            {
                throw new LoadException(
                    $"Failed to create instance of {modelType.Name} using default constructor pattern", ex);
            }
        }

        /// <summary>
        /// Creates a model instance using the registered factory for the specified type.
        /// </summary>
        /// <typeparam name="TModel">The type of model to create.</typeparam>
        /// <param name="config">The configuration to use for model creation.</param>
        /// <returns>A new instance of the specified model type.</returns>
        /// <exception cref="LoadException">Thrown when no factory is registered for the specified model type.</exception>
        public TModel CreateModel<TModel>()
            where TModel : class, INeuralCodec
        {
            var mType = typeof(TModel);

            if (!_factories.TryGetValue(mType, out var factory))
            {
                throw new LoadException(
                    $"No factory registered for type: {mType.Name}");
            }

            return (TModel)factory(null);
        }
    }
}