using Microsoft.Win32;
using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Loading
{
    public class ModelRegistry
    {
        //private readonly Dictionary<string, IModelFactory> _factories = new();
        private readonly Dictionary<Type, Func<IModelConfig, INeuralCodec>> _factories = new();
        //public void RegisterFactory(string architecture, IModelFactory factory)
        //{
        //    _factories[architecture] = factory;
        //}

        //public INeuralCodec CreateModel(IModelConfig config)
        //{
        //    if (_factories.TryGetValue(config.Architecture, out var factory))
        //    {
        //        return factory.CreateModel(config);
        //    }
        //    throw new NotSupportedException($"No factory registered for {config.Architecture}");
        //}
        //public TModel CreateModel<TModel>(IModelConfig config) where TModel : class, INeuralCodec
        //{
        //    if (_factories.TryGetValue(config.Architecture, out var factory))
        //    {
        //        return factory.CreateModel(config) as TModel;
        //    }
        //    throw new NotSupportedException($"No factory registered for {config.Architecture}");
        //}
        public void RegisterModel<TModel, TConfig>(Func<TConfig, TModel> factory)
            where TModel : class, INeuralCodec
            where TConfig : class, IModelConfig
        {
            //var configType = typeof(TConfig);
            var modelType = typeof(TModel);

            // Store the factory with type conversion
            _factories[modelType] = (config) => factory((TConfig)config);
        }
        public TModel CreateModel<TModel, TConfig>(TConfig config) 
            where TModel : class, INeuralCodec
            where TConfig : class, IModelConfig
        {
            var mType = typeof(TModel);

            if (!_factories.TryGetValue(mType, out var factory))
            {
                throw new ModelLoadException(
                    $"No factory registered for type: {mType.Name}");
            }

            return (TModel)factory(config);
        }

    }
}