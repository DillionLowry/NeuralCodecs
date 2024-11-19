using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Loading;
using NeuralCodecs.Core.Models;
using System.Reflection;
using System.Text.Json;

namespace NeuralCodecs.Torch
{
    /// <summary>
    /// Primary interface for working with neural audio codecs
    /// </summary>
    ///
    public static partial class NeuralCodecs
    {
        //private readonly IModelLoader _modelLoader;

        //private ModelLoadOptions _options;

        //public NeuralCodecs(IModelLoader modelLoader)
        //{
        //    _modelLoader = modelLoader ?? throw new ArgumentNullException(nameof(modelLoader));
        //}
        static NeuralCodecs() { }
        ////public void RegisterModelFactory(string architecture, IModelFactory factory)
        ////{
        ////    _registry.RegisterFactory(architecture, factory);
        ////}

        //public static async Task<ModelOperationResult<TModel>> LoadModelAsync<TModel, TConfig>(
        //    string source,
        //    IModelLoader loader,
        //    TConfig? config = default,
        //    ModelLoadOptions? options = default)
        //    where TModel : class, INeuralCodec
        //    where TConfig : class, IModelConfig
        //{
        //    try
        //    {
        //        options ??= new ModelLoadOptions();

        //        var model = await loader.LoadModelAsync<TModel, TConfig>(source, config, options);
        //        return ModelOperationResult<TModel>.FromSuccess(model);
        //    }
        //    catch (Exception ex)
        //    {
        //        return ModelOperationResult<TModel>.FromError(ex);
        //    }
        //}
        //public ModelOperationResult<TModel> CreateModel<TModel, TConfig>(TConfig config)
        //    where TModel : class, INeuralCodec
        //    where TConfig : class, IModelConfig
        //{
        //    try
        //    {
        //        var model = _registry.CreateModel<TModel, TConfig>(config);
        //        return ModelOperationResult<TModel>.FromSuccess(model);
        //    }
        //    catch (Exception ex)
        //    {
        //        return ModelOperationResult<TModel>.FromError(ex);
        //    }
        //}
    //    public async Task<ModelOperationResult<ModelInfo>> GetModelInfo(string source)
    //    {
    //        try
    //        {
    //            var info = await _modelLoader.GetModelInfo(source);
    //            return ModelOperationResult<ModelInfo>.FromSuccess(info);
    //        }
    //        catch (Exception ex)
    //        {
    //            return ModelOperationResult<ModelInfo>.FromError(ex);
    //        }
    //    }

    //    public void ClearCache(string modelId = null)
    //    {
    //        _modelLoader.ClearCache(modelId);
    //    }

    //    public string GetDefaultCacheDirectory() => _modelLoader.GetDefaultCacheDirectory();
    }
}