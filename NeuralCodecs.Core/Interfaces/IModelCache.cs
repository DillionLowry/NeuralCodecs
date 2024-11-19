using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Interfaces
{
    public interface IModelCache
    {
        Task<string> GetCachedPath(string modelId, string revision);
        Task<string> CacheModel(string modelId, string sourcePath, string revision);
        void ClearCache(string modelId = null);
        string GetDefaultCacheDirectory();
        string GetCacheDirectory();
    }
}
